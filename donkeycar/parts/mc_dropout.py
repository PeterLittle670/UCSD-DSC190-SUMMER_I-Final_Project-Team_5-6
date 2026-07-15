"""
MC-Dropout confidence estimation for DonkeyCar.

This part wraps an already-loaded KerasLinear pilot and, on every drive-loop
iteration, runs the model N times with dropout left *active* (Monte-Carlo
Dropout). The spread of the N steering predictions is used as an uncertainty
signal:

  * ``mean_steering`` / ``mean_throttle`` -- the averaged prediction. This
    becomes the command sent to the car (same behaviour as a normal single
    pass, just averaged over the stochastic sub-networks).
  * ``raw_variance``   -- variance of the N steering outputs for this frame.
  * ``smoothed_variance`` -- exponential moving average of ``raw_variance``
    over time, to damp frame-to-frame flicker.

IMPORTANT (scope / caveats):
  * This is a *relative*, per-model uncertainty signal, NOT a formally
    calibrated Bayesian probability. Turning ``smoothed_variance`` into a
    human-readable "confidence %" requires the separate calibration step
    (percentile thresholds collected from a representative drive).
  * Only the default linear architecture (``KerasLinear`` /
    ``default_n_linear``) is supported in v1. That model contains Dropout
    layers and *no* BatchNormalization, which is why calling it with
    ``training=True`` cleanly re-enables dropout without side effects.
  * MC-Dropout needs the full Keras interpreter. The TFLite / TensorRT
    inference paths cannot keep dropout stochastic and are not supported.
"""
import logging
import time

import numpy as np
import tensorflow as tf

from donkeycar.utils import normalize_image

logger = logging.getLogger(__name__)


class MCDropoutConfidence:
    """
    DonkeyCar Part. Drop-in replacement for a KerasLinear inference part that
    additionally emits an uncertainty signal.

    Run signature::

        angle, throttle, confidence, raw_variance, smoothed_variance = \
            part.run(img_arr)

    ``confidence`` is a 0-100 %% derived from the calibration file, or None if
    the model has no calibration yet.

    :param pilot:       a loaded KerasPilot (KerasLinear). Its underlying Keras
                        model is reused directly -- we do not reload weights.
    :param num_passes:  number of stochastic forward passes (N). Wired to
                        ``cfg.MC_DROPOUT_PASSES``.
    :param alpha:       EMA smoothing factor in [0, 1]. Higher = more
                        responsive, lower = smoother. Wired to
                        ``cfg.MC_DROPOUT_ALPHA``.
    :param calibration_path: path to a ``<model>.calib.json`` produced by
                        ``donkeycar.parts.mc_calibrate``. Optional.
    """

    def __init__(self, pilot, num_passes=15, alpha=0.2, calibration_path=None,
                 interval=0.0):
        self.pilot = pilot
        self.num_passes = int(num_passes)
        self.alpha = float(alpha)
        self.smoothed_variance = None  # lazily initialised on first frame

        # Minimum seconds between stochastic (N-pass) uncertainty updates.
        # 0 = update every frame. Between updates, a cheap single
        # deterministic pass still produces a fresh steering command each
        # loop iteration; only the uncertainty numbers are held.
        # NOTE: the EMA alpha applies per *update*, so with a longer interval
        # the variance is smoothed over fewer, more spaced-out samples.
        self.interval = float(interval)
        self.last_mc_time = None
        self.raw_variance = 0.0

        # Optional calibration: maps smoothed variance -> displayed confidence %.
        # Without it, the part still emits variance but confidence is None.
        self.calibration = None
        if calibration_path:
            try:
                from donkeycar.parts.mc_calibrate import load_calibration
                self.calibration = load_calibration(calibration_path)
                logger.info(f'MCDropoutConfidence: loaded calibration from '
                            f'{calibration_path}')
            except Exception as e:
                logger.warning(f'MCDropoutConfidence: could not load '
                               f'calibration {calibration_path} ({e}); '
                               f'confidence %% will be unavailable.')

        # Reach the raw Keras model. The interpreter must be the plain Keras
        # one -- TFLite/TensorRT interpreters have no callable Keras model and
        # cannot keep dropout stochastic.
        interpreter = getattr(pilot, 'interpreter', None)
        self.model = getattr(interpreter, 'model', None)
        if self.model is None or not callable(self.model):
            raise ValueError(
                'MCDropoutConfidence requires a Keras pilot with a callable '
                'model (the default KerasInterpreter). TFLite/TensorRT '
                'interpreters are not supported.')
        self.input_keys = list(interpreter.input_keys)

        # Sanity check: warn loudly if the model has no dropout, because then
        # every "stochastic" pass is identical and the signal is meaningless.
        n_dropout = sum(1 for layer in self.model.layers
                        if 'dropout' in layer.__class__.__name__.lower())
        if n_dropout == 0:
            logger.warning(
                'MCDropoutConfidence: the loaded model has NO dropout layers. '
                'The uncertainty signal will be identically zero. Retrain '
                'with a dropout-bearing architecture (e.g. the default '
                'linear model).')
        else:
            logger.info(f'MCDropoutConfidence: found {n_dropout} dropout '
                        f'layer(s); running N={self.num_passes} stochastic '
                        f'passes per frame, EMA alpha={self.alpha}.')

    def _stochastic_forward(self, img_arr, other_arr):
        """
        Run N forward passes with dropout active, batched into a single call.

        We replicate the input N times along the batch axis and do ONE
        ``model(..., training=True)`` call. Keras samples an independent
        dropout mask per batch element, so the N rows of the output are N
        genuine Monte-Carlo samples -- far cheaper than an N-iteration Python
        loop (~5x faster for N=15 in local timing).

        :return: (angle_samples, throttle_samples) as 1-D np arrays of len N.
        """
        norm_img = normalize_image(img_arr).astype(np.float32)
        # Batch of N identical images: shape (N, H, W, C).
        img_batch = np.repeat(norm_img[np.newaxis, ...], self.num_passes, axis=0)

        values = [img_batch]
        for arr in other_arr:
            a = np.asarray(arr, dtype=np.float32)
            values.append(np.repeat(a[np.newaxis, ...], self.num_passes, axis=0))

        input_dict = {k: tf.convert_to_tensor(v)
                      for k, v in zip(self.input_keys, values)}

        outputs = self.model(input_dict, training=True)
        # Linear model returns a list [angle (N,1), throttle (N,1)].
        angle_samples = np.asarray(outputs[0]).reshape(-1)
        throttle_samples = np.asarray(outputs[1]).reshape(-1)
        return angle_samples, throttle_samples

    def _confidence(self):
        """Displayed confidence % from the smoothed variance, or None if the
        model is uncalibrated."""
        if self.calibration is None or self.smoothed_variance is None:
            return None
        from donkeycar.parts.mc_calibrate import variance_to_confidence
        return variance_to_confidence(self.smoothed_variance, self.calibration)

    def run(self, img_arr, *other_arr):
        if img_arr is None:
            # Nothing to predict on yet; emit a neutral default. Downstream
            # code treats None gracefully.
            return 0.0, 0.0, self._confidence(), 0.0, \
                (self.smoothed_variance or 0.0)

        # Rate limiting: between uncertainty updates, drive on a cheap single
        # deterministic pass and hold the last uncertainty values.
        now = time.time()
        if (self.interval > 0 and self.last_mc_time is not None
                and (now - self.last_mc_time) < self.interval):
            angle, throttle = self.pilot.run(img_arr, *other_arr)
            return (float(angle), float(throttle), self._confidence(),
                    self.raw_variance, (self.smoothed_variance or 0.0))
        self.last_mc_time = now

        angle_samples, throttle_samples = \
            self._stochastic_forward(img_arr, other_arr)

        mean_steering = float(np.mean(angle_samples))
        mean_throttle = float(np.mean(throttle_samples))
        raw_variance = float(np.var(angle_samples))
        self.raw_variance = raw_variance

        # Exponential moving average of the variance.
        if self.smoothed_variance is None:
            self.smoothed_variance = raw_variance
        else:
            self.smoothed_variance = (
                self.alpha * raw_variance
                + (1.0 - self.alpha) * self.smoothed_variance)

        return (mean_steering, mean_throttle, self._confidence(),
                raw_variance, self.smoothed_variance)

    def shutdown(self):
        pass


class ConfidenceThrottleScaler:
    """
    Feature 2: scale the pilot throttle down when the MC-Dropout confidence
    drops. The philosophy is "hesitate, don't guess": this only ever reduces
    throttle magnitude and NEVER touches steering.

    Behaviour (all thresholds configurable):
      * confidence >= reduced_threshold  -> full throttle, unchanged.
      * critical_threshold <= confidence < reduced_threshold -> "reduced" tier:
        throttle is linearly scaled from full (at reduced_threshold) down to
        ``min_scale`` (at critical_threshold). It never drops below
        ``min_scale`` from this signal alone.
      * confidence < critical_threshold -> "critical" tier: throttle held at
        ``min_scale``, and if the critical tier is *sustained* for
        ``stop_duration`` seconds, throttle is forced to zero (stop).

    Passthrough (no change to throttle) when:
      * confidence is None (feature disabled upstream / model uncalibrated), or
      * throttle is None.

    Run signature::

        scaled_throttle = part.run(throttle, confidence)

    Intended to be added with ``run_condition='run_pilot'`` so it only affects
    autopilot throttle, giving zero behaviour change to manual driving.
    """

    def __init__(self, reduced_threshold=65.0, critical_threshold=25.0,
                 min_scale=0.4, stop_duration=1.0):
        self.reduced_threshold = float(reduced_threshold)
        self.critical_threshold = float(critical_threshold)
        self.min_scale = float(min_scale)
        self.stop_duration = float(stop_duration)
        self.critical_since = None   # timestamp critical tier began, else None
        self._warned_no_conf = False
        logger.info(f'ConfidenceThrottleScaler: reduced<{reduced_threshold}%, '
                    f'critical<{critical_threshold}%, min_scale={min_scale}, '
                    f'stop_after={stop_duration}s')

    def run(self, throttle, confidence):
        if throttle is None:
            return throttle
        if confidence is None:
            # No usable signal -> do not interfere with driving.
            if not self._warned_no_conf:
                logger.warning('ConfidenceThrottleScaler: confidence is None '
                               '(is --uncertainty on and the model '
                               'calibrated?); throttle passed through '
                               'unchanged.')
                self._warned_no_conf = True
            self.critical_since = None
            return throttle

        # Normal tier: full throttle.
        if confidence >= self.reduced_threshold:
            self.critical_since = None
            return throttle

        # Reduced tier: linearly interpolate scale from 1.0 down to min_scale.
        if confidence >= self.critical_threshold:
            self.critical_since = None
            span = self.reduced_threshold - self.critical_threshold
            frac = (confidence - self.critical_threshold) / span if span > 0 \
                else 0.0
            scale = self.min_scale + (1.0 - self.min_scale) * frac
            return throttle * scale

        # Critical tier: hold at min_scale, and force a stop if sustained.
        now = time.time()
        if self.critical_since is None:
            self.critical_since = now
        if now - self.critical_since >= self.stop_duration:
            return 0.0
        return throttle * self.min_scale

    def shutdown(self):
        pass
