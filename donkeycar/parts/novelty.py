"""
Feature-space novelty (out-of-distribution) detection.

MC-Dropout variance (``donkeycar.parts.mc_dropout``) measures whether the
model's dropout sub-networks *agree* with each other -- an ambiguity signal.
It does NOT measure whether the current input looks anything like the data
the model was trained on: a scene the network has never seen can still get a
confident, low-variance (wrong) answer if its sub-networks happen to agree.
This module adds the complementary signal: how far the current frame's
internal feature representation is from the training distribution, via
Mahalanobis distance on the model's own learned features.

Two independent consumers share the math and calibration schema here:
  * ``FeatureNoveltyDetector`` -- a live Part computing one scalar novelty
    score per frame, cheap enough to run every drive-loop iteration
    (a single deterministic forward pass to the ``dense_2`` layer, no
    MC-Dropout stochasticity needed since this isn't measuring agreement).
  * ``donkeycar.parts.gradcam_uncertainty`` -- reuses ``mahalanobis_diag``
    directly on ``conv2d_5`` features (pooled across space) to render a
    spatial "where is this unfamiliar" heatmap for offline analysis.

Caveats (read before trusting the numbers):
  * **Diagonal covariance only.** Feature dimensions are treated as
    independent (per-dimension mean/variance, not a full covariance
    matrix). Cheap and numerically robust, but ignores correlations
    between features -- a combination of individually-normal feature
    values could still be jointly unusual and go undetected.
  * **Position-independent spatial pooling.** The spatial (conv2d_5)
    statistics pool every grid location across every calibration frame
    into one distribution, discarding *where* in the image a feature
    normally appears. This is simple and cheap, but can mask novelty in
    channels that have legitimate positional structure (e.g. a channel
    that reliably fires near the top of the frame for sky/horizon) --
    pooling folds that within-frame positional swing into the channel's
    "normal" variance, raising the bar for detecting genuine novelty
    there. A per-location (or coarse-grid) statistics model would fix
    this at the cost of real complexity; not done here.
  * **Dead-dimension handling.** A near-constant (e.g. dead ReLU) feature
    dimension has ~zero variance; naively dividing by it would let a
    single dimension dominate the whole distance. Dimensions are flagged
    "inactive" at calibration time (via a variance floor, relative to the
    typical variance across all dimensions) and excluded from the sum
    entirely, rather than papering over it with a global epsilon alone.
  * **Complementary, not a replacement, for MC-Dropout confidence.** A
    frame can be low-novelty (looks like training data) yet still
    high-variance (the model is internally torn about what to do), or
    the reverse (novel-looking, but the sub-networks still happen to
    agree). Read both signals together, not as substitutes.
  * **Model scope**: like the rest of this toolkit, only the default
    linear architecture (``KerasLinear``) with a callable Keras
    interpreter is supported -- not TFLite/TensorRT (their graphs cannot
    be reached by name to build the feature sub-model), and not other
    architectures (this module assumes layers named ``dense_2`` /
    ``conv2d_5`` from the default linear model).
"""
import logging

import numpy as np
import tensorflow as tf

from donkeycar.utils import normalize_image

logger = logging.getLogger(__name__)


def mahalanobis_diag(vec, mean, var, active_dims=None, eps=1e-6):
    """
    Diagonal-covariance Mahalanobis distance: sum((x_i - mean_i)^2 / (var_i + eps))
    over the active dimensions only.

    :param vec:         (..., d) array -- a single feature vector or a batch/
                        grid of them (any leading shape), last axis = features.
    :param mean:        (d,) array, per-dimension mean of the reference
                        (training) distribution.
    :param var:         (d,) array, per-dimension variance of the reference
                        distribution.
    :param active_dims: optional (d,) boolean array. Dimensions where False
                        are excluded from the sum entirely (see module
                        docstring on dead-dimension handling). Defaults to
                        all dimensions active.
    :param eps:         small constant added to variance for numerical safety
                        on the *active* dimensions (dead/near-zero-variance
                        dimensions should be excluded via active_dims instead
                        of relying on eps alone).
    :return:            scalar distance, or (...,) array of distances if vec
                        has leading batch/grid dimensions.
    """
    vec = np.asarray(vec, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)
    if active_dims is None:
        active_dims = np.ones(mean.shape[0], dtype=bool)
    else:
        active_dims = np.asarray(active_dims, dtype=bool)

    diff_sq = (vec - mean) ** 2
    denom = var + eps
    terms = diff_sq / denom
    terms = terms * active_dims       # zero out inactive dimensions
    return terms.sum(axis=-1)


def fit_diagonal_gaussian(feature_matrix, min_var_frac=1e-3, abs_eps=1e-6):
    """
    Fit a diagonal Gaussian (per-dimension mean/variance) to a matrix of
    feature vectors, and flag dimensions too close to constant to trust.

    :param feature_matrix: (n_samples, d) array.
    :param min_var_frac:   a dimension is flagged "inactive" if its variance
                           is below `median(var) * min_var_frac` -- a floor
                           *relative* to this feature space's own typical
                           variance (an absolute constant would be the wrong
                           scale for different layers/models; see module
                           docstring).
    :param abs_eps:        absolute floor under the relative one, in case
                           every dimension happens to have tiny variance
                           (e.g. a degenerate/tiny calibration set).
    :return:               dict with 'mean', 'var', 'active_dims' (all lists,
                           JSON-serialisable), and 'eps' (float) -- the eps
                           to use for the active dimensions at scoring time.
    """
    feature_matrix = np.asarray(feature_matrix, dtype=np.float64)
    mean = feature_matrix.mean(axis=0)
    var = feature_matrix.var(axis=0)

    median_var = float(np.median(var))
    floor = max(median_var * min_var_frac, abs_eps)
    active_dims = var >= floor
    n_inactive = int((~active_dims).sum())
    if n_inactive:
        logger.info(f'fit_diagonal_gaussian: {n_inactive}/{len(var)} '
                    f'dimensions flagged inactive (variance < {floor:.3g}), '
                    f'excluded from distance calculation.')

    return {
        'mean': mean.tolist(),
        'var': var.tolist(),
        'active_dims': active_dims.tolist(),
        'eps': floor,
    }


class FeatureNoveltyDetector:
    """
    DonkeyCar Part. Computes a live "novelty" score for each frame: how far
    the frame's ``dense_2`` feature vector is from the distribution seen
    during training, via diagonal-covariance Mahalanobis distance.

    Unlike ``MCDropoutConfidence``, this is a pure auxiliary observer -- it
    does NOT produce steering/throttle, and rides alongside whichever part
    already drives the car (the plain pilot or the MC-Dropout pilot). A
    single deterministic forward pass per frame is enough (no dropout
    stochasticity needed -- novelty is a distance-from-training-distribution
    measure, not an agreement measure), so this is cheap enough to run every
    drive-loop iteration independently of whether MC-Dropout confidence is
    also enabled.

    Run signature::

        score, raw_distance, smoothed_distance = part.run(img_arr)

    ``score`` is a 0-100 novelty % derived from the calibration file (see
    ``donkeycar.parts.mc_calibrate``), or ``None`` if the model has no
    calibration, or an older calibration made before novelty stats existed.

    :param pilot:            a loaded KerasPilot (KerasLinear). Its
                             underlying Keras model is reused directly -- no
                             separate load.
    :param calibration_path: path to a ``<model>.calib.json`` produced by
                             ``donkeycar.parts.mc_calibrate``. Optional.
    :param alpha:            EMA smoothing factor in [0, 1] for the raw
                             distance -- same role as MCDropoutConfidence's.
    """

    def __init__(self, pilot, calibration_path=None, alpha=0.2):
        self.alpha = float(alpha)
        self.smoothed_distance = None

        interpreter = getattr(pilot, 'interpreter', None)
        self.model = getattr(interpreter, 'model', None)
        if self.model is None or not callable(self.model):
            raise ValueError(
                'FeatureNoveltyDetector requires a Keras pilot with a '
                'callable model (the default KerasInterpreter). '
                'TFLite/TensorRT interpreters are not supported.')

        self.feat_model = tf.keras.Model(
            self.model.inputs, self.model.get_layer('dense_2').output)

        self.calibration = None
        if calibration_path:
            try:
                from donkeycar.parts.mc_calibrate import load_calibration
                calib = load_calibration(calibration_path)
                self.calibration = calib.get('novelty_global')
                if self.calibration is None:
                    logger.warning(
                        f'FeatureNoveltyDetector: {calibration_path} has no '
                        f'"novelty_global" stats (older calibration?); '
                        f'novelty score will be unavailable -- re-run '
                        f'mc_calibrate to add novelty stats.')
                else:
                    logger.info(f'FeatureNoveltyDetector: loaded novelty '
                                f'calibration from {calibration_path}')
            except Exception as e:
                logger.warning(f'FeatureNoveltyDetector: could not load '
                               f'calibration {calibration_path} ({e}); '
                               f'novelty score will be unavailable.')

    def run(self, img_arr):
        if img_arr is None:
            return self._score(), 0.0, (self.smoothed_distance or 0.0)

        norm = normalize_image(img_arr).astype(np.float32)
        feat = self.feat_model(norm[np.newaxis, ...], training=False)
        feat = np.asarray(feat)[0]

        if self.calibration is not None:
            mean = np.array(self.calibration['mean'])
            var = np.array(self.calibration['var'])
            active = np.array(self.calibration['active_dims'])
            eps = self.calibration['eps']
            raw_distance = float(
                mahalanobis_diag(feat, mean, var, active, eps))
        else:
            raw_distance = 0.0

        if self.smoothed_distance is None:
            self.smoothed_distance = raw_distance
        else:
            self.smoothed_distance = (
                self.alpha * raw_distance
                + (1.0 - self.alpha) * self.smoothed_distance)

        return self._score(), raw_distance, self.smoothed_distance

    def _score(self):
        if self.calibration is None or self.smoothed_distance is None:
            return None
        from donkeycar.parts.mc_calibrate import novelty_distance_to_score
        return novelty_distance_to_score(self.smoothed_distance,
                                         self.calibration)

    def shutdown(self):
        pass
