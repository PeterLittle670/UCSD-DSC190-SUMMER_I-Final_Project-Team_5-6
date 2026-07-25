"""
Calibration for the MC-Dropout uncertainty signal.

The raw / smoothed steering variance produced by
``donkeycar.parts.mc_dropout.MCDropoutConfidence`` is not interpretable on its
own -- its scale depends on the trained model and the track/domain. This module
turns that variance into a *relative, per-model* "confidence %" by:

  1. Replaying a recorded tub through the MC-Dropout part to collect a
     distribution of ``smoothed_variance`` values.
  2. Computing percentile thresholds (default 50 / 80 / 97) from that
     distribution.
  3. Building a monotonic piecewise-linear map from variance -> confidence %,
     anchored on those percentiles, and saving it next to the model as
     ``<model>.calib.json``.

At drive time, ``variance_to_confidence()`` maps the live smoothed variance to
a displayed percentage using those anchors.

WHICH TUB you calibrate against matters. The whole idea is "how does this
frame compare to the training distribution" -- so calibration should be run
against the tub(s) the model was actually trained on, not e.g. a later
inference/test drive (that would make "novelty" mean "unlike this other
drive" rather than "unlike what the model learned", and would make
"confidence %" relative to the wrong baseline entirely). ``find_training_tubs()``
looks this up automatically from DonkeyCar's own training-run database when
available; see its docstring and the GUI launcher's auto-calibrate feature
(``donkeycar.parts.uncertainty_viewer``) for how this is used in practice.

IMPORTANT: this is a relative, per-model calibration, NOT a formally calibrated
Bayesian probability. A displayed "90%" means "this frame's uncertainty is low
*relative to this model's own baseline drive*", nothing more.

The same replay pass also fits calibration statistics for the complementary
feature-space novelty (out-of-distribution) signal -- see
``donkeycar.parts.novelty``: per-frame ``dense_2``/``conv2d_5`` features are
collected alongside the variance, a diagonal Gaussian is fit to each, and
percentile-anchored novelty-score maps are stored as ``novelty_global``/
``novelty_spatial`` blocks in the same ``calib.json`` (purely additive --
older calibrations without these keys still work, novelty just displays as
unavailable). ``novelty_distance_to_score()`` is the novelty analogue of
``variance_to_confidence()``.
"""
import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

# Percentiles used to define the confidence tiers. p50 (median) anchors "high
# confidence", p80 the normal/reduced boundary, p97 the reduced/critical one.
DEFAULT_PERCENTILES = (50, 80, 97)

# Confidence % assigned at each anchor (plus the distribution min/max ends).
# Monotonically decreasing: low variance -> high confidence.
_ANCHOR_CONFIDENCE = {
    'min': 99.0,
    'p50': 90.0,
    'p80': 65.0,
    'p97': 25.0,
    'max': 5.0,
}

# Novelty % assigned at each Mahalanobis-distance percentile anchor.
# Monotonically INCREASING (opposite direction from confidence): low
# distance from the training distribution -> low novelty (familiar), high
# distance -> high novelty (unfamiliar).
_ANCHOR_NOVELTY = {
    'min': 2.0,
    'p50': 10.0,
    'p80': 45.0,
    'p97': 80.0,
    'max': 98.0,
}

# Frames per OOD-encoder forward batch. The extractor resizes and converts the
# whole list it is handed into one float32 array, so an unchunked call over a
# large tub allocates far more than the uint8 frames themselves.
_OOD_CHUNK = 256


def build_calibration(smoothed_variances, num_passes, alpha,
                      percentiles=DEFAULT_PERCENTILES, model_path=None,
                      dense2_features=None, conv5_features=None,
                      tta_variances=None, tta_num_samples=None,
                      tta_strength=None, tta_alpha=None,
                      ood_features=None, ood_encoder=None, ood_input_size=None,
                      ood_alpha=1.0, ood_encoder_file=None,
                      augmentation_info=None):
    """
    Turn a collected distribution of smoothed variances into a calibration
    dict (JSON-serialisable). Optionally also fits novelty-detection
    (Mahalanobis distance) statistics from per-frame feature vectors:

    :param dense2_features: optional (n_frames, 50) array of dense_2 layer
                            outputs, one per calibration frame -- fits the
                            *global* (whole-image) novelty statistics.
    :param conv5_features:  optional (n_frames, h, w, 64) array of conv2d_5
                            layer outputs -- fits the *spatial* novelty
                            statistics, pooling every grid location across
                            every frame into one distribution (see
                            donkeycar.parts.novelty module docstring for the
                            position-independence trade-off).
    :param tta_variances:   optional list of per-frame test-time-augmentation
                            steering variances -- fits the ``tta`` stability
                            calibration block (see donkeycar.parts.tta).
    :param augmentation_info: optional provenance dict recording whether the
                            novelty baselines were widened with training-style
                            augmented frames (see ``calibrate_from_tub``).
                            Stored as-is under the ``augmentation`` key so a
                            calibration can be told apart from a clean-only
                            one after the fact; omitted when None, keeping
                            older calibration files valid.
    """
    v = np.asarray(smoothed_variances, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError('No variance samples collected for calibration.')

    p50, p80, p97 = (float(np.percentile(v, p)) for p in percentiles)
    vmin, vmax = float(v.min()), float(v.max())

    # Build monotonic anchor points (variance ascending) for np.interp. We
    # nudge duplicates so the x-values are strictly increasing.
    xs = [vmin, p50, p80, p97, vmax]
    ys = [_ANCHOR_CONFIDENCE['min'], _ANCHOR_CONFIDENCE['p50'],
          _ANCHOR_CONFIDENCE['p80'], _ANCHOR_CONFIDENCE['p97'],
          _ANCHOR_CONFIDENCE['max']]
    xs = _make_strictly_increasing(xs)

    calib = {
        'model': os.path.basename(model_path) if model_path else None,
        'num_passes': int(num_passes),
        'alpha': float(alpha),
        'n_frames': int(v.size),
        'percentiles': {'p50': p50, 'p80': p80, 'p97': p97},
        'variance_stats': {'min': vmin, 'max': vmax,
                           'mean': float(v.mean()), 'std': float(v.std())},
        'tiers': {
            'normal_below': p80,
            'reduced_below': p97,
            'critical_at_or_above': p97,
        },
        'confidence_anchors': {'variance': xs, 'confidence': ys},
        'note': ('Relative per-model uncertainty calibration, NOT a formally '
                 'calibrated probability.'),
    }

    if dense2_features is not None:
        calib['novelty_global'] = _build_novelty_block(
            np.asarray(dense2_features, dtype=np.float64), percentiles)
    if conv5_features is not None:
        conv5 = np.asarray(conv5_features, dtype=np.float64)
        # Pool every spatial location across every frame into ONE
        # distribution -- position-independent, see novelty.py docstring.
        pooled = conv5.reshape(-1, conv5.shape[-1])
        calib['novelty_spatial'] = _build_novelty_block(pooled, percentiles)
    if tta_variances is not None:
        calib['tta'] = _build_tta_block(
            tta_variances, tta_num_samples, tta_strength, tta_alpha,
            percentiles)
    if ood_features is not None:
        calib['novelty_ood'] = _build_ood_block(
            ood_features, ood_encoder, ood_input_size, ood_alpha,
            ood_encoder_file, percentiles)
    if augmentation_info is not None:
        calib['augmentation'] = augmentation_info

    return calib


def _build_tta_block(tta_variances, num_samples, strength, alpha,
                     percentiles=DEFAULT_PERCENTILES):
    """
    Build the TTA (test-time augmentation) stability calibration block from a
    distribution of per-frame TTA steering variances. Mirrors the confidence
    calibration exactly (descending variance -> stability % map), with its own
    anchors, so ``tta_variance_to_stability`` reads standing within this
    model's own TTA-variance distribution.
    """
    v = np.asarray(tta_variances, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError('No TTA variance samples collected for calibration.')

    p50, p80, p97 = (float(np.percentile(v, p)) for p in percentiles)
    vmin, vmax = float(v.min()), float(v.max())
    xs = _make_strictly_increasing([vmin, p50, p80, p97, vmax])
    # Reuse the confidence anchor levels: low variance -> high stability.
    ys = [_ANCHOR_CONFIDENCE['min'], _ANCHOR_CONFIDENCE['p50'],
          _ANCHOR_CONFIDENCE['p80'], _ANCHOR_CONFIDENCE['p97'],
          _ANCHOR_CONFIDENCE['max']]

    return {
        'num_samples': int(num_samples),
        'strength': float(strength),
        'alpha': float(alpha),
        'n_frames': int(v.size),
        'percentiles': {'p50': p50, 'p80': p80, 'p97': p97},
        'variance_stats': {'min': vmin, 'max': vmax,
                           'mean': float(v.mean()), 'std': float(v.std())},
        'stability_anchors': {'variance': xs, 'stability': ys},
        'note': ('Robustness of the prediction to photometric input '
                 'perturbation. Relative per-model signal, not a probability.'),
    }


def _build_ood_block(ood_features, encoder_name, input_size, alpha,
                     encoder_file, percentiles=DEFAULT_PERCENTILES):
    """
    Build the live-novelty calibration block from GENERIC-ENCODER features
    (see donkeycar.parts.ood for why this replaces the task-collapsed
    steering-model features). Reuses the same diagonal-Gaussian /
    percentile-anchor machinery as the legacy novelty block, plus the encoder
    metadata the drive-time detector needs to rebuild/validate the matching
    extractor.
    """
    block = _build_novelty_block(
        np.asarray(ood_features, dtype=np.float64), percentiles)
    block.update({
        'encoder': encoder_name,
        'input_size': int(input_size),
        'alpha': float(alpha),
        'feat_dim': int(np.asarray(ood_features).shape[1]),
        'encoder_file': encoder_file,
    })
    return block


def _build_novelty_block(feature_matrix, percentiles=DEFAULT_PERCENTILES):
    """
    Fit a diagonal Gaussian to `feature_matrix` (n_samples, d) and build a
    percentile-anchored distance -> novelty-score map, mirroring the
    confidence calibration's structure but ascending (low distance -> low
    score, high distance -> high score).
    """
    from donkeycar.parts.novelty import fit_diagonal_gaussian, mahalanobis_diag

    stats = fit_diagonal_gaussian(feature_matrix)
    mean = np.array(stats['mean'])
    var = np.array(stats['var'])
    active = np.array(stats['active_dims'])
    eps = stats['eps']

    distances = mahalanobis_diag(feature_matrix, mean, var, active, eps)
    p50, p80, p97 = (float(np.percentile(distances, p)) for p in percentiles)
    dmin, dmax = float(distances.min()), float(distances.max())

    xs = _make_strictly_increasing([dmin, p50, p80, p97, dmax])
    ys = [_ANCHOR_NOVELTY['min'], _ANCHOR_NOVELTY['p50'],
          _ANCHOR_NOVELTY['p80'], _ANCHOR_NOVELTY['p97'],
          _ANCHOR_NOVELTY['max']]

    return {
        'mean': stats['mean'],
        'var': stats['var'],
        'active_dims': stats['active_dims'],
        'eps': eps,
        'distance_stats': {'min': dmin, 'max': dmax,
                           'mean': float(distances.mean()),
                           'std': float(distances.std())},
        'score_anchors': {'distance': xs, 'score': ys},
    }


def _make_strictly_increasing(xs, eps=1e-9):
    """np.interp requires strictly increasing x; nudge any flat/duplicate x."""
    out = list(xs)
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + eps
    return out


def variance_to_confidence(variance, calib):
    """
    Map a (smoothed) variance to a displayed confidence % in [min..max anchor],
    using the calibration's anchor points. Monotonically decreasing.
    """
    anchors = calib['confidence_anchors']
    xs = anchors['variance']
    ys = anchors['confidence']
    # np.interp needs increasing xs and clamps outside the range automatically.
    # ys is decreasing, which np.interp handles fine.
    return float(np.interp(variance, xs, ys))


def novelty_distance_to_score(distance, novelty_calib_block):
    """
    Map a Mahalanobis distance to a displayed novelty % in [min..max anchor],
    using a novelty calibration block (calib['novelty_global'] or
    calib['novelty_spatial']). Monotonically increasing: low distance (looks
    like training data) -> low novelty %, high distance -> high novelty %.
    """
    anchors = novelty_calib_block['score_anchors']
    return float(np.interp(distance, anchors['distance'], anchors['score']))


def tta_variance_to_stability(variance, tta_calib_block):
    """
    Map a (smoothed) TTA steering variance to a displayed stability % using the
    tta block's anchors (calib['tta']). Same descending-interp machinery as
    ``variance_to_confidence`` -- low variance (prediction is robust to input
    perturbation) -> high stability %, high variance -> low stability %.
    """
    anchors = tta_calib_block['stability_anchors']
    return float(np.interp(variance, anchors['variance'], anchors['stability']))


def find_training_tubs(cfg, model_path):
    """
    Look up which tub(s) `model_path` was actually trained on, via
    DonkeyCar's own ``PilotDatabase`` (``<MODELS_PATH>/database.json``,
    written automatically by ``donkey train``). Returns a list of tub paths,
    or None if no matching entry exists (the model wasn't trained via
    ``donkey train``, or the database has since been moved/deleted/pruned).

    This matters because calibration is supposed to establish "what does
    normal (training) data look like" -- calibrating against some OTHER tub
    (e.g. a post-training inference/test drive) undermines the whole point,
    especially for novelty detection: a frame only ever looks "normal"
    relative to whatever tub it happens to be compared against. If the
    calibration tub isn't the training tub, "novelty" quietly becomes
    "unlike this other drive" instead of "unlike what the model learned."
    """
    from donkeycar.pipeline.database import PilotDatabase
    try:
        db = PilotDatabase(cfg)
    except Exception as e:
        logger.debug(f'find_training_tubs: could not open PilotDatabase '
                     f'({e})')
        return None
    name = os.path.splitext(os.path.basename(model_path))[0]
    entry = db.get_entry(name)
    if entry is None or not entry.get('Tubs'):
        return None
    return [t.strip() for t in entry['Tubs'].split(',') if t.strip()]


def default_calib_path(model_path):
    """Calibration file sits next to the model: <model>.calib.json"""
    return os.path.splitext(model_path)[0] + '.calib.json'


def default_encoder_path(model_path):
    """OOD-encoder sidecar sits next to the model:
    <model>.novelty_encoder.h5. Saved at calibration time so a Pi driving
    offline never needs to download the encoder's ImageNet weights."""
    return os.path.splitext(model_path)[0] + '.novelty_encoder.h5'


def save_calibration(calib, path):
    with open(path, 'w') as f:
        json.dump(calib, f, indent=2)
    logger.info(f'Saved calibration to {path}')


def load_calibration(path):
    with open(path) as f:
        return json.load(f)


def calibrate_from_tub(cfg, tub_paths, model_path, num_passes=None,
                       alpha=None, limit=None, percentiles=DEFAULT_PERCENTILES,
                       out_path=None, progress_callback=None, augment=None):
    """
    Replay a tub through the MC-Dropout part, collect smoothed variances plus
    (for novelty detection) per-frame dense_2/conv2d_5 feature vectors, build
    and save a calibration file. Returns (calib_dict, out_path).

    :param progress_callback: optional ``callback(stage, current, total)``,
                              called during the per-frame replay
                              (stage='calibrate') and around the OOD encoder's
                              batched extraction step (stage='calibrate_ood').
                              Used by the GUI launcher's auto-calibrate step to
                              show progress; harmless to omit for CLI use.
    :param augment:           whether to widen the *novelty* baselines with
                              training-style augmented frames (see below).
                              ``None`` (default) reads
                              ``XAI_CALIBRATE_WITH_AUGMENTATIONS`` from cfg.

    AUGMENTATION-AWARE NOVELTY BASELINES
    ------------------------------------
    Novelty is a distance-from-training-distribution measure, so its baseline
    has to describe the inputs the model was actually *trained* on. Training
    applies ``cfg.AUGMENTATIONS`` (shadow / gamma / noise / brightness / blur)
    only inside ``BatchSequence.image_processor`` when ``is_train=True``, so a
    baseline fit on raw tub frames alone describes a *narrower* input
    distribution than the model was trained for. A perfectly ordinary
    shadowed or dim frame then scores as highly novel at runtime and -- with
    ``XAI_THROTTLE_SCALING_ENABLED`` -- needlessly throttles the car down in
    exactly the conditions augmentation was added to survive.

    Unlike the confidence/TTA percentiles (which are re-collected through the
    retrained model every time and so rescale themselves), the novelty
    baseline cannot self-correct: the live signal measures distance in a
    *frozen* ImageNet encoder's feature space (see ``donkeycar.parts.ood``),
    which never sees a training epoch. So it is fixed here instead.

    When ``cfg.AUGMENTATIONS`` is non-empty, a strided subset of frames is
    additionally passed through the same ``ImageAugmentation`` pipeline used
    in training, ``XAI_CALIBRATE_AUG_PASSES`` times each (re-randomised per
    pass, since each ``run()`` samples fresh parameters), and those feature
    vectors are pooled with the clean ones before fitting. The resulting
    Gaussian covers the training-time input envelope rather than just its
    clean centre.

    Deliberately scoped to the novelty statistics only:

      * The MC-Dropout variance and TTA replays are left untouched. Both parts
        carry an EMA over a *time-ordered* frame sequence; splicing augmented
        frames into that stream would corrupt the smoothing. Their outputs are
        therefore bit-for-bit identical to before this option existed.
      * With no ``AUGMENTATIONS`` configured the augmenter is skipped
        entirely, so calibrations for un-augmented models are also unchanged.
    """
    # Imported here so this module is cheap to import without TF.
    from donkeycar.parts.keras import KerasLinear
    from donkeycar.parts.mc_dropout import MCDropoutConfidence
    from donkeycar.pipeline.types import TubDataset
    from donkeycar.utils import normalize_image
    import tensorflow as tf

    num_passes = num_passes if num_passes is not None \
        else getattr(cfg, 'XAI_CONFIDENCE_PASSES', 15)
    alpha = alpha if alpha is not None \
        else getattr(cfg, 'XAI_CONFIDENCE_ALPHA', 0.2)

    logger.info(f'Loading model {model_path}')
    pilot = KerasLinear()
    pilot.load(model_path)
    part = MCDropoutConfidence(pilot, num_passes=num_passes, alpha=alpha)

    # Optional TTA (test-time augmentation) stability calibration -- only when
    # enabled, since it adds M deterministic forward passes per frame. Reuses
    # the same loaded pilot; collects per-frame TTA steering variance below.
    tta_enabled = getattr(cfg, 'XAI_TTA_ENABLED', False)
    tta_part = None
    if tta_enabled:
        from donkeycar.parts.tta import TTAStabilityDetector
        tta_samples = getattr(cfg, 'XAI_TTA_SAMPLES', 8)
        tta_strength = getattr(cfg, 'XAI_TTA_STRENGTH', 0.2)
        tta_alpha = getattr(cfg, 'XAI_TTA_ALPHA', 0.2)
        tta_part = TTAStabilityDetector(
            pilot, num_samples=tta_samples, alpha=tta_alpha,
            strength=tta_strength, seed=0)
        logger.info(f'Also collecting TTA stability stats '
                    f'(M={tta_samples}, strength={tta_strength}).')

    # One extra, cheap deterministic sub-model reused from the same loaded
    # Keras model -- feeds the novelty-detection statistics below. A single
    # forward pass per frame (no dropout stochasticity needed: novelty is a
    # distance-from-training-distribution measure, not an agreement measure).
    model = pilot.interpreter.model
    feat_model = tf.keras.Model(
        model.inputs,
        [model.get_layer('dense_2').output, model.get_layer('conv2d_5').output])

    tub_paths = [os.path.expanduser(p) for p in tub_paths]
    dataset = TubDataset(config=cfg, tub_paths=tub_paths)
    records = dataset.get_records()
    if limit:
        records = records[:limit]
    logger.info(f'Replaying {len(records)} frames to collect variance '
                f'distribution (N={num_passes}, alpha={alpha})...')

    # Generic-encoder novelty (OOD) features -- the fix for the steering
    # model's feature space being task-collapsed (see donkeycar.parts.ood).
    # Collected on a strided subset to bound cost and batched after the loop.
    # A failure here (e.g. no internet to fetch encoder weights) degrades
    # gracefully: the calib simply omits the 'novelty_ood' block and live
    # novelty reports "unavailable" rather than breaking calibration.
    ood_name = getattr(cfg, 'XAI_NOVELTY_ENCODER', 'mobilenet_v2')
    ood_input = getattr(cfg, 'XAI_NOVELTY_ENCODER_INPUT', None)
    ood_alpha = getattr(cfg, 'XAI_NOVELTY_ENCODER_ALPHA', 1.0)
    ood_extractor = None
    try:
        from donkeycar.parts.ood import CPUEncoderExtractor
        ood_extractor = CPUEncoderExtractor(ood_name, ood_input, ood_alpha)
    except Exception as e:
        logger.warning(f'Could not build OOD encoder ({e}); novelty_ood block '
                       f'will be skipped -- live novelty will be unavailable.')
    ood_stride = max(1, len(records) // 3000) if ood_extractor else 1
    ood_imgs = []

    # Augmentation-aware novelty baseline (see the docstring). Self-gating:
    # an empty AUGMENTATIONS list means no augmenter and no behaviour change.
    aug_list = getattr(cfg, 'AUGMENTATIONS', None) or []
    if augment is None:
        augment = getattr(cfg, 'XAI_CALIBRATE_WITH_AUGMENTATIONS', True)
    aug_passes = int(getattr(cfg, 'XAI_CALIBRATE_AUG_PASSES', 2))
    augmenter = None
    if augment and aug_list and aug_passes > 0:
        try:
            from donkeycar.pipeline.augmentations import ImageAugmentation
            augmenter = ImageAugmentation(cfg, 'AUGMENTATIONS')
        except Exception as e:
            logger.warning(f'Could not build the training augmentation '
                           f'pipeline ({e}); novelty baselines will be fit on '
                           f'clean frames only, which can read as falsely '
                           f'novel under shadow/low light.')
    # Strided so the extra samples stay bounded regardless of tub size: a
    # diagonal Gaussian only needs enough samples per dimension, not one per
    # frame. Clean frames still outnumber augmented ones, which matches
    # training (augmentations apply with probability p, not always).
    aug_max_samples = int(getattr(cfg, 'XAI_CALIBRATE_AUG_MAX_SAMPLES', 1500))
    aug_stride = 1
    if augmenter is not None:
        aug_stride = max(1, (len(records) * aug_passes) //
                         max(1, aug_max_samples))
        logger.info(f'Novelty baselines will also include augmented frames '
                    f'({aug_list}, {aug_passes} pass(es), every '
                    f'{aug_stride} frame(s)) so shadow/low-light conditions '
                    f'the model was trained for do not read as novel.')
    aug_ood_imgs = []
    n_aug_samples = 0

    smoothed = []
    dense2_features = []
    conv5_features = []
    tta_variances = [] if tta_enabled else None
    for i, record in enumerate(records):
        img = record.image()  # uint8, already resized to model input size
        # part.run -> (angle, throttle, confidence, raw_var, smoothed_var)
        smooth_var = part.run(img)[4]
        smoothed.append(smooth_var)

        norm = normalize_image(img).astype(np.float32)
        dense2_out, conv5_out = feat_model(norm[np.newaxis, ...], training=False)
        dense2_features.append(np.asarray(dense2_out)[0])
        conv5_features.append(np.asarray(conv5_out)[0])

        if ood_extractor is not None and i % ood_stride == 0:
            ood_imgs.append(img)

        # Extra novelty samples only -- deliberately NOT fed to part.run /
        # tta_part.run, whose EMAs assume a contiguous time-ordered stream.
        if augmenter is not None and i % aug_stride == 0:
            for _ in range(aug_passes):
                aug_img = augmenter.run(img)
                aug_norm = normalize_image(aug_img).astype(np.float32)
                aug_d2, aug_c5 = feat_model(aug_norm[np.newaxis, ...],
                                            training=False)
                dense2_features.append(np.asarray(aug_d2)[0])
                conv5_features.append(np.asarray(aug_c5)[0])
                if ood_extractor is not None:
                    aug_ood_imgs.append(aug_img)
                n_aug_samples += 1

        if tta_part is not None:
            # tta_part.run -> (stability, raw_var, smoothed_var)
            tta_variances.append(tta_part.run(img)[2])

        if (i + 1) % 200 == 0:
            logger.info(f'  {i + 1}/{len(records)} frames')
        if progress_callback:
            progress_callback('calibrate', i + 1, len(records))

    tta_kwargs = {}
    if tta_enabled:
        tta_kwargs = dict(
            tta_variances=tta_variances,
            tta_num_samples=getattr(cfg, 'XAI_TTA_SAMPLES', 8),
            tta_strength=getattr(cfg, 'XAI_TTA_STRENGTH', 0.2),
            tta_alpha=getattr(cfg, 'XAI_TTA_ALPHA', 0.2))

    ood_kwargs = {}
    if ood_extractor is not None and ood_imgs:
        if progress_callback:
            progress_callback('calibrate_ood', 0, 1)
        try:
            all_ood_imgs = ood_imgs + aug_ood_imgs
            logger.info(f'Extracting OOD encoder features from '
                        f'{len(all_ood_imgs)} frames '
                        f'({len(ood_imgs)} clean, stride {ood_stride}; '
                        f'{len(aug_ood_imgs)} augmented)...')
            # Chunked: the extractor stacks and preprocesses the whole list
            # into one float32 batch, which at encoder resolution is far
            # larger than the uint8 frames. The returned vectors are small,
            # so chunking bounds peak memory without changing the result.
            ood_features = np.concatenate(
                [ood_extractor.extract_batch(all_ood_imgs[c:c + _OOD_CHUNK])
                 for c in range(0, len(all_ood_imgs), _OOD_CHUNK)], axis=0)
            encoder_path = default_encoder_path(model_path)
            ood_extractor.model.save(encoder_path)
            ood_kwargs = dict(
                ood_features=ood_features, ood_encoder=ood_extractor.name,
                ood_input_size=ood_extractor.input_size, ood_alpha=ood_alpha,
                ood_encoder_file=os.path.basename(encoder_path))
            logger.info(f'Saved OOD encoder sidecar -> {encoder_path}')
        except Exception as e:
            logger.warning(f'OOD feature/block build failed ({e}); skipping '
                           f'novelty_ood block.')
        if progress_callback:
            progress_callback('calibrate_ood', 1, 1)

    augmentation_info = {
        'applied': augmenter is not None,
        'augmentations': list(aug_list) if augmenter is not None else [],
        'passes': aug_passes if augmenter is not None else 0,
        'stride': aug_stride if augmenter is not None else None,
        'n_augmented_samples': n_aug_samples,
        'n_clean_samples': len(records),
        'scope': 'novelty baselines only (confidence/TTA use clean frames)',
    }
    calib = build_calibration(smoothed, num_passes, alpha,
                              percentiles=percentiles, model_path=model_path,
                              dense2_features=dense2_features,
                              conv5_features=conv5_features,
                              augmentation_info=augmentation_info,
                              **tta_kwargs, **ood_kwargs)
    out_path = out_path or default_calib_path(model_path)
    save_calibration(calib, out_path)
    dataset.close()
    return calib, out_path


def _summary(calib):
    p = calib['percentiles']
    s = calib['variance_stats']
    lines = [
        f"  frames analysed : {calib['n_frames']}",
        f"  N passes / alpha: {calib['num_passes']} / {calib['alpha']}",
        f"  variance min/max: {s['min']:.6f} / {s['max']:.6f}",
        f"  variance mean   : {s['mean']:.6f}",
        f"  p50 / p80 / p97 : {p['p50']:.6f} / {p['p80']:.6f} / {p['p97']:.6f}",
        "  tiers           : normal < p80, reduced < p97, critical >= p97",
        "  example mapping :",
    ]
    for label, v in (('median (p50)', p['p50']), ('p80', p['p80']),
                     ('p97', p['p97'])):
        lines.append(f"      variance {v:.6f} -> "
                     f"{variance_to_confidence(v, calib):5.1f}% confidence")

    aug = calib.get('augmentation')
    if aug:
        if aug.get('applied'):
            lines.append(f"  novelty baseline: clean + augmented "
                         f"({aug['n_augmented_samples']} augmented samples "
                         f"from {aug['augmentations']}, "
                         f"{aug['passes']} pass(es))")
        else:
            lines.append("  novelty baseline: clean frames only "
                         "(no AUGMENTATIONS configured)")

    ood = calib.get('novelty_ood')
    if ood:
        n_active = int(np.sum(ood['active_dims']))
        ds = ood['distance_stats']
        lines.append(f"  novelty (LIVE) : {ood['encoder']} encoder "
                     f"({ood['feat_dim']}-dim, input {ood['input_size']})")
        lines.append(f"      active dims     : {n_active}/{len(ood['active_dims'])}")
        lines.append(f"      distance min/max: {ds['min']:.1f} / {ds['max']:.1f}")

    for key, label in (('novelty_global', 'legacy global (dense_2, 50-dim)'),
                       ('novelty_spatial', 'offline spatial map (conv2d_5, pooled)')):
        block = calib.get(key)
        if not block:
            continue
        n_active = int(np.sum(block['active_dims']))
        n_total = len(block['active_dims'])
        ds = block['distance_stats']
        lines.append(f"  novelty {label}:")
        lines.append(f"      active dims     : {n_active}/{n_total}")
        lines.append(f"      distance min/max: {ds['min']:.4f} / {ds['max']:.4f}")

    tta = calib.get('tta')
    if tta:
        tp = tta['percentiles']
        ts = tta['variance_stats']
        lines.append(f"  TTA stability (M={tta['num_samples']}, "
                     f"strength={tta['strength']}):")
        lines.append(f"      variance min/max: {ts['min']:.6f} / {ts['max']:.6f}")
        lines.append(f"      p50 / p80 / p97 : {tp['p50']:.6f} / "
                     f"{tp['p80']:.6f} / {tp['p97']:.6f}")
        for label, v in (('median (p50)', tp['p50']), ('p97', tp['p97'])):
            lines.append(f"      variance {v:.6f} -> "
                         f"{tta_variance_to_stability(v, tta):5.1f}% stability")
    return "\n".join(lines)


def main(args=None):
    import argparse
    import donkeycar as dk

    parser = argparse.ArgumentParser(
        prog='mc_calibrate',
        description='Build an MC-Dropout confidence calibration from a tub.')
    parser.add_argument('--tub', nargs='+', required=True,
                        help='one or more tub paths to replay')
    parser.add_argument('--model', required=True, help='path to the .h5 model')
    parser.add_argument('--config', default=None,
                        help='path to config.py (defaults to ./config.py, '
                             'falling back to the bundled complete template)')
    parser.add_argument('--limit', type=int, default=None,
                        help='only use the first N frames')
    parser.add_argument('--passes', type=int, default=None,
                        help='override XAI_CONFIDENCE_PASSES')
    parser.add_argument('--alpha', type=float, default=None,
                        help='override XAI_CONFIDENCE_ALPHA')
    parser.add_argument('--out', default=None,
                        help='output calibration path '
                             '(default <model>.calib.json)')
    aug_group = parser.add_mutually_exclusive_group()
    aug_group.add_argument('--augment', dest='augment', action='store_true',
                           default=None,
                           help='widen the novelty baselines with augmented '
                                'frames (overrides '
                                'XAI_CALIBRATE_WITH_AUGMENTATIONS)')
    aug_group.add_argument('--no-augment', dest='augment',
                           action='store_false',
                           help='fit the novelty baselines on clean frames '
                                'only, even if AUGMENTATIONS is configured')
    parsed = parser.parse_args(args)

    config_path = parsed.config
    if config_path is None and not os.path.exists('config.py'):
        # Fall back to the bundled complete-car template so this runs even
        # outside a generated car directory.
        config_path = os.path.join(os.path.dirname(dk.__file__),
                                   'templates', 'cfg_complete.py')
        logger.warning(f'No ./config.py found; using bundled template '
                       f'defaults at {config_path}')
    cfg = dk.load_config(config_path)

    calib, out_path = calibrate_from_tub(
        cfg, parsed.tub, parsed.model, num_passes=parsed.passes,
        alpha=parsed.alpha, limit=parsed.limit, out_path=parsed.out,
        augment=parsed.augment)

    print('\nMC-Dropout calibration complete:')
    print(_summary(calib))
    print(f'\n  saved to: {out_path}')


if __name__ == '__main__':
    main()
