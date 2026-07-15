"""
Offline Grad-CAM "uncertainty map" tool (Feature 3 of the uncertainty toolkit).

For frames of a recorded tub, this tool runs Grad-CAM once per MC-Dropout
pass (same N stochastic passes as the live confidence signal) and computes
the *pixel-wise variance* of the N attention maps. Regions where the model's
attention is inconsistent across dropout sub-networks are rendered as a
heatmap overlay -- a spatial answer to "where was the model uncertain?".

This is offline-only by design: each analysed frame costs one batched
forward + backward pass (N Grad-CAMs share a single gradient tape), which is
fine for post-drive analysis but too heavy for the 20Hz drive loop.

Frame selection (feasibility): by default only the top-K most uncertain
frames are analysed, ranked by the steering variance either logged in the
tub (``pilot/smoothed_variance``, written when driving with --uncertainty)
or recomputed by replaying the tub through the MC-Dropout part. ``--all``
forces every frame (use only for short drives).

Output (consumed by the Feature 4 viewer):
    <out>/
      data.json            -- timeline of variance/confidence for ALL frames,
                              plus an entry per analysed frame
      images/<idx>_orig.jpg      -- camera frame
      images/<idx>_unc.png       -- uncertainty-map overlay (attention variance)
      images/<idx>_attn.png      -- mean-attention overlay (where it looked)

Usage:
    python -m donkeycar.parts.gradcam_uncertainty \
        --tub data/ --model models/mypilot.h5 [--top-k 50] [--all] \
        [--percentile 90] [--limit N] [--passes 15] [--out dir]

Caveats: linear model only (same scope as the live signal); the variance map
inherits MC-Dropout's blind spots -- it measures disagreement between dropout
sub-networks, not distance from the training distribution.
"""
import argparse
import json
import logging
import os
import time

import numpy as np
import tensorflow as tf
from PIL import Image

logger = logging.getLogger(__name__)


class GradCamUncertainty:
    """
    Computes N stochastic Grad-CAM maps per frame (dropout active, one map
    per MC-Dropout pass) and reduces them to a mean-attention map and a
    pixel-wise variance ("uncertainty") map.

    All N maps are produced in a single batched forward+backward pass: the
    image is replicated N times, dropout gives each batch row an independent
    sub-network, and the gradient of the summed steering output w.r.t. the
    conv features yields each row's own Grad-CAM weights (rows are
    independent, so d(sum steer_i)/d(feat_j) == d(steer_j)/d(feat_j)).
    """

    def __init__(self, model, num_passes=15, conv_layer_name=None):
        self.model = model
        self.num_passes = int(num_passes)

        if conv_layer_name:
            conv_layer = model.get_layer(conv_layer_name)
        else:
            conv_layer = self._last_conv_layer(model)
        logger.info(f'GradCamUncertainty: using conv layer '
                    f'"{conv_layer.name}" {conv_layer.output.shape}, '
                    f'N={self.num_passes}')

        # Steering is the first model output (n_outputs0 on the linear model).
        self.grad_model = tf.keras.Model(
            model.inputs, [conv_layer.output, model.outputs[0]])

    @staticmethod
    def _last_conv_layer(model):
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer
        raise ValueError('No Conv2D layer found in model; Grad-CAM needs a '
                         'convolutional architecture.')

    def attention_maps(self, norm_img):
        """
        :param norm_img: float32 image, [0,1], shape (H, W, C)
        :return: np array (N, h, w) of per-pass Grad-CAM maps, each
                 normalised to [0, 1] by its own max (so the variance across
                 maps measures *shape* disagreement, not magnitude).
        """
        batch = tf.convert_to_tensor(
            np.repeat(norm_img[np.newaxis, ...], self.num_passes, axis=0))

        with tf.GradientTape() as tape:
            features, steering = self.grad_model(batch, training=True)
            target = tf.reduce_sum(steering)
        grads = tape.gradient(target, features)          # (N, h, w, c)

        # Grad-CAM: channel weights = spatially-pooled gradients.
        weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
        cams = tf.nn.relu(tf.reduce_sum(weights * features, axis=-1))
        cams = cams.numpy()                              # (N, h, w)

        # Per-map max-normalisation; guard all-zero maps.
        maxes = cams.reshape(self.num_passes, -1).max(axis=1)
        maxes[maxes == 0] = 1.0
        return cams / maxes[:, None, None]

    def uncertainty_map(self, img_arr):
        """
        :param img_arr: uint8 [0,255] camera frame (H, W, C)
        :return: (mean_map, var_map) both float (H, W) upsampled to image
                 size; mean_map in [0,1], var_map normalised to [0,1] with
                 its raw peak value returned as third element.
        """
        from donkeycar.utils import normalize_image
        norm = normalize_image(img_arr).astype(np.float32)
        cams = self.attention_maps(norm)

        mean_map = cams.mean(axis=0)
        var_map = cams.var(axis=0)
        raw_peak = float(var_map.max())

        h, w = img_arr.shape[:2]
        mean_up = _resize_map(mean_map, w, h)
        var_up = _resize_map(var_map, w, h)
        if raw_peak > 0:
            var_up = var_up / var_up.max()
        if mean_up.max() > 0:
            mean_up = mean_up / mean_up.max()
        return mean_up, var_up, raw_peak


def _resize_map(map2d, width, height):
    """Bilinear upsample a float map to (height, width)."""
    img = Image.fromarray(map2d.astype(np.float32), mode='F')
    return np.asarray(img.resize((width, height), Image.BILINEAR))


def _jet(x):
    """Simple jet colormap: float map [0,1] -> float RGB (H, W, 3)."""
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def overlay_heatmap(img_arr, map01, strength=0.6):
    """
    Blend a [0,1] heat map over a uint8 image; the map value doubles as the
    per-pixel alpha so cold regions stay as plain camera image.
    """
    base = img_arr.astype(np.float32) / 255.0
    heat = _jet(map01)
    alpha = (map01 * strength)[..., None]
    out = base * (1 - alpha) + heat * alpha
    return (out * 255).astype(np.uint8)


def _collect_variances(records, model_path, cfg, num_passes, alpha):
    """
    Get a per-frame (raw, smoothed) steering-variance list, preferring values
    logged in the tub (drives recorded with --uncertainty), else replaying
    the frames through the MC-Dropout part.
    """
    logged = [r.underlying.get('pilot/smoothed_variance') for r in records]
    n_logged = sum(v is not None for v in logged)
    if n_logged >= 0.5 * len(records) and n_logged > 0:
        logger.info(f'Using logged uncertainty for {n_logged}/{len(records)} '
                    f'frames.')
        return [v if v is not None else 0.0 for v in logged]

    logger.info(f'No logged uncertainty in tub; replaying {len(records)} '
                f'frames through MC-Dropout (N={num_passes})...')
    from donkeycar.parts.keras import KerasLinear
    from donkeycar.parts.mc_dropout import MCDropoutConfidence
    pilot = KerasLinear()
    pilot.load(model_path)
    part = MCDropoutConfidence(pilot, num_passes=num_passes, alpha=alpha)
    smoothed = []
    for i, record in enumerate(records):
        smoothed.append(part.run(record.image())[4])
        if (i + 1) % 200 == 0:
            logger.info(f'  {i + 1}/{len(records)} frames')
    return smoothed


def analyze_tub(cfg, tub_path, model_path, out_dir, num_passes=15, alpha=0.2,
                top_k=50, percentile=None, analyze_all=False, limit=None,
                export_frames=False):
    """
    Run the full Feature 3 pipeline on one tub. Returns the data.json dict.
    """
    from donkeycar.parts.keras import KerasLinear
    from donkeycar.parts.mc_calibrate import (default_calib_path,
                                              load_calibration,
                                              variance_to_confidence)
    from donkeycar.pipeline.types import TubDataset

    t0 = time.time()
    model_path = os.path.expanduser(model_path)
    tub_path = os.path.expanduser(tub_path)

    dataset = TubDataset(config=cfg, tub_paths=[tub_path])
    records = dataset.get_records()
    if limit:
        records = records[:limit]
    if not records:
        raise ValueError(f'No records found in {tub_path}')

    # Per-frame uncertainty timeline (logged or replayed).
    variances = _collect_variances(records, model_path, cfg, num_passes, alpha)

    # Confidence % if this model has a calibration.
    calib = None
    calib_path = default_calib_path(model_path)
    if os.path.exists(calib_path):
        calib = load_calibration(calib_path)
        logger.info(f'Loaded calibration {calib_path}')
    else:
        logger.warning(f'No calibration at {calib_path}; timeline will have '
                       f'variance only.')

    def conf(v):
        return variance_to_confidence(v, calib) if calib is not None else None

    timeline = []
    for record, v in zip(records, variances):
        timeline.append({
            'index': record.underlying.get('_index'),
            'timestamp_ms': record.underlying.get('_timestamp_ms'),
            'variance': float(v),
            'confidence': conf(v),
            # image filename inside the tub's images/ dir, so the viewer can
            # show the camera frame for non-analysed frames when the tub is
            # available locally.
            'tub_image': record.underlying.get('cam/image_array'),
        })

    # Optionally export every camera frame so the analysis dir is fully
    # self-contained (playback works with just this folder, no tub needed --
    # useful when copying results off a cluster). ~5-8 KB per frame.
    if export_frames:
        frames_dir = os.path.join(out_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        logger.info(f'Exporting {len(records)} camera frames to {frames_dir}')
        for record, entry in zip(records, timeline):
            name = f"{entry['index']:06d}.jpg"
            Image.fromarray(record.image()).save(
                os.path.join(frames_dir, name))
            entry['image'] = f'frames/{name}'

    # Select frames to analyse.
    order = np.argsort(variances)[::-1]        # most uncertain first
    if analyze_all:
        selected = list(range(len(records)))
    elif percentile is not None:
        thresh = float(np.percentile(variances, percentile))
        selected = [i for i in range(len(records)) if variances[i] >= thresh]
        logger.info(f'{len(selected)} frames at/above p{percentile} '
                    f'(variance >= {thresh:.6f})')
    else:
        selected = list(order[:top_k])
    logger.info(f'Analysing {len(selected)} of {len(records)} frames.')

    # Build the Grad-CAM engine on the raw Keras model.
    pilot = KerasLinear()
    pilot.load(model_path)
    engine = GradCamUncertainty(pilot.interpreter.model, num_passes=num_passes)

    images_dir = os.path.join(out_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    frames = {}
    for n, i in enumerate(sorted(selected)):
        record = records[i]
        img = record.image()
        idx = record.underlying.get('_index', i)

        mean_map, var_map, raw_peak = engine.uncertainty_map(img)
        orig_name = f'{idx:06d}_orig.jpg'
        unc_name = f'{idx:06d}_unc.png'
        attn_name = f'{idx:06d}_attn.png'
        Image.fromarray(img).save(os.path.join(images_dir, orig_name))
        Image.fromarray(overlay_heatmap(img, var_map)).save(
            os.path.join(images_dir, unc_name))
        Image.fromarray(overlay_heatmap(img, mean_map)).save(
            os.path.join(images_dir, attn_name))

        frames[str(idx)] = {
            'orig': f'images/{orig_name}',
            'overlay': f'images/{unc_name}',
            'attention': f'images/{attn_name}',
            'variance': float(variances[i]),
            'confidence': conf(variances[i]),
            'attention_variance_peak': raw_peak,
        }
        if (n + 1) % 10 == 0:
            logger.info(f'  analysed {n + 1}/{len(selected)} frames')

    data = {
        'model': os.path.basename(model_path),
        'tub': tub_path,
        'num_passes': num_passes,
        'alpha': alpha,
        'n_records': len(records),
        'n_analyzed': len(selected),
        'frames_exported': bool(export_frames),
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'note': ('Attention-variance maps from MC-Dropout Grad-CAM. '
                 'Relative per-model signal, not a calibrated probability.'),
        'timeline': timeline,
        'frames': frames,
    }
    with open(os.path.join(out_dir, 'data.json'), 'w') as f:
        json.dump(data, f)
    dataset.close()
    logger.info(f'Done in {time.time() - t0:.1f}s -> {out_dir} '
                f'({len(selected)} frames analysed)')
    return data


def main(args=None):
    import donkeycar as dk

    parser = argparse.ArgumentParser(
        prog='gradcam_uncertainty',
        description='Offline Grad-CAM uncertainty maps for a recorded tub.')
    parser.add_argument('--tub', required=True, help='tub path to analyse')
    parser.add_argument('--model', required=True, help='path to .h5 model')
    parser.add_argument('--config', default=None, help='path to config.py')
    parser.add_argument('--out', default=None,
                        help='output dir (default <tub>/gradcam_analysis)')
    parser.add_argument('--passes', type=int, default=None,
                        help='MC-Dropout passes (default cfg or 15)')
    parser.add_argument('--top-k', type=int, default=50,
                        help='analyse the K most uncertain frames (default)')
    parser.add_argument('--percentile', type=float, default=None,
                        help='instead analyse frames >= this variance '
                             'percentile (e.g. 90)')
    parser.add_argument('--all', action='store_true',
                        help='analyse every frame (short drives only)')
    parser.add_argument('--limit', type=int, default=None,
                        help='only consider the first N records')
    parser.add_argument('--export-frames', action='store_true',
                        help='also copy every camera frame into the output '
                             'dir so playback in the viewer works without '
                             'the tub present (self-contained results)')
    parsed = parser.parse_args(args)

    config_path = parsed.config
    if config_path is None and not os.path.exists('config.py'):
        config_path = os.path.join(os.path.dirname(dk.__file__),
                                   'templates', 'cfg_complete.py')
        logger.warning(f'No ./config.py; using bundled defaults.')
    cfg = dk.load_config(config_path)

    num_passes = parsed.passes or getattr(cfg, 'MC_DROPOUT_PASSES', 15)
    alpha = getattr(cfg, 'MC_DROPOUT_ALPHA', 0.2)
    out_dir = parsed.out or os.path.join(os.path.expanduser(parsed.tub),
                                         'gradcam_analysis')

    data = analyze_tub(cfg, parsed.tub, parsed.model, out_dir,
                       num_passes=num_passes, alpha=alpha,
                       top_k=parsed.top_k, percentile=parsed.percentile,
                       analyze_all=parsed.all, limit=parsed.limit,
                       export_frames=parsed.export_frames)
    print(f"\nGrad-CAM uncertainty analysis complete:")
    print(f"  frames in drive : {data['n_records']}")
    print(f"  frames analysed : {data['n_analyzed']}")
    print(f"  output          : {out_dir}")
    print(f"\nView it with:")
    print(f"  python -m donkeycar.parts.uncertainty_viewer "
          f"--analysis {out_dir}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
