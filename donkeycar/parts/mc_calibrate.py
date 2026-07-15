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

IMPORTANT: this is a relative, per-model calibration, NOT a formally calibrated
Bayesian probability. A displayed "90%" means "this frame's uncertainty is low
*relative to this model's own baseline drive*", nothing more.
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


def build_calibration(smoothed_variances, num_passes, alpha,
                      percentiles=DEFAULT_PERCENTILES, model_path=None):
    """
    Turn a collected distribution of smoothed variances into a calibration
    dict (JSON-serialisable).
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
    return calib


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


def default_calib_path(model_path):
    """Calibration file sits next to the model: <model>.calib.json"""
    return os.path.splitext(model_path)[0] + '.calib.json'


def save_calibration(calib, path):
    with open(path, 'w') as f:
        json.dump(calib, f, indent=2)
    logger.info(f'Saved calibration to {path}')


def load_calibration(path):
    with open(path) as f:
        return json.load(f)


def calibrate_from_tub(cfg, tub_paths, model_path, num_passes=None,
                       alpha=None, limit=None, percentiles=DEFAULT_PERCENTILES,
                       out_path=None):
    """
    Replay a tub through the MC-Dropout part, collect smoothed variances, build
    and save a calibration file. Returns (calib_dict, out_path).
    """
    # Imported here so this module is cheap to import without TF.
    from donkeycar.parts.keras import KerasLinear
    from donkeycar.parts.mc_dropout import MCDropoutConfidence
    from donkeycar.pipeline.types import TubDataset

    num_passes = num_passes if num_passes is not None \
        else getattr(cfg, 'MC_DROPOUT_PASSES', 15)
    alpha = alpha if alpha is not None \
        else getattr(cfg, 'MC_DROPOUT_ALPHA', 0.2)

    logger.info(f'Loading model {model_path}')
    pilot = KerasLinear()
    pilot.load(model_path)
    part = MCDropoutConfidence(pilot, num_passes=num_passes, alpha=alpha)

    tub_paths = [os.path.expanduser(p) for p in tub_paths]
    dataset = TubDataset(config=cfg, tub_paths=tub_paths)
    records = dataset.get_records()
    if limit:
        records = records[:limit]
    logger.info(f'Replaying {len(records)} frames to collect variance '
                f'distribution (N={num_passes}, alpha={alpha})...')

    smoothed = []
    for i, record in enumerate(records):
        img = record.image()  # uint8, already resized to model input size
        # part.run -> (angle, throttle, confidence, raw_var, smoothed_var)
        smooth_var = part.run(img)[4]
        smoothed.append(smooth_var)
        if (i + 1) % 200 == 0:
            logger.info(f'  {i + 1}/{len(records)} frames')

    calib = build_calibration(smoothed, num_passes, alpha,
                              percentiles=percentiles, model_path=model_path)
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
                        help='override MC_DROPOUT_PASSES')
    parser.add_argument('--alpha', type=float, default=None,
                        help='override MC_DROPOUT_ALPHA')
    parser.add_argument('--out', default=None,
                        help='output calibration path '
                             '(default <model>.calib.json)')
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
        alpha=parsed.alpha, limit=parsed.limit, out_path=parsed.out)

    print('\nMC-Dropout calibration complete:')
    print(_summary(calib))
    print(f'\n  saved to: {out_path}')


if __name__ == '__main__':
    main()
