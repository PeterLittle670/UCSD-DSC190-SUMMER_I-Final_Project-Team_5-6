import unittest

import numpy as np
import tensorflow as tf

from donkeycar.parts.tta import photometric_batch, TTAStabilityDetector
from donkeycar.parts.mc_calibrate import build_calibration


def _tiny_pilot():
    """Minimal single-image-input, two-output (angle/throttle) model wrapped in
    a stub pilot exposing .interpreter.model / .interpreter.input_keys, the
    surface TTAStabilityDetector reaches for."""
    tf.random.set_seed(0)
    inp = tf.keras.Input((16, 16, 3), name='img_in')
    x = tf.keras.layers.Conv2D(4, 3, activation='relu')(inp)
    x = tf.keras.layers.Flatten()(x)
    angle = tf.keras.layers.Dense(1, name='angle')(x)
    throttle = tf.keras.layers.Dense(1, name='throttle')(x)
    model = tf.keras.Model(inp, [angle, throttle])

    class _Interp:
        def __init__(self, m):
            self.model = m
            self.input_keys = ['img_in']

    class _Pilot:
        def __init__(self, m):
            self.interpreter = _Interp(m)

    return _Pilot(model)


class TestPhotometricBatch(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.img = self.rng.random((16, 16, 3)).astype(np.float32)

    def test_shape_and_range(self):
        batch = photometric_batch(self.img, 8, 0.2, self.rng)
        self.assertEqual(batch.shape, (8, 16, 16, 3))
        self.assertGreaterEqual(batch.min(), 0.0)
        self.assertLessEqual(batch.max(), 1.0)

    def test_zero_strength_is_identity(self):
        batch = photometric_batch(self.img, 5, 0.0, self.rng)
        for m in range(5):
            np.testing.assert_allclose(batch[m], self.img, atol=1e-6)

    def test_higher_strength_gives_more_variance(self):
        weak = photometric_batch(self.img, 16, 0.05, np.random.default_rng(1))
        strong = photometric_batch(self.img, 16, 0.4, np.random.default_rng(1))
        self.assertGreater(strong.var(axis=0).mean(), weak.var(axis=0).mean())

    def test_geometry_is_preserved(self):
        # A bright top half vs dark bottom half must stay that way for every
        # augmented copy -- photometric augmentation never moves content.
        img = np.zeros((16, 16, 3), dtype=np.float32)
        img[:8] = 0.9
        img[8:] = 0.1
        batch = photometric_batch(img, 8, 0.2, np.random.default_rng(2))
        for m in range(8):
            self.assertGreater(batch[m, :8].mean(), batch[m, 8:].mean())


class TestTTAStabilityDetector(unittest.TestCase):

    def setUp(self):
        self.pilot = _tiny_pilot()
        self.img = (np.random.default_rng(3).random((16, 16, 3)) * 255).astype(np.uint8)

    def test_run_returns_three_values_and_nonneg_variance(self):
        part = TTAStabilityDetector(self.pilot, num_samples=8, strength=0.2,
                                    seed=0)
        stability, raw_var, smoothed_var = part.run(self.img)
        self.assertIsNone(stability)          # no calibration loaded
        self.assertGreaterEqual(raw_var, 0.0)
        self.assertGreaterEqual(smoothed_var, 0.0)

    def test_none_frame_passthrough(self):
        part = TTAStabilityDetector(self.pilot, seed=0)
        stability, raw_var, smoothed_var = part.run(None)
        self.assertIsNone(stability)
        self.assertEqual(raw_var, 0.0)

    def test_stability_uses_calibration_block_when_present(self):
        part = TTAStabilityDetector(self.pilot, num_samples=8, seed=0)
        # Attach a real tta block built from a synthetic variance distribution.
        rng = np.random.default_rng(4)
        calib = build_calibration(
            rng.gamma(2.0, 0.01, 200), num_passes=15, alpha=0.2,
            tta_variances=rng.gamma(2.0, 1e-4, 200),
            tta_num_samples=8, tta_strength=0.2, tta_alpha=0.2)
        part.calibration = calib['tta']
        part.run(self.img)
        score = part._score()
        self.assertIsNotNone(score)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)


if __name__ == '__main__':
    unittest.main()
