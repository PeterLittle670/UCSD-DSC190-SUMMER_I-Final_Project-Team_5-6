import unittest

import numpy as np

from donkeycar.parts.mc_calibrate import (
    build_calibration, variance_to_confidence, novelty_distance_to_score,
    _make_strictly_increasing,
)


class TestConfidenceCalibration(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(42)
        variances = rng.gamma(shape=2.0, scale=0.01, size=1000)  # always > 0
        self.calib = build_calibration(variances, num_passes=15, alpha=0.2)

    def test_confidence_is_monotonically_decreasing_with_variance(self):
        xs = self.calib['confidence_anchors']['variance']
        samples = np.linspace(xs[0], xs[-1], 50)
        scores = [variance_to_confidence(v, self.calib) for v in samples]
        for a, b in zip(scores, scores[1:]):
            self.assertGreaterEqual(a, b - 1e-9)

    def test_confidence_clamps_outside_anchor_range(self):
        below_range = variance_to_confidence(-1.0, self.calib)
        above_range = variance_to_confidence(1e6, self.calib)
        self.assertAlmostEqual(below_range, 99.0, places=6)  # min-variance anchor
        self.assertAlmostEqual(above_range, 5.0, places=6)   # max-variance anchor


class TestNoveltyCalibration(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(7)
        variances = rng.gamma(shape=2.0, scale=0.01, size=500)
        dense2 = rng.normal(size=(500, 50))
        self.calib = build_calibration(variances, num_passes=15, alpha=0.2,
                                       dense2_features=dense2)
        self.block = self.calib['novelty_global']

    def test_novelty_block_is_built(self):
        self.assertIn('novelty_global', self.calib)

    def test_novelty_is_monotonically_increasing_with_distance(self):
        xs = self.block['score_anchors']['distance']
        samples = np.linspace(xs[0], xs[-1], 50)
        scores = [novelty_distance_to_score(d, self.block) for d in samples]
        for a, b in zip(scores, scores[1:]):
            self.assertLessEqual(a, b + 1e-9)

    def test_novelty_clamps_outside_anchor_range(self):
        below_range = novelty_distance_to_score(-1.0, self.block)
        above_range = novelty_distance_to_score(1e6, self.block)
        self.assertAlmostEqual(below_range, 2.0, places=6)   # min-distance anchor
        self.assertAlmostEqual(above_range, 98.0, places=6)  # max-distance anchor

    def test_no_dense2_features_means_no_novelty_block(self):
        rng = np.random.default_rng(8)
        variances = rng.gamma(shape=2.0, scale=0.01, size=100)
        calib = build_calibration(variances, num_passes=15, alpha=0.2)
        self.assertNotIn('novelty_global', calib)
        self.assertNotIn('novelty_spatial', calib)


class TestMakeStrictlyIncreasing(unittest.TestCase):

    def test_nudges_duplicate_values_strictly_increasing(self):
        xs = _make_strictly_increasing([1.0, 1.0, 1.0, 2.0])
        for a, b in zip(xs, xs[1:]):
            self.assertLess(a, b)

    def test_already_increasing_values_are_unchanged(self):
        xs = _make_strictly_increasing([1.0, 2.0, 3.0])
        self.assertEqual(xs, [1.0, 2.0, 3.0])


if __name__ == '__main__':
    unittest.main()
