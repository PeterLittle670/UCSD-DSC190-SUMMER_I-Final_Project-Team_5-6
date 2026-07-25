import unittest

import numpy as np

from donkeycar.parts.mc_calibrate import (
    build_calibration, variance_to_confidence, novelty_distance_to_score,
    tta_variance_to_stability, _make_strictly_increasing,
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


class TestTTACalibration(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(11)
        variances = rng.gamma(shape=2.0, scale=0.01, size=500)
        tta_variances = rng.gamma(shape=2.0, scale=1e-4, size=500)
        self.calib = build_calibration(variances, num_passes=15, alpha=0.2,
                                       tta_variances=tta_variances,
                                       tta_num_samples=8, tta_strength=0.2,
                                       tta_alpha=0.2)
        self.block = self.calib['tta']

    def test_tta_block_is_built_with_metadata(self):
        self.assertIn('tta', self.calib)
        self.assertEqual(self.block['num_samples'], 8)
        self.assertEqual(self.block['strength'], 0.2)

    def test_stability_is_monotonically_decreasing_with_variance(self):
        xs = self.block['stability_anchors']['variance']
        samples = np.linspace(xs[0], xs[-1], 50)
        scores = [tta_variance_to_stability(v, self.block) for v in samples]
        for a, b in zip(scores, scores[1:]):
            self.assertGreaterEqual(a, b - 1e-9)

    def test_stability_clamps_outside_anchor_range(self):
        below = tta_variance_to_stability(-1.0, self.block)
        above = tta_variance_to_stability(1e6, self.block)
        self.assertAlmostEqual(below, 99.0, places=6)  # min-variance anchor
        self.assertAlmostEqual(above, 5.0, places=6)   # max-variance anchor

    def test_no_tta_variances_means_no_tta_block(self):
        rng = np.random.default_rng(12)
        calib = build_calibration(rng.gamma(2.0, 0.01, 100), num_passes=15,
                                  alpha=0.2)
        self.assertNotIn('tta', calib)


class TestAugmentationProvenance(unittest.TestCase):
    """The 'augmentation' block records how the novelty baselines were fit
    (clean-only vs widened with training-style augmented frames), so a
    calibration can be told apart from an older one after the fact."""

    def _variances(self, n=200, seed=21):
        return np.random.default_rng(seed).gamma(shape=2.0, scale=0.01, size=n)

    def test_absent_when_not_supplied_keeps_older_calibrations_valid(self):
        calib = build_calibration(self._variances(), num_passes=15, alpha=0.2)
        self.assertNotIn('augmentation', calib)

    def test_recorded_verbatim_when_supplied(self):
        info = {'applied': True, 'augmentations': ['SHADOW', 'GAMMA'],
                'passes': 2, 'stride': 12, 'n_augmented_samples': 1500,
                'n_clean_samples': 9221, 'scope': 'novelty baselines only'}
        calib = build_calibration(self._variances(), num_passes=15, alpha=0.2,
                                  augmentation_info=info)
        self.assertEqual(calib['augmentation'], info)

    def test_does_not_disturb_the_confidence_calibration(self):
        """Provenance is metadata: adding it must not move any number the
        confidence signal reads."""
        v = self._variances()
        plain = build_calibration(v, num_passes=15, alpha=0.2)
        with_info = build_calibration(v, num_passes=15, alpha=0.2,
                                      augmentation_info={'applied': False})
        self.assertEqual(plain['percentiles'], with_info['percentiles'])
        self.assertEqual(plain['confidence_anchors'],
                         with_info['confidence_anchors'])
        self.assertEqual(plain['n_frames'], with_info['n_frames'])
        self.assertEqual(plain['tiers'], with_info['tiers'])

    def test_n_frames_counts_clean_frames_only(self):
        """Augmented samples are pooled into the novelty feature matrices but
        never into the variance distribution, so the reported frame count
        must still describe the replay, not the enlarged feature set."""
        rng = np.random.default_rng(31)
        variances = rng.gamma(shape=2.0, scale=0.01, size=300)
        # 300 clean frames, but 900 novelty feature vectors (clean + augmented)
        dense2 = rng.normal(size=(900, 50))
        calib = build_calibration(variances, num_passes=15, alpha=0.2,
                                  dense2_features=dense2,
                                  augmentation_info={'applied': True,
                                                     'n_augmented_samples': 600})
        self.assertEqual(calib['n_frames'], 300)
        self.assertIn('novelty_global', calib)


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
