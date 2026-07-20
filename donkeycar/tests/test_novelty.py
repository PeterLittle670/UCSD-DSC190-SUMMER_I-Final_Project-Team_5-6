import unittest

import numpy as np

from donkeycar.parts.novelty import mahalanobis_diag, fit_diagonal_gaussian


class TestMahalanobisDiag(unittest.TestCase):

    def test_zero_at_mean(self):
        mean = np.zeros(5)
        var = np.ones(5)
        self.assertAlmostEqual(mahalanobis_diag(mean, mean, var), 0.0, places=6)

    def test_increases_with_distance(self):
        mean = np.zeros(5)
        var = np.ones(5)
        near = mahalanobis_diag(np.full(5, 0.1), mean, var)
        far = mahalanobis_diag(np.full(5, 5.0), mean, var)
        self.assertGreater(far, near)

    def test_active_dims_excludes_dimension(self):
        mean = np.zeros(3)
        var = np.ones(3)
        vec = np.array([0.0, 0.0, 100.0])  # huge deviation, only in dim 2
        active_all = np.array([True, True, True])
        active_excl_dim2 = np.array([True, True, False])
        dist_all = mahalanobis_diag(vec, mean, var, active_all)
        dist_excl = mahalanobis_diag(vec, mean, var, active_excl_dim2)
        self.assertAlmostEqual(dist_excl, 0.0, places=6)
        self.assertGreater(dist_all, dist_excl)

    def test_batched_input_returns_one_distance_per_row(self):
        mean = np.zeros(4)
        var = np.ones(4)
        batch = np.stack([np.zeros(4), np.ones(4)])
        distances = mahalanobis_diag(batch, mean, var)
        self.assertEqual(distances.shape, (2,))
        self.assertAlmostEqual(distances[0], 0.0, places=6)
        self.assertGreater(distances[1], 0.0)


class TestFitDiagonalGaussian(unittest.TestCase):

    def test_mean_and_var_match_numpy(self):
        rng = np.random.default_rng(0)
        data = rng.normal(loc=3.0, scale=2.0, size=(500, 5))
        stats = fit_diagonal_gaussian(data)
        np.testing.assert_allclose(stats['mean'], data.mean(axis=0), rtol=1e-9)
        np.testing.assert_allclose(stats['var'], data.var(axis=0), rtol=1e-9)

    def test_flags_near_constant_dimension_inactive(self):
        rng = np.random.default_rng(1)
        data = rng.normal(loc=0.0, scale=1.0, size=(500, 4))
        data[:, 2] = 5.0  # dead dimension: ~zero variance
        stats = fit_diagonal_gaussian(data)
        active = stats['active_dims']
        self.assertFalse(active[2])
        self.assertTrue(all(active[i] for i in (0, 1, 3)))


if __name__ == '__main__':
    unittest.main()
