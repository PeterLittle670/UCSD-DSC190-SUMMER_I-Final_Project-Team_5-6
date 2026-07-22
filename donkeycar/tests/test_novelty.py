import unittest
from unittest import mock

import numpy as np
import tensorflow as tf

from donkeycar.parts.novelty import (mahalanobis_diag, fit_diagonal_gaussian,
                                     FeatureNoveltyDetector)


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


def _tiny_pilot():
    """A minimal model with a layer named 'dense_2', the surface
    FeatureNoveltyDetector reaches for, wrapped in a stub pilot exposing
    .interpreter.model."""
    tf.random.set_seed(0)
    inp = tf.keras.Input((8, 8, 3))
    x = tf.keras.layers.Flatten()(inp)
    feat = tf.keras.layers.Dense(5, name='dense_2')(x)
    out = tf.keras.layers.Dense(1, name='n_outputs0')(feat)
    model = tf.keras.Model(inp, out)

    class _Interp:
        def __init__(self, m):
            self.model = m

    class _Pilot:
        def __init__(self, m):
            self.interpreter = _Interp(m)

    return _Pilot(model)


class TestFeatureNoveltyDetectorInterval(unittest.TestCase):
    # Rate-limiting matters here: this part rides alongside the driving pilot
    # on every frame, so an unthrottled forward pass adds real load (and, on
    # a Pi, real power draw) with no way to turn it down.

    def setUp(self):
        self.pilot = _tiny_pilot()
        self.img = (np.random.default_rng(0).random((8, 8, 3)) * 255).astype(np.uint8)

    def _counting_part(self, interval):
        part = FeatureNoveltyDetector(self.pilot, interval=interval)
        calls = {'n': 0}
        real_feat_model = part.feat_model

        def counting_call(*a, **kw):
            calls['n'] += 1
            return real_feat_model(*a, **kw)

        part.feat_model = counting_call
        return part, calls

    def test_zero_interval_runs_forward_pass_every_frame(self):
        part, calls = self._counting_part(interval=0.0)
        part.run(self.img)
        part.run(self.img)
        part.run(self.img)
        self.assertEqual(calls['n'], 3)

    def test_interval_skips_forward_pass_when_held(self):
        part, calls = self._counting_part(interval=1.0)
        with mock.patch('donkeycar.parts.novelty.time.time', return_value=100.0):
            part.run(self.img)
        self.assertEqual(calls['n'], 1)
        with mock.patch('donkeycar.parts.novelty.time.time', return_value=100.5):
            part.run(self.img)   # within interval -> held, NO forward pass
        self.assertEqual(calls['n'], 1)
        with mock.patch('donkeycar.parts.novelty.time.time', return_value=101.5):
            part.run(self.img)   # past interval -> recompute
        self.assertEqual(calls['n'], 2)

    def test_held_values_match_last_real_update(self):
        part, _ = self._counting_part(interval=1.0)
        with mock.patch('donkeycar.parts.novelty.time.time', return_value=100.0):
            _, raw1, smoothed1 = part.run(self.img)
        with mock.patch('donkeycar.parts.novelty.time.time', return_value=100.3):
            _, raw2, smoothed2 = part.run(self.img)
        self.assertEqual(raw1, raw2)
        self.assertEqual(smoothed1, smoothed2)


if __name__ == '__main__':
    unittest.main()
