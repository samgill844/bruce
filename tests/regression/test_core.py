"""Deterministic regression checks; no archive access or interactive windows."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import bruce
import bruce_c


class CoreTests(unittest.TestCase):
    def test_lazy_import(self):
        import subprocess
        subprocess.run([sys.executable, '-c',
                        'import bruce, sys; '
                        'assert "lightkurve" not in sys.modules; '
                        'assert "matplotlib.pyplot" not in sys.modules'], check=True)

    def test_c_likelihood_casts_and_strides(self):
        for dtype in (np.int32, np.float32, np.float64):
            y = np.arange(10, dtype=dtype)[::2]
            err = np.full(10, 2, dtype=dtype)[::2]
            model = np.ones(10, dtype=dtype)[::2]
            expected = 0.5 * np.sum((y.astype(float) - model)**2 / err**2)
            # The low-level extension historically returns negative log L.
            self.assertAlmostEqual(bruce_c.loglike(y, err, model, 0., 0.), expected)

    def test_c_rejects_invalid_shapes(self):
        with self.assertRaises(ValueError):
            bruce_c.loglike(np.ones(2), np.ones(3), np.ones(2), 0., 0.)
        with self.assertRaises(ValueError):
            bruce_c.loglike(np.ones((2, 2)), np.ones(4), np.ones(4), 0., 0.)
        with self.assertRaises((ValueError, TypeError)):
            bruce.binarystar.lc(np.ones((2, 2)))

    def test_night_boundaries(self):
        split = bruce.data.find_nights_from_data
        self.assertEqual(split([], 1), [])
        assert_array_equal(split([1], 1)[0], [0])
        groups = split([0, .1, 1.05, 1.1, 2.2], .9)
        self.assertEqual([list(x) for x in groups], [[0, 1], [2, 3], [4]])

    def test_filters_and_spline(self):
        from scipy.ndimage import maximum_filter, median_filter
        from scipy.interpolate import UnivariateSpline
        time = np.linspace(0, 1, 41)
        flux = 1 + .01 * np.sin(30 * time)
        flatten = bruce.data.flatten_data_with_function
        filtered = median_filter(maximum_filter(flux, 3), 5)
        expected = np.polyval(np.polyfit(time, filtered, 3), time)
        assert_allclose(flatten(time, flux, method='poly1d', max_median=1,
                                Nmaxfilter=3, Nmedianfilter=5), expected)
        assert_allclose(flatten(time, flux, method='spline', splinesmooth=.001),
                        UnivariateSpline(time, flux, s=.001)(time))
        with self.assertRaises(ValueError):
            flatten(time, flux, method='unknown')

    def test_binning_invalid_inputs(self):
        for time, flux, width in [([], [], .1), ([0, 1], [1], .1),
                                  ([0, 1], [1, 1], 0),
                                  ([1, 0], [1, 1], .1)]:
            with self.assertRaises(ValueError):
                bruce.data.bin_data(np.array(time), np.array(flux), width)

    def test_phase_dispersion_validation(self):
        time = np.arange(5, dtype=float)
        for peaks, periods in [([-1], [1.]), ([5], [1.]), ([0], [0.])]:
            with self.assertRaises(ValueError):
                bruce_c.phase_dispersion(time, np.array(peaks, dtype=np.int32),
                                         np.array(periods), time, time, time)
        dispersion, unused = bruce_c.phase_dispersion(
            time, np.array([0, 2], dtype=np.int32), np.array([2., 3.]),
            time, time, np.ones(5))
        assert_allclose(dispersion, [0., 1. / 3.])

    def test_flux_roundtrip(self):
        f = np.array([.5, 1., 2.])
        e = np.array([.01, .02, .03])
        mags, errors = bruce.data.flux_to_mags(f, e)
        actual, uncertainty = bruce.data.mags_to_flux(mags, errors)
        assert_allclose(actual, f)
        assert_allclose(uncertainty, e)

    def test_vectorised_likelihood(self):
        # Load the numerical module directly so this test needs no archive services.
        spec = importlib.util.spec_from_file_location(
            'sampler_core', Path(__file__).resolve().parents[2] /
            'src/bruce/sampler/sampler.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        y = np.array([1., 2., np.nan, 4.])
        err = np.ones(4)
        model = np.zeros(4)
        values = module.loglike_(y, err, model, jitter=.1)
        self.assertAlmostEqual(module.loglike(y, err, model, jitter=.1),
                               sum(v for v in values if np.isfinite(v)))


if __name__ == '__main__':
    unittest.main()
