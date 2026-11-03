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

    def test_lc_likelihood_matches_numpy(self):
        # Exercise the integer offset parsed by lc_loglike, including its
        # variance term, jitter, exposure integration and converted views.
        for dtype in [np.float32, np.float64]:
            time = np.repeat(np.linspace(-.15, .15, 101).astype(dtype), 2)[::2]
            error = np.repeat(np.linspace(.0007, .0013, 101).astype(dtype), 2)[::2]
            for eccentricity, cadence in [(0., 0.), (.4, 2 / 1440)]:
                parameters = dict(period=3., radius_1=.1, k=.1,
                                  incl=np.arccos(.03), e=eccentricity,
                                  w=.7, accurate_tp=0, cadence=cadence,
                                  noversample=11)
                model = bruce.binarystar.lc(time, **parameters)
                observed = (model + .0002 * np.sin(np.arange(time.size))).astype(dtype)
                observed = np.repeat(observed, 2)[::2]
                for jitter in [0., .0003]:
                    variance = error.astype(float)**2 + jitter**2
                    residual = observed.astype(float) - model
                    for offset in [0, 1]:
                        with self.subTest(dtype=dtype, e=eccentricity,
                                          jitter=jitter, offset=offset):
                            expected = .5 * np.sum(residual**2 / variance +
                                                   offset * np.log(variance))
                            actual = bruce.binarystar.lc(
                                time, observed, error, jitter=jitter,
                                offset=offset, **parameters)
                            assert_allclose(actual, expected, rtol=1e-12, atol=1e-8)

    def test_exposure_sampling_validation(self):
        time = np.linspace(-.1, .1, 21)
        flux, error = np.ones(21), np.full(21, .001)
        for samples in [-1, 0, 1, 2, 10]:
            with self.subTest(samples=samples):
                for observed in [False, True]:
                    with self.assertRaisesRegex(ValueError, "odd integer >= 3"):
                        bruce.binarystar.lc(
                            time, flux if observed else None,
                            error if observed else None,
                            cadence=2 / 1440, noversample=samples)
                for radius in [.1, np.array([.1])]:
                    with self.assertRaisesRegex(ValueError, "odd integer >= 3"):
                        bruce.template_match.template_match_lightcurve(
                            time, flux, error, flux, radius_1=radius,
                            time_trial=np.array([0.]), cadence=2 / 1440,
                            noversample=samples)
        for cadence in [-1., np.nan, np.inf]:
            with self.assertRaisesRegex(ValueError, "cadence"):
                bruce.binarystar.lc(time, cadence=cadence)

    def test_integrated_transit_is_symmetric(self):
        time = np.linspace(.035, .065, 101)
        parameters = dict(period=3., radius_1=.1, k=.1,
                          incl=np.arccos(.03), e=0., w=np.pi/2,
                          accurate_tp=0, cadence=2/1440)
        for samples in [3, 11, 21]:
            positive = bruce.binarystar.lc(time, noversample=samples, **parameters)
            negative = bruce.binarystar.lc(-time, noversample=samples, **parameters)
            assert_allclose(positive, negative, rtol=0., atol=1e-12)
        # The public default must also be valid for non-zero exposures.
        assert_allclose(bruce.binarystar.lc(time, **parameters),
                        bruce.binarystar.lc(time, noversample=11, **parameters))

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
