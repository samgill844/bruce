"""Check the compiled orbital routines against bracketed Kepler solutions."""
import ctypes
import unittest

import numpy as np
from scipy.optimize import brentq

import bruce
import bruce_c


TWO_PI = 2 * np.pi


def reference_true_anomaly(mean_anomaly, eccentricity):
    """Use a signed bracket, independent of the C solver's approximation."""
    mean = (mean_anomaly + np.pi) % TWO_PI - np.pi
    if mean == 0:
        return 0.
    eccentric = brentq(
        lambda value: value - eccentricity * np.sin(value) - mean,
        -np.pi, np.pi, xtol=1e-15,
    )
    return np.arctan2(np.sqrt(1 - eccentricity**2) * np.sin(eccentric),
                      np.cos(eccentric) - eccentricity) % TWO_PI


class KeplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Test the actual extension's C symbols rather than a Python translation.
        cls.library = ctypes.CDLL(bruce_c.__file__)
        cls.eccentric = cls.library.getEccentricAnomaly
        cls.eccentric.argtypes = [ctypes.c_double] * 2
        cls.eccentric.restype = ctypes.c_double
        cls.true = cls.library.getTrueAnomaly
        cls.true.argtypes = [ctypes.c_double] * 6 + [ctypes.c_int]
        cls.true.restype = ctypes.c_double
        cls.periastron = cls.library.t_ecl_to_peri
        cls.periastron.argtypes = [ctypes.c_double] * 5
        cls.periastron.restype = ctypes.c_double

    def assert_same_angle(self, actual, expected, tolerance=2e-9):
        difference = np.arctan2(np.sin(actual - expected),
                                np.cos(actual - expected))
        self.assertLess(abs(difference), tolerance)
        self.assertGreaterEqual(actual, 0.)
        self.assertLess(actual, TWO_PI)

    def test_kepler_residual_and_wrapping(self):
        means = np.r_[np.linspace(-4 * np.pi, 4 * np.pi, 401),
                      [-1e-10, 0., 1e-10, -TWO_PI + .01]]
        for eccentricity in [0., 1e-6, .1, .5, .9, .99, .999999]:
            for mean in means:
                with self.subTest(e=eccentricity, mean=mean):
                    eccentric = self.eccentric(mean, eccentricity)
                    target = mean % TWO_PI
                    residual = eccentric - eccentricity * np.sin(eccentric) - target
                    self.assertLess(abs(residual), 5e-14)
                    self.assertGreaterEqual(eccentric, 0.)
                    self.assertLess(eccentric, TWO_PI)

    def test_true_anomaly_against_bracketed_solver(self):
        period, epoch, omega, inclination = 5., 10., .7, 1.5
        for eccentricity in [1e-6, 9e-6, .1, .9, .99, .9999]:
            for accurate in [0, 1]:
                if accurate:
                    tp = self.periastron(epoch, period, inclination,
                                         eccentricity, omega)
                else:
                    nu_zero = np.pi / 2 - omega
                    e_zero = np.arctan2(
                        np.sqrt(1 - eccentricity**2) * np.sin(nu_zero),
                        eccentricity + np.cos(nu_zero),
                    )
                    tp = epoch - (e_zero - eccentricity * np.sin(e_zero)) * period / TWO_PI
                for mean in [-TWO_PI + .01, -2., -.01, .01, 2., np.pi]:
                    with self.subTest(e=eccentricity, accurate=accurate, mean=mean):
                        time = tp + period * mean / TWO_PI
                        actual = self.true(time, eccentricity, omega, period,
                                           epoch, inclination, accurate)
                        expected = reference_true_anomaly(mean, eccentricity)
                        self.assert_same_angle(actual, expected)

    def test_approximate_conjunction_and_epoch_shift(self):
        for eccentricity in [0., 1e-6, .3, .9]:
            for omega in [-3., .7, 3., 5.]:
                expected = (np.pi / 2 - omega) % TWO_PI
                self.assert_same_angle(
                    self.true(10., eccentricity, omega, 5., 10., 1.5, 0),
                    expected,
                )
                # Translating both dates must leave the orbit unchanged.
                self.assert_same_angle(
                    self.true(12., eccentricity, omega, 5., 10., 1.5, 0),
                    self.true(112., eccentricity, omega, 5., 110., 1.5, 0),
                )
        first = self.true(12., .3, .7, 5., 10., 1.5, 0)
        shifted = self.true(12., .3, .7, 5., 11., 1.5, 0)
        self.assertGreater(abs(first - shifted), .1)

    def test_circular_orbit_and_periodicity(self):
        for accurate in [0, 1]:
            for omega in [-3., .7, 5.]:
                for time in [-12., 10., 12.]:
                    expected = (TWO_PI * (time - 10.) / 5. + np.pi / 2 - omega) % TWO_PI
                    self.assert_same_angle(
                        self.true(time, 0., omega, 5., 10., 1.5, accurate),
                        expected,
                    )
            for eccentricity in [.3, .9]:
                first = self.true(4.99, eccentricity, .7, 5., 10., 1.5, accurate)
                self.assert_same_angle(
                    self.true(9.99, eccentricity, .7, 5., 10., 1.5, accurate), first,
                )

    def test_public_rv_wrapper_uses_epoch(self):
        time = np.array([-5., 0., 10., 12., 15.])
        eccentricity, omega, period, epoch = .9, .7, 5., 10.
        nu_zero = np.pi / 2 - omega
        e_zero = np.arctan2(np.sqrt(1 - eccentricity**2) * np.sin(nu_zero),
                            eccentricity + np.cos(nu_zero))
        mean_zero = e_zero - eccentricity * np.sin(e_zero)
        expected = []
        for value in time:
            mean = mean_zero + TWO_PI * (value - epoch) / period
            nu = reference_true_anomaly(mean, eccentricity)
            expected.append(20. * (np.cos(nu + omega) +
                                   eccentricity * np.cos(omega)) + 3.)
        actual = bruce.binarystar.rv1(
            time, t_zero=epoch, period=period, K1=20., e=eccentricity,
            w=omega, V0=3., incl=1.5, accurate_tp=0,
        )
        np.testing.assert_allclose(actual, expected, rtol=0., atol=1e-9)

    def test_invalid_orbits_return_nan(self):
        for eccentricity in [-.1, 1., 1.1, np.nan, np.inf]:
            self.assertTrue(np.isnan(self.eccentric(.5, eccentricity)))
            self.assertTrue(np.isnan(self.true(1., eccentricity, .7, 5., 0., 1.5, 0)))
        for period in [0., -1., np.nan, np.inf]:
            self.assertTrue(np.isnan(self.true(1., .3, .7, period, 0., 1.5, 1)))
        for mean in [np.nan, np.inf, -np.inf]:
            self.assertTrue(np.isnan(self.eccentric(mean, .3)))


if __name__ == '__main__':
    unittest.main()
