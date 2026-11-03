"""Optional independent integration check for the paper's reference geometry."""
from importlib.util import find_spec
import unittest

import numpy as np
import bruce


@unittest.skipUnless(find_spec('ellc'), 'Optional comparison requires ellc')
class EllcComparisonTests(unittest.TestCase):
    def test_matched_power2_reference_case(self):
        import ellc
        radius, ratio, c, alpha = .05, .1, .715452, .643777
        duration = bruce.binarystar.transit_width(radius, ratio, 0.)
        phase = np.linspace(-.55*duration, .55*duration, 1001)
        pars = dict(radius_1=radius, radius_2=radius*ratio, sbratio=0.,
                    incl=90., period=1., ld_1='power-2', ldc_1=[c, alpha],
                    shape_1='sphere', shape_2='sphere', verbose=0)
        reference = ellc.lc(phase, grid_1='very_fine', grid_2='very_fine', **pars)
        fine = ellc.lc(phase, grid_1='fine', grid_2='fine', **pars)
        model = bruce.binarystar.lc(
            phase, radius_1=radius, k=ratio, incl=np.pi/2,
            period=1., e=0., c=c, alpha=alpha, cadence=0., accurate_tp=0)
        self.assertTrue(np.all(np.isfinite(model)))
        self.assertLess(np.max(np.abs(model-reference)), 100e-6)
        self.assertLess(np.max(np.abs(fine-reference)), 2e-6)
        # This bound applies only to this recorded geometry and power-2 law.
        # It is not a precision guarantee over all stellar/planet parameters.


if __name__ == '__main__':
    unittest.main()
