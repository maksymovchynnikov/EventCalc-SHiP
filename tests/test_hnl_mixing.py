"""Mixing must combine normalized shapes with partial, not total, widths."""
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from funcs import HNLmerging, initLLP, PDG
from eventcalc.runtime_generator import RuntimeModelScan


class HNLMixingTest(unittest.TestCase):
    def test_python_scan_accepts_the_decimal_upper_endpoint(self):
        scan = RuntimeModelScan(SimpleNamespace(model='HNL'))
        scan._initialized = True
        llp = initLLP.LLP.__new__(initLLP.LLP)
        llp.m_min_tabulated = .02
        llp.m_max_tabulated = np.nextafter(5.27, -np.inf)
        llp.compute_mass_dependent_properties = Mock()
        scan._llp_template = llp
        point = scan._mass_template(5.27)
        self.assertEqual(point.mass, llp.m_max_tabulated)
        with self.assertRaises(ValueError):
            scan._mass_template(5.270001)

    @classmethod
    def setUpClass(cls):
        rows = json.loads((ROOT / 'Distributions/HNL/HNL-decay.json').read_text())
        cls.rows = rows
        cls.functions = HNLmerging.get_MatrixElements_funcs(np.asarray([r[5:] for r in rows], object))
        cls.widths = np.loadtxt(ROOT / 'Distributions/HNL/HNLdecayWidth.dat')

    def test_conditional_shapes_follow_channel_partial_widths(self):
        for mass in (.15, .5, 1.5, 2.5, 4.5):
            for mixing in ([1/3]*3, [.05, .15, .8]):
                llp = initLLP.LLP.__new__(initLLP.LLP)
                llp.PDGs = [r[1] for r in self.rows]
                llp.BrRatios_raw = [[r[a+2] for r in self.rows] for a in range(3)]
                widths = [lambda m, a=a: np.interp(m, self.widths[:, 0], self.widths[:, a+1]) for a in range(3)]
                mixed = llp.create_Msquared_triple_functions(*self.functions, *mixing, *widths, mass=mass)
                for channel in ('2ev', '2muv', 'emuv', 'etauv', 'mutauv'):
                    index = next(i for i, r in enumerate(self.rows) if r[0] == channel)
                    pdgs = [p for p in llp.PDGs[index] if p != -999]
                    m1, m2, m3 = [PDG.get_mass(p) for p in pdgs]
                    if mass <= m1+m2+m3:
                        continue
                    pure = [f[index] for f in self.functions]
                    # Independent adaptive integration in E1, analytical
                    # quadratic integration in E3 using three sample values.
                    def integral(f):
                        def inner(e1):
                            s = mass**2+m1**2-2*mass*e1
                            c = (mass-e1)*(s+m3**2-m2**2)/(2*s)
                            h = np.sqrt(max(0., (e1**2-m1**2)*(s-(m2+m3)**2)*(s-(m2-m3)**2)))/(2*s)
                            return h/3*(f(mass,e1,c-h,m1,m2,m3)+4*f(mass,e1,c,m1,m2,m3)+f(mass,e1,c+h,m1,m2,m3))
                        upper = (mass**2+m1**2-(m2+m3)**2)/(2*mass)
                        span = upper-m1
                        # Resolve the light-lepton boundary layer as well as
                        # the square-root endpoints in an independent adaptive integral.
                        def transformed(t):
                            return inner(m1+span*np.sin(t)**2)*span*np.sin(2*t)
                        return quad(transformed,0,np.pi/2,epsabs=1e-24,epsrel=2e-10,
                                    points=[.001, .01, np.pi/2-.01, np.pi/2-.001],limit=200)[0]
                    norms = np.array([integral(f) for f in pure])
                    actual_norm = integral(mixed[index])
                    p = np.array([mixing[a]*widths[a](mass)*np.interp(mass,
                        np.array(self.rows[index][a+2])[:,0],np.array(self.rows[index][a+2])[:,1]) for a in range(3)])
                    expected = p / p.sum()
                    e1 = np.linspace(m1, (mass**2+m1**2-(m2+m3)**2)/(2*mass), 9)[1:-1]
                    s = mass**2+m1**2-2*mass*e1
                    e3 = (mass-e1)*(s+m3**2-m2**2)/(2*s)
                    target = sum(w*f(mass,e1,e3,m1,m2,m3)/n for w,f,n in zip(expected,pure,norms) if w>0)
                    with self.subTest(mass=mass, mixing=mixing, channel=channel):
                        np.testing.assert_allclose(mixed[index](mass,e1,e3,m1,m2,m3)/actual_norm,target,rtol=2e-7,atol=1e-12)


if __name__ == '__main__':
    unittest.main()
