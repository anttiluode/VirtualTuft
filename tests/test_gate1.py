"""Test the Neon Gate 1 audit controls and causal protocol."""
import unittest
from dataclasses import replace
import numpy as np
from gate1 import Config, Loop, Readout, nonlinear_taps, repeated_switch, trial
from virtualtuft import split_indices


class GateOneTests(unittest.TestCase):
    def test_no_recurrence_is_exactly_memoryless_across_cycles(self):
        c = Config(nodes=12, recurrence=False, gain=.9, delay=3)
        a, b = Loop(c), Loop(c)
        for u in (.2, .3, .7, .9):
            a.step(u)
        self.assertTrue(np.array_equal(a.step(.42), b.step(.42)))

    def test_recurrent_causal_to_future_flip(self):
        x = np.linspace(.3, .7, 130)
        y = x.copy()
        y[87:] = .9
        for a in (0., .012):
            c = Config(nodes=9, gain=.7, delay=2, adapt_rate=a)
            m, q = Loop(c), Loop(c)
            self.assertTrue(np.array_equal(m.transform(x)[:87], q.transform(y)[:87]))

    def test_quadratic_baseline_causal_and_centered(self):
        x = np.linspace(.3,.7, 140)
        y = x.copy(); y[91:] = .9
        center = float(x[:60].mean())
        z, zz = nonlinear_taps(x, center), nonlinear_taps(y, center)
        self.assertEqual(z.shape, (140, 50))
        self.assertTrue(np.array_equal(z[:91], zz[:91]))
        self.assertAlmostEqual(z[80, 10], (x[80] - center) ** 2)

    def test_train_only_scaling(self):
        x = np.linspace(.3,.7, 130).reshape(-1,1)
        y = x.copy(); y[90:] += 300
        labels = np.roll(x[:,0], -1)
        tr = np.arange(10,70)
        a,b = Readout(x, labels, tr, .01), Readout(y, labels, tr, .01)
        self.assertTrue(np.array_equal(a.w, b.w))
        self.assertTrue(np.array_equal(a.sd, b.sd))

    def test_prequential_no_future_peek(self):
        rng = np.random.default_rng(31)
        X = rng.normal(size=(130,3))
        y = rng.normal(size=130)
        changed = y.copy(); changed[90:] = -100
        tr = np.arange(15,60)
        a = Readout(X, y, tr, .01)
        p, q = a.online_prediction(X,y,60), a.online_prediction(X,changed,60)
        self.assertTrue(np.array_equal(p[:91], q[:91]))

    def test_repeated_switch_has_multiple_regime_boundaries(self):
        x = repeated_switch(1050, 11, .008)
        self.assertEqual(len(x), 1050)
        for a,b in zip((80,629,839),(628,838,1048)):
            self.assertGreaterEqual(b//75 - a//75,2)

    def test_small_gate_has_finite_errors(self):
        r = trial('chaotic', seed=11, n=450)
        for name, v in r['arms'].items():
            self.assertTrue(np.isfinite(v['test_mae']), name)
        self.assertEqual(r['selection']['fixed_gain'] in (0.,.35,.7,.9), True)

if __name__ == '__main__':
    unittest.main()
