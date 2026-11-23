"""Run with: python -m unittest discover -s tests -v"""
import unittest
from dataclasses import replace
import numpy as np
from virtualtuft import (LoopConfig, VirtualTuft, make_world, taps,
                         split_indices, run_trial)


class GateZeroTests(unittest.TestCase):
    def test_all_worlds_are_reproducible(self):
        for kind in ("periodic", "period2", "chaotic", "draining", "switch", "shuffled"):
            a = make_world(kind, n=140, seed=77)
            self.assertTrue(np.array_equal(a, make_world(kind, n=140, seed=77)))
            self.assertTrue(np.all((a > 0) & (a <= 1)))

    def test_causal_state_no_future_input(self):
        c = LoopConfig(nodes=8, delay_cycles=2)
        x = np.linspace(.2, .8, 100)
        y = x.copy()
        y[81:] = 1 - y[81:]
        a, b = VirtualTuft(c), VirtualTuft(c)
        z1, z2 = a.transform(x), b.transform(y)
        self.assertTrue(np.array_equal(z1[:81], z2[:81]))

    def test_zero_feedback_changes_reservoir_with_slot_coupling(self):
        a = VirtualTuft(LoopConfig(nodes=8, feedback=0))
        b = VirtualTuft(LoopConfig(nodes=8, feedback=.8))
        x = np.linspace(.2, .8, 40)
        self.assertGreater(np.max(abs(a.transform(x)-b.transform(x))), 1e-4)

    def test_adaptive_gain_is_bounded(self):
        machine = VirtualTuft(LoopConfig(nodes=10, feedback=.8, adapt_rate=30))
        machine.transform(make_world("chaotic", 130))
        self.assertTrue(-.95 <= machine.feedback <= .95)
        self.assertTrue(np.all(np.isfinite(machine.state)))

    def test_split_forecasts_target_future(self):
        tr, va, te = split_indices(400)
        self.assertLess(tr.max(), va.min())
        self.assertLess(va.max(), te.min())
        self.assertLess(te.max() + 1, 400)
        t = taps(np.arange(100.), 12)
        self.assertEqual(t[45, 3], 42)

    def test_benchmark_has_comparable_arms(self):
        record = run_trial("period2", seed=11, n=150, nodes=6)
        for key in ("clock_mean", "persistence", "linear_taps", "fixed_loop", "adaptive_loop"):
            self.assertTrue(np.isfinite(record["arms"][key]["test_mae"]))


if __name__ == "__main__":
    unittest.main()