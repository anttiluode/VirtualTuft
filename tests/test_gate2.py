"""Gate 2 regression tests: hidden state, bias, recurrence and causality."""
import unittest
import numpy as np
from gate2 import generate, machine, structured_features, trial, SEEDS
from virtualtuft import split_indices


class GateTwoTests(unittest.TestCase):
    def test_observation_noise_does_not_change_hidden_dynamics(self):
        u0, x0, d0 = generate('henon', 240, 103, 0.)
        u1, x1, d1 = generate('henon', 240, 103, .02)
        np.testing.assert_array_equal(x0, x1)
        np.testing.assert_array_equal(d0, d1)
        self.assertGreater(np.std(u1-u0), .01)

    def test_hidden_driver_alignment_ordinary_henon(self):
        u,x,hidden = generate('henon', 240, 103, 0.)
        np.testing.assert_allclose(x[1:], 1 - 1.4*x[:-1]**2 + .3*hidden[:-1], atol=2e-14)
        np.testing.assert_allclose(hidden[1:], x[:-1], atol=2e-14)

    def test_delayed_world_d3_is_finite_and_aligned(self):
        u,x,hidden = generate('delayed_d3', 700, 103, .02)
        self.assertTrue(np.all(np.isfinite(x)))
        np.testing.assert_allclose(x[1:], 1-1.1*x[:-1]**2+.3*hidden[:-1], atol=2e-14)
        np.testing.assert_allclose(hidden[3:],x[:-3],atol=2e-14)

    def test_stateless_machine_really_forgets_all_earlier_cycles(self):
        a=np.array([.2,.3,.5,.7]); b=np.array([.7,.9,.1,.7])
        x,_=machine(a,.4,'stateless',bias_on=True)
        y,_=machine(b,.4,'stateless',bias_on=True)
        np.testing.assert_array_equal(x[-1],y[-1])

    def test_pure_echo_has_no_leaky_memory_when_gain_is_zero(self):
        # Pure-echo implementation is memoryless if its explicit echo is zero.
        a=np.array([.2,.3,.5,.7]); b=np.array([.7,.9,.1,.7])
        x,_=machine(a,.4,'pure_echo',gain=0.,bias_on=True)
        y,_=machine(b,.4,'pure_echo',gain=0.,bias_on=True)
        np.testing.assert_array_equal(x[-1],y[-1])

    def test_bias_breaks_odd_symmetry(self):
        x=np.array([.15,.31,.42,-.21,-.46,.33])
        a,_=machine(x,0.,'stateless',bias_on=False)
        b,_=machine(-x,0.,'stateless',bias_on=False)
        np.testing.assert_allclose(a,-b,atol=1e-12)
        c,_=machine(x,0.,'stateless',bias_on=True)
        d,_=machine(-x,0.,'stateless',bias_on=True)
        self.assertGreater(float(np.mean(np.abs(c+d))),.04)

    def test_future_flip_does_not_change_states(self):
        x=np.linspace(.4,.7,160); y=x.copy(); y[80:] = -.7
        for mode in ('stateless','leaky','pure_echo','combined'):
            gain=.7 if mode in ('pure_echo','combined') else 0.
            a,_=machine(x,.5,mode,gain=gain,delay=3,bias_on=True)
            b,_=machine(y,.5,mode,gain=gain,delay=3,bias_on=True)
            np.testing.assert_array_equal(a[:80],b[:80],err_msg=mode)

    def test_structured_baseline_uses_known_correct_lag_only(self):
        x=np.linspace(-.5,.5,70)
        a=structured_features(x,0.,3)
        self.assertAlmostEqual(a[37,0], x[37])
        self.assertAlmostEqual(a[37,1], x[37]**2)
        self.assertAlmostEqual(a[37,2], x[34])
        y=x.copy();y[53:] = 5.
        np.testing.assert_array_equal(a[:53],structured_features(y,0.,3)[:53])

    def test_frozen_seed_set_and_disjoint_splits(self):
        self.assertEqual(SEEDS,(101,103,107,109,113))
        tr,va,te=split_indices(1200)
        self.assertLess(tr[-1],va[0]);self.assertLess(va[-1],te[0]);self.assertLess(te[-1],1199)

    def test_short_smoke_is_finite_and_has_all_required_controls(self):
        r=trial('henon',seed=101,n=460,noise=.02)
        for key in ('current','structured','quadratic_50','knn_archive','stateless_biased',
                    'leaky_odd','leaky_biased','pure_echo_biased',
                    'combined_odd','combined_biased','adaptive_biased'):
            self.assertTrue(np.isfinite(r['arms'][key]['test_mae']),key)
        self.assertTrue(np.isfinite(r['hidden_probe']['direct_history']['test_mae']))

if __name__=='__main__':unittest.main()
