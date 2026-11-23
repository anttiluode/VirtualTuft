"""VirtualTuft Gate 2: can delayed recurrence recover hidden temporal state?

Audit protocol (2026-11-23): Hénon, delayed nonlinear map, independently held-out
seeds, fixed random biases, separate leaky/echo/combined controls, strong explicit
history baselines. All readings causal, hyperparameters chosen on validation ONLY.
Optional numba acceleration; NumPy-only fallback. No neuron or myelin claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np

from gate1 import Readout, nonlinear_taps, fit_selected
from virtualtuft import split_indices, taps, mae

try:
    from numba import njit
except ImportError:
    def njit(func):
        return func

SEEDS = (101, 103, 107, 109, 113)
PENALTIES = (.001, .01, .1)
GAINS = (0., .35, .7, .9)
DELAYS = (1, 2, 3)
KINDS = ("henon", "delayed_d3")
NOISES = (0.0, .02)


@njit
def _machine(u, mask, bias, center, leak, gain, delay, slot_carry, input_strength, adapt, boundary):
    """One sequential nonlinear unit sampled in 50 slots per input cycle.

    delay is in complete cycles. Circular previous-slot boundary is explicitly
    controlled by slot_carry. Missing history is zero. State is causal.
    Bias is inside the nonlinear response, never estimated from held-out data.
    """
    n = len(u); m = len(mask)
    X = np.zeros((n, m)); gain_trace = np.zeros(n)
    previous = np.zeros(m)
    g = gain; e2 = 0.
    for t in range(n):
        new = np.empty(m)
        prior_slot = previous[m-1] if boundary else 0.
        echo = 0.
        if t >= delay:
            for i in range(m):
                echo += mask[i] * X[t-delay, i] / m
        if adapt > 0 and t >= delay:
            # Local homeostatic gain adaptation, intentionally NOT downstream
            # supervised prediction or the auxiliary echo-predictor gradient.
            e2 = .99 * e2 + .01 * echo * echo
            g = min(.95, max(0., g + adapt * (.025 - e2)))
        for i in range(m):
            drive = input_strength*mask[i]*(u[t]-center) + slot_carry*prior_slot + bias[i]
            if t >= delay:
                drive += g * X[t-delay, i]
            v = np.tanh(drive)
            new[i] = (1-leak)*previous[i] + leak*v
            prior_slot = new[i]
        X[t] = new
        previous = new
        gain_trace[t] = g
    return X, gain_trace


def machine(u, center, kind, gain=0., delay=1, bias_on=True, adapt=0., nodes=50):
    rng = np.random.default_rng(2026)
    mask = rng.choice(np.array([-1., 1.]), nodes)
    bias = rng.uniform(-.65, .65, nodes) if bias_on else np.zeros(nodes)
    if kind == "stateless":
        return _machine(u, mask, bias, center, 1., 0., 1, .4, 1.6, 0., False)
    if kind == "leaky":
        return _machine(u, mask, bias, center, .65, 0., 1, .4, 1.6, 0., True)
    if kind == "pure_echo":
        return _machine(u, mask, bias, center, 1., gain, delay, 0., 1.6, adapt, False)
    if kind == "combined":
        return _machine(u, mask, bias, center, .65, gain, delay, .4, 1.6, adapt, True)
    raise ValueError(kind)


def generate(kind, n, seed, observation_noise):
    """Noise is added to observed x only, never to the hidden world evolution.

    henon: x[t+1] = 1 - 1.4*x[t]^2 + .3*x[t-1].
    delayed_d3: x[t+1] = 1 - 1.1*x[t]^2 + .3*x[t-3].
    The second is a distinct delayed quadratic world; a=1.4 is unstable with
    lag 3 and would not be a valid bound-preserving benchmark.
    """
    if kind not in KINDS:
        raise ValueError(kind)
    rng = np.random.default_rng(seed)
    d, a = (1, 1.4) if kind == "henon" else (3, 1.1)
    hidden = np.zeros(n + 501)
    hidden[:d+2] = rng.uniform(-.12, .12, d+2)
    for t in range(d+1, len(hidden)-1):
        hidden[t+1] = 1 - a*hidden[t]**2 + .3*hidden[t-d]
        if abs(hidden[t+1]) > 5 or not np.isfinite(hidden[t+1]):
            raise ArithmeticError("unbounded synthetic delayed world")
    x = hidden[501:].copy()
    observed = x + rng.normal(0, observation_noise, len(x))
    # Hidden driver for causal x[t+1]: x[t-d], inaccessible to online models.
    driver = hidden[501-d:501-d+n].copy()
    return observed, x, driver


def structured_features(u, center, lag):
    """Intentionally privileged equation-form baseline with known correct lag.

    This baseline is not a blind discovery method. It shows what using the
    correct nonlinear law structure buys a digital predictor.
    """
    q = taps(u, max(lag+1, 2))
    z0 = q[:, 0] - center
    zlag = q[:, lag] - center
    return np.column_stack((z0, z0*z0, zlag))


def current_features(u, center):
    z = u - center
    return np.column_stack((z, z*z))


def knn_predict(u, labels, tr, va, te, lag, k=10):
    """Fixed training archive; nearest neighbors don't see validation/test labels."""
    X = taps(u, lag)[:-1]
    # Train-only scaling. x is same variable across lags; one scale is enough.
    center = np.mean(u[tr]); scale = max(float(np.std(u[tr])), .02)
    X = (X-center)/scale
    ref = X[tr]
    ret = {}
    for key, idx in (("val",va),("test",te)):
        d2 = np.sum((X[idx,None,:]-ref[None,:,:])**2, axis=-1)
        nn = np.argpartition(d2, min(k,len(tr)-1), axis=1)[:,:k]
        dist = np.take_along_axis(d2, nn, axis=1)
        ww = 1/(np.sqrt(dist)+.025)
        ret[key] = np.sum(labels[tr[nn]]*ww,axis=1)/np.sum(ww,axis=1)
    return mae(ret["val"],labels[va]), mae(ret["test"],labels[te])


def _fit(X, labels, tr, va, te):
    v, lam, m = fit_selected(X, labels, tr, va, PENALTIES)
    return {"val_mae":v, "test_mae":mae(m.prediction(X[te]),labels[te]), "ridge":lam}, m


def trial(kind="henon", seed=101, n=1200, noise=.02, export_states=False):
    u, hidden_x, driver = generate(kind,n,seed,noise)
    tr, va, te = split_indices(n); labels = u[1:]
    center = float(np.mean(u[tr]))
    lag = 1 if kind == "henon" else 3
    arms = {}; states = {}
    for name,X in (("current",current_features(u,center)[:-1]),
                   ("structured",structured_features(u,center,lag)[:-1]),
                   ("quadratic_50",nonlinear_taps(u,center)[:-1])):
        arms[name],_ = _fit(X,labels,tr,va,te)
    val, test = knn_predict(u,labels,tr,va,te,lag+1)
    arms["knn_archive"]={"val_mae":val,"test_mae":test,"archive_count":len(tr),"lag_count":lag+1}
    arms["clock"]={"test_mae":mae(np.full(len(te),np.mean(labels[tr])), labels[te])}
    arms["persistence"]={"test_mae":mae(u[te], labels[te])}
    # Bias/symmetry ablation: same masks, dynamics, split, gain=0.
    for mode in ("stateless","leaky"):
        for bias_on in (False,True):
            name = mode+("_biased" if bias_on else "_odd")
            X,_ = machine(u,center,mode,bias_on=bias_on)
            arms[name],_ = _fit(X[:-1],labels,tr,va,te)
            if export_states: states[name]=X
    # Pure delayed echo: no leaky persistence, no prior-slot cross-cycle carry.
    # Combined: leaky state + returning delay + within-slot cascade.
    for mode in ("pure_echo","combined"):
        for bias_on in (False, True):
            name=mode+("_biased" if bias_on else "_odd")
            picks=[]
            for g in (GAINS[1:] if mode=="pure_echo" else GAINS):
                for d in DELAYS:
                    X,_=machine(u,center,mode,gain=g,delay=d,bias_on=bias_on)
                    result,_ = _fit(X[:-1],labels,tr,va,te)
                    picks.append((result["val_mae"],g,d,result["ridge"]))
            _,g,d,lam=min(picks)
            X,_=machine(u,center,mode,gain=g,delay=d,bias_on=bias_on)
            m = Readout(X[:-1], labels, tr, lam)
            pred=m.prediction(X[:-1])
            arms[name]={"val_mae":mae(pred[va],labels[va]),"test_mae":mae(pred[te],labels[te]),
                         "gain":g,"delay_cycles":d,"ridge":lam}
            if export_states: states[name]=X
            # Adapted gain is given matched fixed-loop settings selected on val,
            # not tuned on test. Adaptation is a local homeostatic heuristic.
            if mode=="combined" and bias_on:
                Xa,trace=machine(u,center,mode,gain=g,delay=d,bias_on=True,adapt=.002)
                r,_=_fit(Xa[:-1],labels,tr,va,te)
                r.update(start_gain=g,final_gain=float(trace[-1]),adapt_rate=.002)
                arms["adaptive_biased"]=r
                if export_states: states["adaptive_biased"]=Xa
    # Hidden-state probe is an EXTERNAL test with true hidden labels; not used
    # by any of the above readouts. To count as a distinctive representation,
    # it must outperform an equally informed explicit-history probe.
    probes={}
    probe_labels=driver[:-1]
    for name,X in (("direct_history",taps(u,lag+1)[:-1]),
                   ("leaky_biased",states.get("leaky_biased")),
                   ("pure_echo_biased",states.get("pure_echo_biased")),
                   ("combined_biased",states.get("combined_biased"))):
        if X is None:
            # Recreate selected machine when not retaining trace.
            if name!="direct_history":
                a=arms[name]; mode=name.rsplit("_",1)[0]
                X,_=machine(u,center,mode,gain=a.get("gain",0.),delay=a.get("delay_cycles",1),bias_on=True)
        probes[name],_= _fit(X[:-1] if len(X)==len(u) else X,probe_labels,tr,va,te)
    result={"world":kind,"seed":seed,"n":n,"observation_noise":noise,"hidden_driver_lag":lag,
            "protocol":"causal t -> noisy observed t+1, train 60%, val 20%, test 20%, wash-in 80",
            "arms":arms,"hidden_probe":probes,"symmetry_break":"fixed uniform random bias in [-.65,.65] per slot; bias chosen without labels"}
    if export_states: return result,{"observed":u,"hidden_x":hidden_x,"hidden_driver":driver,**states}
    return result


def benchmark(seeds=SEEDS,n=1200,noises=NOISES):
    receipts=[trial(k,s,n,noise) for k in KINDS for noise in noises for s in seeds]
    summary={}
    for k in KINDS:
        for noise in noises:
            rows=[r for r in receipts if r["world"]==k and r["observation_noise"]==noise]
            summary[f"{k}/noise={noise}"]={name:round(float(np.mean([r["arms"][name]["test_mae"] for r in rows])),6)
                for name in rows[0]["arms"]}
    return {"experiment":"VirtualTuft Gate 2 hidden-state + symmetry audit",
            "status":"synthetic frozen confirmatory seeds; exploratory new world mechanics",
            "freeze":{"seeds":list(seeds),"n":n,"noises":list(noises),"gain_options":list(GAINS),"delay_options":list(DELAYS),"penalties":list(PENALTIES),"bias_seed":2026,"adaptive_rate":.002,"worlds":list(KINDS)},
            "summary_test_mae":summary,"trials":receipts}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action",nargs="?",choices=("benchmark","trace"),default="benchmark")
    ap.add_argument("--world",choices=KINDS,default="henon")
    ap.add_argument("--seed",type=int,default=101)
    ap.add_argument("--n",type=int,default=1200)
    ap.add_argument("--noise",type=float,default=.02)
    ap.add_argument("--output",default="")
    a=ap.parse_args()
    if a.action=="trace":
        result,states=trial(a.world,a.seed,a.n,a.noise,True)
        dst=Path(a.output or "results/gate2_trace.npz");dst.parent.mkdir(exist_ok=True,parents=True)
        np.savez_compressed(dst,**states)
        print(json.dumps(result,indent=2))
    else:
        result=benchmark(n=a.n)
        dst=Path(a.output or "results/gate2.json");dst.parent.mkdir(exist_ok=True,parents=True)
        dst.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
        print(json.dumps(result["summary_test_mae"],indent=2))
    print("Receipt:",dst)

if __name__=="__main__":
    main()
