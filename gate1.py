"""VirtualTuft Gate 1: independent Neon's audit follow-up (2026-11-23).

Preserves Gate 0 code and receipts. Adds nonlinear history, a genuine no-temporal-
recurrence control, train-only centering, decoupled local gain learning, and
prequential online readout controls. All synthetic. No biological claims.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from virtualtuft import make_world, split_indices, mae, taps

WORLDS = ("periodic", "period2", "chaotic", "draining", "switch", "switch_repeating", "shuffled")
SEEDS = (11, 17, 29)
PENALTIES = (.01, .1, 1.0)
ONLINE_ETAS = (.005, .02, .08, .25)
GAINS = (0.0, .35, .70, .90)
DELAYS = (1, 2, 3)


def repeated_switch(n: int, seed: int, noise: float, block: int = 75) -> np.ndarray:
    """Changing logistic-map worlds with *repeated* transitions in each partition.

    Starts a new trajectory on every block; transitions include a reset and
    are not a continuous physical faucet. This is intentionally a separate
    generalization benchmark from Gate 0's zero-shot-chaos `switch`.
    """
    rng = np.random.default_rng(seed)
    pieces = []
    for i, start in enumerate(range(0, n, block)):
        m = min(block, n - start)
        r = (2.9, 3.2, 3.9)[i % 3]
        # Each segment is generated from a distinct RNG seed, with its own wash-in.
        # Noise is added only once below (not internally to the logistic iterates).
        x = float(rng.uniform(.15, .85))
        for _ in range(220):
            x = r * x * (1 - x)
        a = np.empty(m)
        for j in range(m):
            x = r * x * (1 - x)
            a[j] = x
        pieces.append(a)
    y = .25 + .60 * np.concatenate(pieces)
    y += rng.normal(0, noise, n)
    return np.clip(y, .05, 1.0)


def world(kind: str, n: int, seed: int, noise: float) -> np.ndarray:
    return repeated_switch(n, seed, noise) if kind == "switch_repeating" else make_world(kind, n, seed, noise)


def nonlinear_taps(stream: np.ndarray, center: float, count: int = 50) -> np.ndarray:
    """50 causal features: 10 centered lags, 10 squares, 28 products, 2 cubes.

    Centering uses *only training inputs*, and all feature values at t use
    observations at or before t. Input history is compared at equal readout
    budget with the 50-node virtual reservoir.
    """
    if count != 50:
        raise ValueError("Gate 1 predeclares 50 nonlinear-history features")
    z = taps(stream, 10) - center
    cols = [z[:, k] for k in range(10)]
    cols += [z[:, k] ** 2 for k in range(10)]
    cols += [z[:, i] * z[:, j] for i in range(8) for j in range(i + 1, 8)]
    cols.extend((z[:, 0] ** 3, z[:, 1] ** 3))
    return np.column_stack(cols)


@dataclass(frozen=True)
class Config:
    nodes: int = 50
    gain: float = .7
    delay: int = 1
    center: float = .55
    leak: float = .7
    slot_coupling: float = .5
    input_strength: float = 1.6
    mask_seed: int = 2026
    recurrence: bool = True
    adapt_rate: float = 0.0
    mean_rate: float = .01
    predictor_rate: float = .03


class Loop:
    """Sequential virtual-node loop; one nonlinear element, sampled N times.

    If recurrence=False: *all* between-cycle state, slot boundary carry and
    feedback are removed. Same-cycle nonlinear slot-to-slot propagation remains,
    but state at t is exactly independent of times <t.

    If adapt_rate > 0: an independently learned echo coefficient predicts
    input from delayed activity, while gain is a separate physical parameter.
    A bounded local rule modifies gain using the *current observed input* and
    earlier echo only. This heuristic is NOT a correct gradient for long-horizon
    prediction or evidence of biologically plausible plasticity.
    """
    def __init__(self, c: Config):
        if c.nodes < 2 or c.delay < 1 or not (0 < c.leak <= 1):
            raise ValueError("invalid loop configuration")
        self.c = c
        self.mask = np.random.default_rng(c.mask_seed).choice([-1., 1.], c.nodes)
        self.state = np.zeros(c.nodes)
        self.history: list[np.ndarray] = []
        self.gain = c.gain
        self.echo_weight = 0.0
        self.mean = c.center
        self.gain_trace: list[float] = []

    def step(self, u: float) -> np.ndarray:
        c = self.c
        delayed = self.history[-c.delay] if c.recurrence and len(self.history) >= c.delay else np.zeros(c.nodes)
        if c.recurrence and c.adapt_rate:
            echo = float(np.mean(self.mask * delayed))
            e = float(u) - (self.mean + self.echo_weight * echo)
            # The auxiliary echo predictor weight is NOT the reservoir gain.
            self.echo_weight += c.predictor_rate * e * echo / (.05 + echo * echo)
            self.mean += c.mean_rate * (float(u) - self.mean)
            self.gain = float(np.clip(self.gain + c.adapt_rate * e * echo / (.05 + echo * echo), 0.0, .95))
        new = np.empty(c.nodes)
        previous_slot = float(self.state[-1]) if c.recurrence else 0.0
        for i in range(c.nodes):
            drive = c.input_strength * self.mask[i] * (float(u) - c.center) + c.slot_coupling * previous_slot
            if c.recurrence:
                drive += self.gain * delayed[i]
            instant = np.tanh(drive)
            new[i] = ((1 - c.leak) * self.state[i] + c.leak * instant) if c.recurrence else instant
            previous_slot = new[i]
        self.state = new
        if c.recurrence:
            self.history.append(new.copy())
            if len(self.history) > c.delay:
                self.history.pop(0)
        self.gain_trace.append(self.gain)
        return new.copy()

    def transform(self, stream: np.ndarray) -> np.ndarray:
        return np.array([self.step(float(u)) for u in stream])


class Readout:
    def __init__(self, X: np.ndarray, y: np.ndarray, tr: np.ndarray, penalty: float):
        train = X[tr]
        self.mean = np.mean(train, axis=0)
        # Do not explosively amplify rare or constant virtual slots.
        self.sd = np.maximum(np.std(train, axis=0), .02)
        Z = np.column_stack((np.ones(len(tr)), (train - self.mean) / self.sd))
        reg = np.eye(Z.shape[1]) * penalty
        reg[0, 0] = 0.0
        self.w = np.linalg.solve(Z.T @ Z + reg, Z.T @ y[tr])

    def features(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack((np.ones(len(X)), (X - self.mean) / self.sd))

    def prediction(self, X: np.ndarray) -> np.ndarray:
        return self.features(X) @ self.w

    def online_prediction(self, X: np.ndarray, y: np.ndarray, start: int, eta: float = .25) -> np.ndarray:
        """Prequential: predict t+1, then update ONLY AFTER it has arrived.

        Train-optimized initial readout is frozen through the training window;
        subsequent labels in validation/test are revealed one at a time, causally.
        """
        Z = self.features(X)
        w = self.w.copy()
        predictions = np.full(len(y), np.nan)
        predictions[:start] = Z[:start] @ w
        for t in range(start, len(y)):
            v = Z[t]
            predictions[t] = float(v @ w)
            err = y[t] - predictions[t]
            w += eta * err * v / (1. + v @ v)
        return predictions


def select_online(m: Readout, X: np.ndarray, labels: np.ndarray, va: np.ndarray):
    """Select online learning rate using prequential validation ONLY."""
    options = []
    for eta in ONLINE_ETAS:
        forecast = m.online_prediction(X, labels, int(va[0]), eta)
        options.append((mae(forecast[va], labels[va]), eta, forecast))
    return min(options, key=lambda z: (z[0], z[1]))


def fit_selected(X, labels, tr, va, penalties=PENALTIES):
    options = []
    for lam in penalties:
        m = Readout(X, labels, tr, lam)
        v = mae(m.prediction(X[va]), labels[va])
        options.append((v, lam, m))
    options.sort(key=lambda r: (r[0], r[1]))
    return options[0]


def trial(kind: str, seed: int = 11, n: int = 1050, noise: float = .008,
          return_traces: bool = False):
    if n < 450:
        raise ValueError("Gate 1 needs >=450 steps for train/val/test")
    u = world(kind, n, seed, noise)
    tr, va, te = split_indices(n)
    labels = u[1:]
    center = float(np.mean(u[tr]))
    x_lin = taps(u, 50)[:-1]
    x_quad = nonlinear_taps(u, center)[:-1]
    # Same-cycle nonlinear computation, but NO state across input cycles.
    x_norecur = Loop(Config(center=center, recurrence=False)).transform(u)[:-1]

    arms = {
        "clock_mean": {"test_mae": mae(np.full(len(te), np.mean(labels[tr])), labels[te])},
        "persistence": {"test_mae": mae(u[te], labels[te])},
    }
    frozen_traces = {}
    for name, X in (("linear_taps", x_lin), ("quadratic_taps", x_quad), ("no_recurrence", x_norecur)):
        val, lam, model = fit_selected(X, labels, tr, va)
        p = model.prediction(X)
        arms[name] = {"val_mae": val, "test_mae": mae(p[te], labels[te]), "ridge": lam}
        frozen_traces[name] = p

    # Select loop gain, delay AND ridge on validation, never on test.
    picks = []
    for gain in GAINS:
        for delay in DELAYS:
            X = Loop(Config(center=center, gain=gain, delay=delay)).transform(u)[:-1]
            v, lam, m = fit_selected(X, labels, tr, va)
            picks.append((v, gain, delay, lam))
    picks.sort()
    _, gain, delay, lam = picks[0]
    fixed_machine = Loop(Config(center=center, gain=gain, delay=delay))
    X_fixed = fixed_machine.transform(u)[:-1]
    m_fixed = Readout(X_fixed, labels, tr, lam)
    p_fixed = m_fixed.prediction(X_fixed)
    v_fixed_online, eta_fixed, p_fixed_online = select_online(m_fixed, X_fixed, labels, va)
    arms["fixed_loop"] = {"val_mae": mae(p_fixed[va], labels[va]), "test_mae": mae(p_fixed[te], labels[te]),
                          "gain": gain, "delay": delay, "ridge": lam}
    arms["fixed_online"] = {"val_mae": v_fixed_online, "test_mae": mae(p_fixed_online[te], labels[te]), "online_eta": eta_fixed}

    # Adapt local gain causally. The auxiliary echo predictor is NOT the gain.
    adaptive_machine = Loop(Config(center=center, gain=gain, delay=delay, adapt_rate=.012))
    X_adapt = adaptive_machine.transform(u)[:-1]
    # Adapted states have a moving distribution. Give BOTH adapted and fixed
    # machines matched online-readout evaluations, without a post-hoc re-tune.
    m_adapt = Readout(X_adapt, labels, tr, lam)
    p_adapt = m_adapt.prediction(X_adapt)
    v_adapt_online, eta_adapt, p_adapt_online = select_online(m_adapt, X_adapt, labels, va)
    arms["adaptive_loop"] = {"test_mae": mae(p_adapt[te], labels[te]),
                              "val_mae": mae(p_adapt[va], labels[va]),
                              "start_gain": gain, "final_gain": adaptive_machine.gain,
                              "echo_predictor_weight": adaptive_machine.echo_weight,
                              "adapt_rate": .012}
    arms["adaptive_online"] = {"val_mae": v_adapt_online, "test_mae": mae(p_adapt_online[te], labels[te]), "online_eta": eta_adapt}
    out = {
        "world": kind, "seed": seed, "n": n, "noise_sd": noise,
        "train_end_forecast": int(tr[-1]), "val_end_forecast": int(va[-1]),
        "train_only_center": center,
        "protocol": "1-step causal forecast, train 60%, val 20%, test 20%, first 80 wash-in",
        "switch_interpretation": "zero-shot chaos; test never has seen chaos in training" if kind == "switch"
            else "repeated within-split transitions" if kind == "switch_repeating" else "stationary or predefined reset",
        "selection": {"fixed_gain": gain, "fixed_delay_cycles": delay, "ridge": lam},
        "arms": arms,
    }
    if return_traces:
        return out, {
            "intervals": u, "fixed_nodes": X_fixed, "adaptive_nodes": X_adapt,
            "nonrecurrent_nodes": x_norecur, "fixed_forecast": p_fixed,
            "fixed_online_forecast": p_fixed_online, "adaptive_forecast": p_adapt,
            "adaptive_online_forecast": p_adapt_online,
            "gain_trace": np.array(adaptive_machine.gain_trace),
            "quadratic_forecast": frozen_traces["quadratic_taps"],
        }
    return out


def benchmark(seeds=SEEDS, n=1050, noise=.008):
    ts = [trial(k, seed, n, noise) for k in WORLDS for seed in seeds]
    all_arms = tuple(ts[0]["arms"])
    summary = {kind: {name: round(float(np.mean([r["arms"][name]["test_mae"] for r in ts if r["world"] == kind])), 6)
                      for name in all_arms} for kind in WORLDS}
    return {"experiment": "VirtualTuft Gate 1 / Neon audit response", "status": "synthetic",
            "freeze": {"seeds": list(seeds), "n": n, "noise": noise,
                       "ridge_penalties": list(PENALTIES), "online_eta_choices": list(ONLINE_ETAS), "feedback_gains": list(GAINS),
                       "delays": list(DELAYS), "readout_online_eta": "selected on prequential validation only",
                       "adapt_gain_eta": .012, "nonlinear_history_features": 50},
            "summary_test_mae": summary, "trials": ts}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("benchmark", "trace"), nargs="?", default="benchmark")
    p.add_argument("--world", choices=WORLDS, default="chaotic")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--n", type=int, default=1050)
    p.add_argument("--noise", type=float, default=.008)
    p.add_argument("--output", default=None)
    args = p.parse_args()
    if args.action == "trace":
        result, traces = trial(args.world, args.seed, args.n, args.noise, return_traces=True)
        dst = Path(args.output or "results/gate1_trace.npz")
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, **traces)
        print(json.dumps(result, indent=2))
        print(f"Trace: {dst}")
    else:
        result = benchmark(n=args.n, noise=args.noise)
        dst = Path(args.output or "results/gate1.json")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
        print(json.dumps(result["summary_test_mae"], indent=2))
        print(f"Receipt: {dst}")

if __name__ == "__main__":
    main()
