"""VirtualTuft Gate 0 — an auditable time-multiplexed delay reservoir.

One sequential nonlinear node is sampled in N slots per input interval. The
previous cycle's slot values are the feedback delay state. A causal readout
predicts the NEXT observed interval. Only NumPy is required.

This is a computational toy, not a biophysical tuft model or proof of
anticipating synchronization. The benchmark distinguishes passive memory from
an explicitly specified local adaptation rule.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np

WORLDS = ("periodic", "period2", "chaotic", "draining", "switch", "shuffled")


def _logistic(n: int, r: float, rng: np.random.Generator) -> np.ndarray:
    x = float(rng.uniform(.15, .85))
    out = np.empty(n)
    for k in range(n + 220):
        x = r * x * (1.0 - x)
        if k >= 220:
            out[k - 220] = x
    return out


def make_world(kind: str, n: int = 1050, seed: int = 11, noise: float = .008) -> np.ndarray:
    """Positive inter-drop intervals, in arbitrary time units.

    'periodic', 'period2' and 'chaotic' are LOGISTIC-MAP SURROGATES, not
    measurements from Shaw's faucet. 'draining' is a slowly increasing interval
    with occasional abrupt refills. 'shuffled' destroys chronological structure
    while preserving the interval histogram of the chaotic source.
    """
    if kind not in WORLDS:
        raise ValueError(f"Unknown world {kind!r}; choose from {WORLDS}")
    if n < 120:
        raise ValueError("Need at least 120 observations")
    rng = np.random.default_rng(seed)
    if kind in ("periodic", "period2", "chaotic"):
        r = {"periodic": 2.9, "period2": 3.2, "chaotic": 3.9}[kind]
        x = _logistic(n, r, rng)
    elif kind == "draining":
        # Increasing gaps = slowing drops. Abrupt reset emulates a refill.
        phase = np.arange(n) % 390
        x = 1.0 - np.exp(-phase / 120.0)
    elif kind == "switch":
        segments = np.array_split(np.arange(n), 3)
        x = np.empty(n)
        for ids, r in zip(segments, (2.9, 3.2, 3.9)):
            x[ids] = _logistic(len(ids), r, rng)
    else:  # shuffled matched-marginal null
        x = _logistic(n, 3.9, rng)
        x = rng.permutation(x)
    y = .25 + .60 * x
    if noise:
        y = y + rng.normal(0, noise, n)
    return np.clip(y, .05, 1.0)


@dataclass(frozen=True)
class LoopConfig:
    nodes: int = 50
    feedback: float = .55
    leak: float = .70
    slot_coupling: float = .50
    input_strength: float = 1.60
    delay_cycles: int = 1
    adapt_rate: float = 0.0
    mask_seed: int = 2026


class VirtualTuft:
    """One nonlinear element sampled over virtual time slots.

    For slot i during cycle t:
       x_i <- (1-leak)*x_i(previous cycle)
              + leak*tanh(input*mask_i + slot_coupling*x_(i-1)
                          + feedback*x_i(t-delay_cycles))
    The i-1 slot is computed earlier in the SAME cycle. Slot 0 receives the
    final slot from the previous cycle. This is an abstract time-multiplexed
    delay reservoir, not a spatial neural network.

    The optional adaptation is a deliberately small, explicit LOCAL rule:
       echo = mean(mask_i * delayed_slot_i)
       feedback += eta * (current_input - (0.55 + feedback*echo)) * echo
    It updates only AFTER current_input is observed. It does NOT optimize the
    downstream task's error or use gradients through time. No benefit is assumed.
    """

    def __init__(self, config: LoopConfig):
        if config.nodes < 2 or config.delay_cycles < 1:
            raise ValueError("nodes >= 2 and delay_cycles >= 1 are required")
        if not 0 < config.leak <= 1:
            raise ValueError("leak must be in (0,1]")
        self.cfg = config
        self.mask = np.random.default_rng(config.mask_seed).choice([-1.0, 1.0], config.nodes)
        self.reset()

    def reset(self):
        self.history: list[np.ndarray] = []
        self.state = np.zeros(self.cfg.nodes, dtype=float)
        self.feedback = self.cfg.feedback
        self.gain_history = []

    def step(self, interval: float) -> np.ndarray:
        c = self.cfg
        if len(self.history) >= c.delay_cycles:
            delayed = self.history[-c.delay_cycles]
        else:
            delayed = np.zeros(c.nodes)
        if c.adapt_rate:
            echo = float(np.mean(self.mask * delayed))
            e = float(interval) - (.55 + self.feedback * echo)
            self.feedback = float(np.clip(self.feedback + c.adapt_rate * e * echo, -.95, .95))
        new = np.empty(c.nodes)
        previous_slot = float(self.state[-1])
        centered_input = float(interval) - .55
        for i in range(c.nodes):
            drive = (c.input_strength * self.mask[i] * centered_input
                     + c.slot_coupling * previous_slot
                     + self.feedback * delayed[i])
            new[i] = (1.0 - c.leak) * self.state[i] + c.leak * np.tanh(drive)
            previous_slot = new[i]
        self.state = new
        self.history.append(new.copy())
        if len(self.history) > c.delay_cycles:
            self.history.pop(0)
        self.gain_history.append(self.feedback)
        return new.copy()

    def transform(self, stream: np.ndarray) -> np.ndarray:
        return np.array([self.step(float(x)) for x in stream])


def taps(stream: np.ndarray, count: int = 50) -> np.ndarray:
    """Explicit causal input history including current interval as first tap."""
    x = np.zeros((len(stream), count))
    for k in range(count):
        if k == 0:
            x[:, 0] = stream
        else:
            x[k:, k] = stream[:-k]
    return x


def ridge_fit(features: np.ndarray, targets: np.ndarray, penalty: float = .03):
    """Fit intercept without penalty and penalize readout weights."""
    design = np.column_stack((np.ones(len(features)), features))
    reg = np.eye(design.shape[1]) * penalty
    reg[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + reg, design.T @ targets)


def predict(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones(len(features)), features)) @ weights


def split_indices(n: int):
    # A forecast made after observing t is scored against the interval at t+1.
    # Discard first 80 forecasts to allow state/history wash-in.
    end = n - 1
    return (np.arange(80, int(.60 * end)),
            np.arange(int(.60 * end), int(.80 * end)),
            np.arange(int(.80 * end), end))


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def _fit_score(features, stream, tr, va, te, penalty=.03):
    labels = stream[1:]
    x = features[:-1]
    w = ridge_fit(x[tr], labels[tr], penalty)
    pred = predict(x, w)
    return {"val_mae": mae(pred[va], labels[va]),
            "test_mae": mae(pred[te], labels[te]),
            "forecast": pred, "weights": w}


def run_trial(kind: str, seed: int = 11, n: int = 1050, nodes: int = 50,
              noise: float = .008, return_states: bool = False):
    stream = make_world(kind, n, seed, noise)
    tr, va, te = split_indices(n)
    labels = stream[1:]
    constant_mean = float(np.mean(labels[tr]))
    base = {
        "clock_mean": {"test_mae": mae(np.full(len(te), constant_mean), labels[te])},
        "persistence": {"test_mae": mae(stream[te], labels[te])},
    }
    hist = _fit_score(taps(stream, nodes), stream, tr, va, te)
    base["linear_taps"] = {"test_mae": hist["test_mae"], "val_mae": hist["val_mae"]}

    # Sweep only on validation data. This gives the FIXED reservoir a strong
    # control, not an artificially weak arbitrary feedback gain.
    candidates = []
    for gain in (0.0, .35, .70, .90):
        for delay in (1, 2, 3):
            c = LoopConfig(nodes=nodes, feedback=gain, delay_cycles=delay)
            machine = VirtualTuft(c)
            states = machine.transform(stream)
            fit = _fit_score(states, stream, tr, va, te)
            candidates.append((fit["val_mae"], gain, delay, fit["test_mae"], states))
    candidates.sort(key=lambda z: (z[0], z[1], z[2]))
    _, best_gain, best_delay, fixed_test, fixed_states = candidates[0]
    base["fixed_loop"] = {"test_mae": fixed_test, "val_mae": candidates[0][0],
                          "gain": best_gain, "delay_cycles": best_delay}

    # Adaptive arm starts at the same (validation-selected) parameters, then
    # uses ONLY the local sensory-echo rule to adapt. Both readouts are learned
    # on the same training window; no downstream weight updates on test data.
    adapt = VirtualTuft(LoopConfig(nodes=nodes, feedback=best_gain,
                                  delay_cycles=best_delay, adapt_rate=.18))
    adaptive_states = adapt.transform(stream)
    fitted = _fit_score(adaptive_states, stream, tr, va, te)
    base["adaptive_loop"] = {"test_mae": fitted["test_mae"],
                             "val_mae": fitted["val_mae"],
                             "start_gain": best_gain,
                             "final_gain": float(adapt.feedback),
                             "delay_cycles": best_delay}
    result = {"world": kind, "seed": seed, "length": n, "noise_sd": noise,
              "nodes": nodes, "protocol": "train 60%, validate 20%, test 20%; 80-step wash-in",
              "arms": base}
    if return_states:
        return result, {"intervals": stream, "fixed_nodes": fixed_states,
                        "adaptive_nodes": adaptive_states,
                        "taps": taps(stream, nodes), "adaptive_gain": np.array(adapt.gain_history),
                        "forecast_fixed": _fit_score(fixed_states, stream, tr, va, te)["forecast"],
                        "forecast_adaptive": fitted["forecast"]}
    return result


def benchmark(seeds=(11, 17, 29), length=1050, nodes=50, noise=.008):
    trials = [run_trial(world, seed, length, nodes, noise)
              for world in WORLDS for seed in seeds]
    summary = {}
    arms = ("clock_mean", "persistence", "linear_taps", "fixed_loop", "adaptive_loop")
    for w in WORLDS:
        ts = [r for r in trials if r["world"] == w]
        summary[w] = {a: float(np.mean([r["arms"][a]["test_mae"] for r in ts])) for a in arms}
    return {"experiment": "VirtualTuft Gate 0", "status": "synthetic benchmark",
            "note": "Not a biological neuron or demonstration of anticipation. MAE in interval units; lower is better.",
            "config": {"seeds": list(seeds), "length": length, "nodes": nodes,
                       "noise_sd": noise, "validation_selected_fixed_gains": [0, .35, .70, .90],
                       "validation_selected_delays": [1, 2, 3], "ridge": .03},
            "summary_test_mae": summary, "trials": trials}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["benchmark", "audit"], nargs="?", default="benchmark")
    ap.add_argument("--world", choices=WORLDS, default="chaotic")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--length", type=int, default=1050)
    ap.add_argument("--nodes", type=int, default=50)
    ap.add_argument("--noise", type=float, default=.008)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    if args.command == "audit":
        record, traces = run_trial(args.world, args.seed, args.length, args.nodes,
                                   args.noise, return_states=True)
        path = Path(args.output or "results/trace.npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **traces)
        print(json.dumps(record, indent=2))
        print(f"Saved {path}: time x virtual-node matrices. The rows are NOT spatial pixels.")
    else:
        record = benchmark(length=args.length, nodes=args.nodes, noise=args.noise)
        path = Path(args.output or "results/gate0.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record["summary_test_mae"], indent=2))
        print(f"Saved {path}")


if __name__ == "__main__":
    main()