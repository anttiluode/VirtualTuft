# Gate 2 — Hidden state, odd symmetry and a genuine returning echo

**2026-11-23 | Completed simulation, 5 seeds × 2 worlds × 2 noise levels = 20 full trials.**

Gate 1 and Gate 0 are preserved. This Gate responds to Neon's second independent audit and to [Gate 2 preregistration #1](https://github.com/anttiluode/VirtualTuft/issues/1). The code is `gate2.py` and the machine-readable record is `results/gate2.json`. All numbers are **mean held-out MAE**, not NMSE, in normalized coordinates. Lower is better. Tests: `python -m unittest discover -s tests -v` (23 tests as of this gate).

## The question

Can a single time-multiplexed `tanh` element with a delayed feedback line preserve hidden state from a partially observed world? Does recurrence add anything over leak-only memory, nonrecurrent nonlinearity, and **properly biased** polynomial-history models?

In Gate 1, centering the input placed the `tanh` unit near its odd-symmetric region. An odd mapping cannot readily produce the **even-degree** features needed for quadratic Hénon dynamics. A fixed, input-independent per-slot bias is a necessary control; zero-bias results were unintentionally handicapped on these quadratic benchmarks.

## Worlds

1. **Ordinary Hénon projection:** `x[t+1] = 1 − 1.4 x[t]² + 0.3 x[t−1]`. The predictor only observes `u[t]=x[t]+measurement_noise`, not the two-dimensional state. The missing variable is recoverable directly from the previous observation; this is a test of temporal memory, not an impossible latent recovery problem.
2. **Delayed quadratic world `delayed_d3`:** `x[t+1] = 1 − 1.1 x[t]² + 0.3 x[t−3]`. This is **not** canonical Hénon. The ordinary `1.4` coefficient diverges when blindly combined with a three-step delay, so this distinct stable family was declared and used instead. It tests whether the longer-lag driver remains accessible.

Each world runs for 500 additional hidden warm-up steps and 1,200 observed steps. Measurement-noise standard deviations are 0 and 0.02; noise is added **only after** generating the hidden dynamics, never fed back into the physical world. The split is chronological (~60% train, 20% validation, 20% test; first 80 discarded). All gains, delay cycles, and ridge penalties are selected with **validation data only**. Seeds 101, 103, 107, 109, 113 were preregistered in issue #1. Disclosure: one short smoke test used seed 101 and `n=460` before the full run; it exposed a controls bug that was repaired (see next paragraph). Treat seed 101 as partly examined, not an untouched confirmation seed.

**Test-detected control bug repaired before reporting:** An initial version of the `stateless` control accidentally carried the last node's value across the cycle boundary. The `test_stateless_machine_really_forgets_all_earlier_cycles` test FAILED; we added an explicit `boundary` switch and re-ran the full benchmark. Test now passes. The final JSON contains only **post-repair** values.

## Arms and fairness

- `current`: current observation with a quadratic feature, no past information.
- `structured`: deliberately privileged polynomial history with exact lag (1 or 3) and the true form's needed square. This is the **oracle-structured** comparator, not a blind method.
- `quadratic_50`: fifty centered polynomial history features, including ten explicit lags, pairwise products, and cubes. Its basis can express both worlds.
- `knn_archive`: 10 nearest neighbors in a causal `lag+1` delay vector, using training states **only**; its archive size is reported.
- `stateless`: nonlinear cascade with **no across-cycle memory** (explicit boundary reset), tested with zero bias and fixed random bias.
- `leaky`: 35% of each virtual slot's previous-cycle activation retained, **zero explicit delayed feedback**; has cross-cycle slot carry. Tested with/without bias.
- `pure_echo`: no leaky memory (`leak=1`), **no previous-cycle slot boundary**, only an explicit delayed feedback tap; gain selected from .35/.7/.9, delay from 1/2/3. Tested with/without bias.
- `combined`: leaky memory + explicit delayed feedback + within-cycle slot coupling, with gain among 0/.35/.7/.9 and delay 1/2/3, selected on validation; tested with/without bias.
- `adaptive_biased`: the validation-selected `combined_biased` machine plus a tiny **homeostatic** echo-energy update of gain (0.002). This is not the downstream task gradient and is not credited as prediction learning.

All machine configurations use the **same 50-slot input mask** and per-slot bias drawn deterministically with RNG seed 2026. Bias is frozen before training and independent of the hidden world or labels. The controls are not fully compute-capacity matched: virtual nodes differ from explicit polynomial features in complexity, and history kNN has a large archive.

## Frozen results — test MAE, five seeds averaged

| World / observation noise | Current | Structured quadratic | Polynomial 50 | kNN archive | Leaky odd | Leaky biased | Combined odd | Combined biased | Pure echo biased | Adaptive biased |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hénon / 0 | 0.16679 | 0.000002 | 0.000002 | 0.02176 | 0.28422 | 0.05671 | 0.28722 | **0.04217** | 0.14547 | 0.14538 |
| Hénon / 0.02 | 0.16908 | **0.03675** | 0.03704 | 0.04187 | 0.28815 | 0.07321 | 0.28815 | **0.06296** | 0.15067 | 0.14591 |
| Delayed d3 / 0 | 0.14619 | 0.000002 | 0.000005 | 0.03513 | 0.25693 | 0.07921 | 0.20283 | **0.04045** | 0.14143 | 0.14294 |
| Delayed d3 / 0.02 | 0.15168 | **0.03163** | 0.03139 | 0.04688 | 0.25932 | 0.09914 | 0.22140 | **0.07535** | 0.14934 | 0.15131 |

*Boldface in the structured/polynomial columns indicates the best overall performance in each noisy world; boldface in combined biased indicates the best reservoir arm.*

### What survives

1. **Neon's odd-symmetry diagnosis is supported.** On noisy Hénon, `leaky_odd=0.288` and `leaky_biased=0.073`; `combined_odd=0.288` and `combined_biased=0.063`. With zero bias the reservoir's nonlinear odd symmetry is an unfair handicap on quadratic tasks.
2. **A specific explicit-return benefit does survive.** With fixed bias, the combined reservoir beats leak-only memory on both worlds: noisy Hénon `0.06296 < 0.07321`; noisy delayed-d3 `0.07535 < 0.09914`. All five seeds select a **nonzero explicit gain** (`0.35` for Hénon, `0.70` for delayed d3), and delay cycle `1` in the combined reservoir. This supports a limited benefit of the returning signal on these fixed synthetic tasks, not time-delay adaptation.
3. **The stronger mathematical baselines win anyway.** `quadratic_50` scores `0.0370` / `0.0314` on noisy Hénon / delayed-d3. The reservoir has not exceeded explicit nonlinear history at equal *readout dimension*.
4. **A pure-echo loop is not automatically better.** `pure_echo_biased` is much worse than the combined system, including on the longer-lag world. The hypothesis that an intact explicit delay automatically beats a leaking state **fails** in these settings.
5. **Our local adaptation remains unsupported.** `adaptive_biased` loses badly to fixed combined settings; the homeostatic rule changes the physical gain during readout training and evaluation, without optimizing the task. It does not establish that all adaptation fails.

### Hidden-state audit (external ridge probe)

An **external** ridge readout trained with privileged hidden ground truth attempts to infer the missing driver `x[t−1]` or `x[t−3]` from the machine's current state. Mean probe MAE on the noisy world:

| World | Direct lag history | Leaky biased | Pure echo biased | Combined biased |
|---|---:|---:|---:|---:|
| Hénon | **0.0163** | 0.1565 | 0.5339 | 0.2064 |
| Delayed d3 | **0.0163** | 0.2610 | 0.4223 | 0.2213 |

The combined reservoir produces features useful for the task, but its hidden driver is **not linearly readable nearly as well as in explicit lag history**. There is no evidence here for discovering a privileged new latent variable.

## The stop condition

Gate 2 does **not** meet the stop condition for a superior time-as-memory representation: a properly specified explicit-history polynomial system outperforms the reservoir, and a simple lag vector decodes the hidden state better than the reservoir. We do, however, isolate a **modest, genuine effect of the returning feedback** after fixing odd symmetry and accurately ablating leakage.

This is a synthetic dynamical-system experiment, not proof about biological tufts, myelin or anticipating synchronization. No `Luotain2` spatial-FFT audit has been run; reshaping virtual-node positions into an image would be unjustified without separate index scrambling and physically meaningful coordinates.

## Reproduce

```bash
pip install -r requirements.txt  # NumPy
python -m unittest discover -s tests -v
python gate2.py benchmark --output results/gate2.json
python gate2.py trace --world delayed_d3 --seed 103 --noise .02 --output results/gate2_trace.npz
```

Optional Numba accelerates the simulation but is not required. All stored results in `results/gate2.json` are reproducible with NumPy-only fallback, with minor floating-point precision variability possible.
