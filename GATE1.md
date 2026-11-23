# Gate 1 — Neon audit correction (November 23, 2026)

**Status:** completed synthetic ablation, **negative for a special benefit from delayed feedback**. Gate 0's original script, receipt, and visualizer remain untouched. This is an *additional* experiment, not a replacement for the frozen Gate 0 observations.

## Why Gate 1 exists

Neon's independent source/receipt audit found that Gate 0 (a) compared a nonlinear reservoir to only *linear* lag taps, (b) called a `feedback=0` reservoir a zero-feedback loop even though leaky/slot carry still made it recurrent, (c) used an arbitrary `0.55` offset in the local gain update, while applying the *same* parameter to input prediction and physical feedback, and (d) presented a `switch` whose training split included no chaotic samples. All four are real concerns, and the first two substantially change what we can claim.

Gate 1 leaves Gate 0 in place for provenance and adds controlled comparisons:

- **Centered quadratic history:** 50 features = 10 direct interval lags + 10 squares + 28 cross-products + 2 cubes, centered using **training inputs only**. Identical readout-feature count to the virtual reservoir, not identical cost or dynamical-state dimension.
- **Truly no between-cycle recurrence:** on each input, a fresh stateless 50-slot nonlinear cascade; its output is *exactly independent* of any preceding input, verified by an equality test. Same-cycle slot interactions remain.
- **Fixed loop:** one sequential `tanh` unit sampled in 50 slots per observed interval; leaky memory plus optional delayed feedback. Fixed feedback gain and delay and ridge penalty chosen on the validation partition.
- **Adaptive loop:** the earlier hard-coded-error estimate is replaced with a running input mean and separate local echo-prediction coefficient. The physical feedback gain gets its **own** bounded heuristic update; no backprop-through-time or future labels. This is exploratory, not a validated biological plasticity rule.
- **Prequential online readout:** separately test fixed and adaptive reservoirs with readout weights updated **only after** the newly predicted interval arrives. Step size selected by prequential validation, not test.
- **Two distinct switch worlds:** Gate 0 `switch` is now labeled *zero-shot chaotic extrapolation*; a new `switch_repeating` cycles through regimes every 75 inputs, putting multiple abrupt changes into each data split. Both are logistic-map surrogates, not physical faucet data.

All features at forecast index `t` depend on observations no later than `t`, and predict interval `t+1`. Training is ~60% of the sequence, validation ~20%, test ~20%, with first 80 cycles discarded as wash-in. Hyperparameter selection is validation-only. Online evaluation reveals each target *after* issuing its forecast.

## Frozen result (3 seeds: 11, 17, 29; 1,050 intervals each)

Mean held-out test **MAE** in normalized interval units. Lower is better. Full per-seed records: [`results/gate1.json`](results/gate1.json).

| World | Linear taps | Centered quadratic taps | No recurrence | Fixed loop | Adaptive loop | Fixed + online readout | Adaptive + online |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| periodic | 0.00679 | 0.00660 | **0.00658** | 0.00668 | 0.00667 | 0.00667 | 0.00666 |
| period2 | 0.00679 | 0.00697 | **0.00669** | 0.00690 | 0.00868 | 0.00691 | 0.00838 |
| chaotic | 0.12738 | **0.01641** | 0.10213 | 0.08561 | 0.08924 | 0.08798 | 0.09138 |
| draining | 0.00876 | **0.00731** | 0.00918 | 0.00739 | 0.00757 | 0.00739 | 0.00749 |
| switch (zero-shot chaos) | 0.17963 | 0.17433 | **0.09692** | 0.14619 | 0.14648 | 0.14517 | 0.14775 |
| switch_repeating | 0.05690 | **0.01745** | 0.03381 | 0.03819 | 0.03751 | 0.03600 | 0.03679 |
| shuffled null | 0.16639 | 0.16794 | 0.16354 | 0.16730 | **0.16225** | 0.16437 | 0.16288 |

**The central correction:** in Gate 0, the fixed loop beat a linear history predictor on the chaotic logistic-map world (0.0581 vs 0.1274 MAE), but Gate 1's stronger nonlinear history baseline achieves **0.0164**. Gate 1 also changes centering and model/penalty selection, so the two gate's loop numbers should not be treated as a controlled before/after improvement. The broader implication stands: the original evidence cannot establish a special advantage of delayed feedback, because the nonlinear history control was missing.

**Explicit delayed feedback was selected at gain `0.0` on all three chaotic seeds**, as well as the repeated-switch and draining worlds. The selected `fixed_loop` still has leaky per-slot/cross-slot memory, but no explicit delayed feedback contribution. On zero-shot switch, a *stateless* nonlinear cascade beats the loop, illustrating how misleading a distribution shift can be.

The revised local gain adaptation **does not convincingly improve prediction**. The small differences on `shuffled` are not grounds for attributing useful learning to the model. The prequential online readout also fails to produce a robust advantage in this limited benchmark.

## Honest limits

- Three seeds, chosen synthetic worlds, no physical time calibration, no continuous-time neuron or literal axonal conduction delay.
- Prediction of **next interval given the just-observed interval**, not the future before its causes are observed. A `tanh` map plus polynomial history is a natural competitor for a logistic map.
- A 50-feature history model and 50 virtual nodes are readout-size matched only. Total capacity, effective degrees of freedom, computational cost and adaptation budget are not fully matched.
- The no-recurrence control is a nonlinear within-cycle cascade, not 50 independent physical units.
- The adaptive gain heuristic is not a calculated gradient of downstream forecast error; its failure is about this rule, not all local plasticity.
- No Luotain2 spatial FFT audit, dendritic modeling, myelin simulation, EEG, or claim about actual cortical mechanisms.
- Gate 1 does not establish anticipation or a superior virtual-tuft computer. It motivates a second task where history alone is harder to decode, such as a genuinely partially observed multi-dimensional source, before making stronger claims.

## Reproduce

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python gate1.py benchmark --output results/gate1.json
python gate1.py trace --world chaotic --seed 11 --output results/gate1_trace.npz
```

The trace contains the observed intervals, fixed/adaptive virtual-node matrices, truly nonrecurrent states, gain trajectory and prequential forecasts. Rows are *time* and columns are *virtual-node slots*, not a physically homogeneous spatial image. The FFT spatial-shift grammar from Luotain2 cannot be applied directly to this matrix without an explicitly validated adapter.
