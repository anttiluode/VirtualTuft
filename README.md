# VirtualTuft — time as memory

**Gate 1 (Nov 23): Independent Neon audit follow-up.** [Methods, controls, seven-world results and stop conditions](GATE1.md) · [causal benchmark code](gate1.py) · [raw JSON receipt](results/gate1.json). A centered polynomial-history baseline **beats** the delayed reservoir on chaotic logistic-map data (test MAE 0.0164 vs 0.0856), and the selected explicit feedback gain is zero on all three chaotic seeds. Gate 0 is preserved as an earlier experiment, not quietly rewritten. The gain-adaptation hypothesis remains **unsupported**.

**Gate 0 | November 23, 2026.** A single nonlinear dynamical node, sampled into 50 sequential virtual time slots, predicts synthetic drop intervals. The goal is to measure when a physical-like delay loop is useful **and when learning its feedback makes it worse**. This is an abstract computational experiment, *not* a model of an actual apical tuft or a demonstration of biological anticipation.

**[Open the offline knob-turning visualizer](index.html)** — open `index.html` in a browser, no server, dependencies, or GPU. For GitHub Pages, publish `main` / root. The graph compares the observed interval with a forecast issued *before* that interval arrived. The heatmap shows the same nonlinear node's 50 successive time slots. The web interface uses an online LMS readout and is **not** the frozen Python benchmark.

## Why

A Takens-style delay-coordinate vector reconstructs useful information about an observable dynamical system from its history. A **delay-based reservoir** uses actual delayed feedback and a nonlinear response to do computations with this history. They are related but not equivalent. This repo asks whether a one-node delay reservoir learns the timing structure of a simple world, and whether changing its feedback locally improves it beyond fixed-delay and direct-history controls.

Companion experiment: Opus's **Pisara** focuses on anticipating synchronization, omissions and learned delay. VirtualTuft focuses on useful temporal representations and the relative value of an adaptive loop. **Luotain2** can later audit the recorded states, but our exported `time × virtual-node` matrix is *not* a homogeneous 2-D spatial field, and must not be passed to its FFT spatial-shift grammar without a justified adapter and controls.

## Run

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python virtualtuft.py benchmark  # saves results/gate0.json
python virtualtuft.py audit --world chaotic --output results/trace.npz
```

Only NumPy is required. The `audit` export contains `intervals`, `fixed_nodes`, `adaptive_nodes`, `taps`, `adaptive_gain`, `forecast_fixed` and `forecast_adaptive` for an external observer. This is numeric data, not a biological measurement.

## Machine and controls

- **Virtual delay loop:** a single `tanh` nonlinear element with leaky response, processed in 50 successive slots per observed interval. Each slot is coupled to the preceding slot and to that slot's state 1–3 cycles earlier. Fixed random input mask. No GPU or backpropagation through time.
- **Fixed-loop control:** feedback gain chosen from `{0, .35, .70, .90}` and delay (whole input cycles) from `{1,2,3}` by the validation split *only*. Readout fitted with ridge regression on the training split.
- **Adaptive loop:** begins at the same validation-selected gain and delay, then updates its feedback gain from a **causal local echo discrepancy** between the current observed input and the returning signal. Delay itself does not adapt in Gate 0. Its ridge readout is trained on the same initial training interval. If adaptation hurts, this is a negative result.
- **Explicit-history control:** linear ridge readout on 50 direct input lags, with the same readout penalty.
- **Clock and persistence controls:** training-set mean, and simply use the last observed interval as the next forecast.

In the Python experiment, **all forecasts at index t use observations only up to t**; the target is interval `t+1`. The first 80 cycles are discarded as wash-in. Fits use roughly the first 60% of each sequence, hyperparameters use the next 20%, and the final 20% is held out. The adapting feedback does continue to receive newly observed inputs during the test—this is causal online adaptation, not a fixed-parameter test. Reported MAE values are in the same arbitrary units as the simulated intervals. Models are comparable in number of readout features but not strictly matched in total dynamical state, computational cost or trainable degrees of freedom.

## Synthetic worlds

`periodic` (logistic `r=2.9`), `period2` (`r=3.2`), `chaotic` (`r=3.9`), `draining` (increasing gaps with an abrupt refill every 390 events), `switch` (a sequence changing from periodic to period-two to chaotic), and `shuffled` (the chaotic interval histogram with temporal order destroyed). Small independent additive input noise is included. **These are not Shaw's measured faucet data.** We do not claim that a logistic map captures the physics of a faucet. An omitted-drop test, phase-matched surrogate test, physical propagation delays and unseen distribution-shift benchmark are future gates, not things already demonstrated.

## Gate 0 result — synthetic, three seeds

Run `python virtualtuft.py benchmark` to reproduce the JSON receipt with seeds 11, 17 and 29. Test MAE averages (smaller is better):

| World | Training-mean clock | Persistence | Linear 50 taps | Best fixed loop | Local adaptive loop |
| --- | ---: | ---: | ---: | ---: | ---: |
| Periodic | 0.0066 | 0.0092 | 0.0066 | 0.0066 | 0.0066 |
| Period-two | 0.0858 | 0.1716 | **0.0067** | 0.0089 | 0.0086 |
| Chaotic | 0.1630 | 0.2850 | 0.1274 | **0.0581** | 0.1037 |
| Draining | 0.0731 | 0.0093 | 0.0087 | 0.0086 | **0.0085** |
| Regime switch | 0.1611 | 0.2809 | 0.1732 | **0.1367** | 0.1454 |
| Shuffled control | 0.1623 | 0.2002 | 0.1664 | 0.1725 | **0.1647** |

**Interpretation.** The fixed nonlinear loop substantially outperforms a linear direct-history readout on one *matched* chaotic logistic world in this limited setting. The linear taps win period-two; neither learning nor nonlinear loop gives a meaningful advantage on a nearly constant periodic signal. The adaptive gain generally fails to improve the strongest fixed-loop setting and is worse on chaos. This is *not* evidence for anticipatory synchronization or for a biological self-model. The fixed-loop parameters were selected per world by validation; a fuller assessment needs more seeds, hidden world families, stronger nonlinear history baselines, a fair adaptation hyperparameter sweep, and true out-of-distribution dynamics.

### Stop conditions and next gates

1. If flexible nonlinear lag-history predictors match or beat the fixed loop at equal budget, there is no special computational advantage here.
2. If local gain adaptation cannot beat validation-selected fixed loops on unseen drift and abrupt regime shifts, stop claiming a learning advantage. Gate 0 currently **does not** establish such an advantage.
3. If an apparent predictive lead vanishes when comparing at equal forecast horizons and controlling excitation or phase, it is not anticipating synchronization.
4. If reshaping virtual nodes into a spatial field creates an apparent Luotain2 law that disappears under index permutations or independent reruns, it's an audit artifact, not discovered physics.

## Background and limits

- Appeltant et al. (2011), [*Information processing using a single dynamical node as complex system*](https://doi.org/10.1038/ncomms1476).
- Hart, Hook & Dawes (2020), [*Embedding and approximation theorems for echo state networks*](https://doi.org/10.1016/j.neunet.2019.12.013). Mathematical reservoir embeddings are not claims about specific neural dendrites.
- Dambre et al. (2012), [*Information processing capacity of dynamical systems*](https://doi.org/10.1038/srep00514). Actual information-processing capacity has **not yet been measured** in Gate 0.
- Shaw (1984), *The Dripping Faucet as a Model Chaotic System*. Its actual experimental regimes and information measures require separately obtained recorded data.

No real EEG, no myelin simulation, no actual synaptic mechanisms, no biological recordings, and no claimed extraction of physical hidden variables. The point is a falsifiable little machine you can inspect and improve.