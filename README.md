# Unified Caching Model (UCM) simulator

Simulation code and results for the survey

> S. Hosny and J. Tadrous, "From Classical to Semantic Caching: A Survey of Wireless Edge Caching."

The UCM puts the four lines of edge caching research in one model. Each line is a layer that can be switched off:

- time: popularity prediction, drift, memory-bounded predictors, and reactive LRU/LFU caches
- people: SP coordination of base-station caches, user participation, and Stackelberg rewards
- place: random-direction mobility and D2D contacts that must last long enough for a transfer
- meaning: semantic coverage of a request by similar items, and the fidelity of a semantic hit

Section II of the paper defines the model, and Section IX reports the experiments below.

## Contents

| Path | What it is |
|---|---|
| `sim/ucm.py` | The simulator: one seeded run of the UCM returns tier shares, hit ratio, latency, fidelity, and SP cost |
| `sim/experiments2.py` | Experiments E1 to E5, E1r, and the closed-form validation (30 seeded runs per point) |
| `sim/figs2.py` | Builds the figures of Section IX from the stored results |
| `sim/res2_*.json` | Raw per-run results for every point, with means and 95% confidence intervals |
| `sim/SIMULATION_PLAN.md` | Design and parameter grounding of the experiments |
| `figs/` | The figures of Section IX (vector PDF) |

## Experiments

| Experiment | Research question | What it varies |
|---|---|---|
| validate | | Oracle, no drift, no D2D: hit ratio against the closed form for C = 10, 20, 40 |
| E1 | RQ1, time | Drift rate and skew for five proactive estimators; predictor memory M |
| E1r | RQ1, time | Drift rate for proactive estimators versus reactive LRU and LFU caches at the BSs (BS-only network) |
| E2 | RQ2, people | Reward level with independent or coordinated BSs, at 2 and 10 m/s |
| E3 | RQ3, place | Speed from 0 to 30 m/s for three device placement rules |
| E4 | RQ4, meaning | Similarity threshold for three BS cache sizes and for coordinated BSs |
| E5 | RQ5, interaction | 2^5 factorial design over the five axes |

## Reproduce

```
pip install -r requirements.txt
./run_all.sh            # figures from the stored results
./run_all.sh --rerun    # also reruns every simulation (about 7,400 runs, roughly 0.5 s each per core)
```

Seeds 0 to 29 are reused across the compared policies (common random numbers), so a rerun reproduces the stored values exactly. The reactive policies (`est='lru'` or `'lfu'`) draw no extra random numbers, so adding them left every earlier result unchanged.

## Main parameters (`DEF` in `sim/ucm.py`)

`F=400` items, Zipf exponent `gamma=0.8`, `K=80` semantic classes, `N=80` devices, 4 BSs in a 600 m square, BS and device caches of 20 and 10 items, D2D range 100 m, transfer time 5 s, speed 2 m/s, drift 0.05 per epoch, 20 epochs of 20 slots (3 warm-up epochs), tier delays 1, 5, 20, 40, 100 ms, backhaul and inter-BS costs 1 and 0.3, reward scale 4. The activity `beta` (probability that a device issues a request in a slot) is 1 in all reported results.

## License

Code: MIT (see `LICENSE`). Results and figures: CC BY 4.0.
