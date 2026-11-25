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
| `sim/run_sim.py` | Command-line runner: set any parameter, sweep one, get hit ratio, tier shares, latency, fidelity, SP cost with 95% CIs |
| `sim/experiments2.py` | Experiments E1 to E5, E1r, and the closed-form validation (30 seeded runs per point) |
| `sim/experiments3.py`, `sim/experiments4.py` | Control experiments added in the revision (pooling, aged and admission-based reactive caches, temporal locality, reward robustness, wrong-reuse accounting, trajectory contacts, interaction sensitivity) |
| `sim/experiments5.py` | V5 controls: E5b (factorial with EWMA as the low level of the time factor) and E2f (reward grid of step 0.05) |
| `sim/realdata.py` | V5 checks on public data: MovieTweetings request-trace replay and STS Benchmark similarity-versus-meaning curve (download commands in the file header; data are not redistributed) |
| `sim/experiments6.py` | V6: E6r, the semantic x coordination interaction versus r = B*Cb/K (20 seeds per point) |
| `sim/figs6.py` | Builds Fig. 6 of the V6 article from `res6_sts.json` and `res6_e6r_summary.json` |
| `review/cited_works.csv` | Every cited work with year, venue, preprint flag, the article sections that cite it, and its code for the synthesis of Section VIII (main axis, axes T/P/L/M/R, evidence A/S/D/X/N); surveys, data sets, and general techniques are marked as not coded |
| `sim/figs2.py` | Builds the figures of Section IX from the stored results |
| `sim/res2_*.json` to `sim/res5_*.json` | Raw per-run results for every point, with means and 95% confidence intervals |
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
| E1p, E1c | RQ1 controls | Pooled vs per-cell estimates; shared vs per-cell reactive state; aged LFU and TinyLFU-style caches; temporal locality (`loc`) |
| E1t, E1m | RQ1 controls | Requests per item per epoch (`T`); memory knee at F = 800 |
| E2r | RQ2 controls | Cost-minimizing reward under Beta cost laws, reward scale, and skew, with bootstrap intervals |
| E3t | RQ3 control | Trajectory-based contact rule (`contact='traj'`) |
| E4s | RQ4 control | Exact / same-class / wrong-class hits; semantic index slots (`idx_slots`) |
| E5b, E2f | V5 controls | EWMA low level in the factorial; fine reward grid for the cost-minimizing reward |
| trace, sts | Public data | One-cache replay of 921,398 timestamped requests; similarity threshold against human ratings on 1,379 sentence pairs |
| E6r, sts2 | V6 | Interaction versus storage per semantic class r; similarity versus meaning for three encoders on STS-B and a Quora duplicate-question sample |
| E5s, E5sb, E5t | RQ5 controls | Semantic x coordination contrast under changed K, Cb, class-correlated and cell-specific popularity; factorial at tau = 0.8 and 0.9 |

## Reproduce

```
pip install -r requirements.txt
./run_all.sh            # figures from the stored results
./run_all.sh --rerun    # also reruns every simulation (about 7,400 runs, roughly 0.5 s each per core)
```

Seeds 0 to 29 are reused across the compared policies (common random numbers), so a rerun reproduces the stored values exactly. The reactive policies (`est='lru'`, `'lfu'`, `'lfua'`, `'tlfu'`) draw no extra random numbers, and every option added in the revision defaults to the earlier behavior, so the stored `res2_*` results are reproduced exactly.

## Run your own simulations and change parameters

```
cd sim
python3 run_sim.py --list                                   # every parameter and its default
python3 run_sim.py                                          # the paper's default setting, 30 seeds
python3 run_sim.py v=10 p_part=0.5 coop=1                   # faster devices, half participation, coordinated BSs
python3 run_sim.py est=lru drift=0.2 --seeds 10             # reactive LRU caches under faster drift
python3 run_sim.py tau=0.85 --vary Cb=5,10,20,40            # semantic reuse, sweep BS cache size
python3 run_sim.py tau=0.85 --vary K=20,40,80,160 --out my.json
```

From Python: `from ucm import run; run(seed=0, v=10, coop=True)` returns one run as a vector named by `ucm.NAMES` (`hit`, `own`, `d2d`, `bs`, `ibs`, `miss`, `fid`, `lat`, `sp_cost`, `pay`, ...). To rerun a paper experiment with other settings, edit its grid in `experiments2.py` to `experiments6.py` (each experiment is a short function listing the parameter points) and run it as in `run_all.sh`. The simulator is Python with NumPy only; there is no MATLAB version.

Parameters most worth changing: `F` catalog size, `gamma` Zipf skew, `drift` popularity change per epoch, `est` (`oracle`, `stale`, `window`, `ewma`, `mem`, `lru`, `lfu`, `lfua`, `tlfu`), `M` predictor memory, `Cb`/`Cd` BS and device cache sizes, `N` devices, `v` speed (m/s), `r` D2D range (m), `tf` transfer time (s), `coop` BS coordination, `p_part` user participation, `kappa` reward scale, `aware` mobility-aware placement, `tau` similarity threshold (1 = no semantic reuse), `K` semantic classes, `s` within-class spread.

## Main parameters (`DEF` in `sim/ucm.py`)

`F=400` items, Zipf exponent `gamma=0.8`, `K=80` semantic classes, `N=80` devices, 4 BSs in a 600 m square, BS and device caches of 20 and 10 items, D2D range 100 m, transfer time 5 s, speed 2 m/s, drift 0.05 per epoch, 20 epochs of 20 slots (3 warm-up epochs), tier delays 1, 5, 20, 40, 100 ms, backhaul and inter-BS costs 1 and 0.3, reward scale 4. The activity `beta` (probability that a device issues a request in a slot) is 1 in all reported results.

## License

Code: MIT (see `LICENSE`). Results and figures: CC BY 4.0.

## Data and encoders for the V6 checks (`realdata.py sts2`)

Data are not redistributed. Besides the files listed in the header of `sim/realdata.py`:

- `data/qqp_dev_sample.tsv`: a 1,999-pair sample of the GLUE QQP development set (columns id, qid1, qid2, question1, question2, is_duplicate), from a public copy in Microsoft's NeuronBlocks repository (md5 4a25960ed86e6188213be0b44e2088dd).
- `data/quora_first1000.csv`: the first 1,000 rows of the Quora question-pairs data with columns question1_en, question2_en, is_duplicate, from a public English-French copy. After removing duplicate pairs the two files give 2,995 pairs.
- Encoders: `sentence-transformers/all-MiniLM-L6-v2` and `sentence-transformers/all-mpnet-base-v2`. Hugging Face was not reachable from our build machine, so we loaded the weights from GitHub mirrors (henrytanner52/all-MiniLM-L6-v2 at f92edf0, homer6/all-mpnet-base-v2 at bd44305) and checked them by reproducing the published STS-B Spearman correlations (0.82 and 0.83). Set `MINILM` and `MPNET` to the local model folders. Loading the models from Hugging Face gives the same weights.
