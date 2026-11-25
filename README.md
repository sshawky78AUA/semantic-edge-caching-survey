# Unified Caching Model (UCM) simulator

Simulation code, results, and coding table for the survey

> S. Hosny and J. Tadrous, "From Classical to Semantic Caching: A Survey of Content Caching at the Wireless Edge," submitted to *IEEE Communications Surveys & Tutorials*, 2026.

The UCM places the four lines of edge caching research in one model. Each line is a layer that can be switched off:

- **time:** popularity prediction, drift, memory-bounded predictors, and reactive LRU/LFU caches
- **people:** SP coordination of base-station (BS) caches, user participation, and rewards
- **place:** random-direction mobility and D2D contacts that must last long enough for a transfer
- **meaning:** semantic coverage of a request by similar items, and the fidelity of a semantic hit

Section IX of the article defines the model and reports the experiments. The supplement gives the derivations, the experiment design, and the control experiments. Everything in both documents can be regenerated from this repository.

## Requirements

Python 3.10 or later with NumPy and Matplotlib:

```
pip install -r requirements.txt
```

The simulator itself uses only NumPy. The checks on public data (`sim/realdata.py`) need additional packages and downloaded data; see the end of this file.

## Run your own simulations

`sim/run_ucm.py` runs the model with any parameters you choose. Run it from the `sim/` folder.

```
cd sim
python3 run_ucm.py                          # default network (Table XI of the article), 20 seeds
python3 run_ucm.py --list                   # every parameter, its default, and its meaning
python3 run_ucm.py coop=True tau=0.85       # coordinated BSs with semantic matching
python3 run_ucm.py --sweep v=0,2,5,10,20,30 aware=True              # effect of device speed
python3 run_ucm.py --sweep tau=1,0.95,0.9,0.85,0.8 --sweep coop=False,True --plot tau.pdf
python3 run_ucm.py --sweep est=oracle,stale,ewma,lru,lfu,tlfu drift=0.2 --metrics hit,bs,miss
python3 run_ucm.py --sweep p_part=0,0.2,0.4,0.6,0.8,1 --metrics sp_cost,hit --csv reward.csv
```

- `name=value` sets a parameter; every other parameter keeps its default.
- `--sweep name=v1,v2,...` varies one parameter, and a second `--sweep` gives a grid.
- `--runs` sets the number of seeded runs per point (default 20); results are printed as mean ± 95% confidence interval.
- Every point uses the same seeds (common random numbers), so differences between points are not caused by different random demand or movement.
- `--metrics` chooses the printed metrics (`all` prints every metric).
- `--plot` draws the first metric against the first swept parameter.
- `--csv` saves the means and intervals; `--out` saves every run as JSON.

The same model is available from Python:

```python
from ucm import run, NAMES, DEF
out = run(seed=0, coop=True, tau=0.85, v=10.0)   # one run, any parameters of DEF
print(dict(zip(NAMES, out)))
```

### Parameters (`DEF` in `sim/ucm.py`)

| Parameter | Default | Meaning |
|---|---|---|
| `F` | 400 | Items in the catalog |
| `K` | 80 | Semantic classes |
| `gamma` | 0.8 | Zipf exponent of popularity |
| `N` | 80 | Devices |
| `area` | 600 | Side of the square region (m); 4 BSs at fixed positions |
| `r` | 100 | D2D range (m) |
| `v` | 2 | Device speed (m/s), random-direction mobility |
| `tf` | 5 | Time to complete a D2D transfer (s) |
| `dt` | 5 | Slot length (s) |
| `Cb`, `Cd` | 20, 10 | BS and device cache sizes (items) |
| `drift` | 0.05 | Popularity drift per epoch |
| `est` | `ewma` | Popularity estimator: `oracle`, `stale` (last epoch), `window`, `ewma`, `mem` (memory-bounded); reactive BS caches `lru`, `lfu`, `lfua` (aged LFU), `tlfu` (TinyLFU-style admission) |
| `alpha`, `W`, `M` | 0.5, 3, None | EWMA weight, window length (epochs), counters of the memory-bounded estimator |
| `tau` | 1.0 | Similarity threshold of a semantic hit (1 = exact match only) |
| `s` | 0.4 | Spread of item embeddings around their class center |
| `p_part` | 1.0 | Reward ρ: devices whose private cost is at most ρ cache for others (1 = all join) |
| `aware` | True | Mobility-aware device placement |
| `coop` | False | Coordinated BS caches |
| `E`, `T`, `warm` | 20, 20, 3 | Epochs, slots per epoch, warm-up epochs |
| `lat` | (1, 5, 20, 40, 100) | Delays (ms) of own cache, D2D, local BS, other BS, backhaul |
| `c_bh`, `c_ib`, `kappa` | 1.0, 0.3, 4.0 | SP cost of a backhaul and an inter-BS fetch, reward scale |
| `beta` | 1.0 | Probability that a device issues a request in a slot |
| `info` | `pooled` | `pooled`, `shared`, or `local` estimates at the BSs |
| `cost_dist` | `uniform` | Private participation costs: `uniform`, `beta25`, `beta52` |
| `idx_slots` | 0 | BS cache slots given up to the semantic index |
| `loc`, `loc_len` | 0.0, 50 | Temporal locality: probability of re-referencing one of the last `loc_len` requests |
| `lam_cell` | 0.0 | Weight of a cell-specific popularity |
| `class_corr` | False | Popularity ranks assigned to whole semantic classes |
| `contact` | `mean` | D2D contact rule: `mean` (article) or `traj` (along simulated trajectories) |

### Metrics (`NAMES` in `sim/ucm.py`)

`hit` hit ratio; `own`, `d2d`, `bs`, `ibs`, `miss` share of requests served by the device's own cache, D2D, the local BS, another BS, and the backhaul; `lat` mean latency (ms), also `lat_part` and `lat_non` for participating and other devices; `fid` mean fidelity of hits; `sp_cost` SP cost per request and `pay` the reward part of it; `participants` number of participating devices; `exact`, `sem_ok`, `sem_wrong` share of requests answered by the exact item, by an item of the same class, and by an item of another class.

## Reproduce the article

```
./run_all.sh            # figures and table values from the stored results (seconds)
./run_all.sh --quick    # runs every experiment with 3 seeds per point into quick/ (smoke test)
./run_all.sh --rerun    # reruns every simulation with the seeds of the article
```

Seeds are fixed (seeds 0 to 29, or 0 to 19 for the control experiments), and the same seeds are used for every compared policy, so `--rerun` reproduces the stored results exactly. `UCM_WORKERS` sets the number of processes (default: all cores). The full rerun is about 23,000 runs of roughly 0.5 s each (about 3 CPU-hours, so under half an hour on 8 cores).

`sim/tables.py` prints the numbers of the tables from the stored results, for example `python3 tables.py rho` for Table S-II.

| Script and command | Experiment | Output | In the article |
|---|---|---|---|
| `experiments2.py validate` | Closed-form check: oracle, no drift, no D2D, C = 10, 20, 40 | `res2_validate.json` | Sec. IX, Sec. S-II |
| `experiments2.py e1` | E1: drift and skew for five estimators; predictor memory | `res2_e1.json` | Fig. S1 |
| `experiments2.py e1r` | E1r: proactive estimators vs. reactive LRU/LFU | `res2_e1r.json` | Fig. 5 |
| `experiments2.py e2` | E2: reward with independent or coordinated BSs, 2 and 10 m/s | `res2_e2.json` | Fig. 6, Table S-II |
| `experiments2.py e3` | E3: speed 0 to 30 m/s for three placement rules | `res2_e3.json` | Fig. S2 |
| `experiments2.py e4` | E4: similarity threshold for three BS cache sizes and coordinated BSs | `res2_e4.json` | Fig. 7 |
| `experiments2.py e5` | E5: 2^5 factorial design over the five factors | `res2_e5.json` | Fig. 8, Sec. S-VI |
| `experiments3.py e1p`, `experiments4.py e1c` | Pooled vs. per-cell estimates; aged LFU, TinyLFU-style; temporal locality | `res3_e1p.json`, `res4_e1c.json` | Fig. 5, Sec. S-III |
| `experiments4.py e1t`, `e1m` | Requests per item per epoch; memory knee at F = 800 | `res4_e1t.json`, `res4_e1m.json` | Sec. S-III |
| `experiments3.py e2r 0..2` | Cost-minimizing reward under other cost laws, κ, and γ | `res3_e2r*.json` | Table S-II |
| `experiments5.py e2f` | Reward grid of step 0.05, 30 seeds | `res5_e2f.json` | Sec. IX (`tables.py rho_fine`) |
| `experiments4.py e3t` | Trajectory-based contact rule | `res4_e3t.json` | Fig. S2, Sec. S-V |
| `experiments3.py e4s` | Exact / same-class / wrong-class hits; semantic index slots | `res3_e4s.json` | Fig. 7, Table S-I |
| `experiments4.py e5s`, `e5sb`, `e5t 0.8`, `e5t 0.9` | Interaction under changed K, Cb, class-correlated and cell-specific popularity; factorial at τ = 0.8, 0.9 | `res4_e5*.json` | Sec. S-VI |
| `experiments5.py e5b` | Factorial with EWMA as the low level of prediction | `res5_e5b.json` | Sec. S-VI |
| `experiments6.py e6r` | Semantic × coordination interaction versus r = B·Cb/K | `res6_e6r.json`, `res6_e6r_summary.json` | Table S-III |
| `realdata.py trace` | Replay of 921,398 MovieTweetings requests | `res5_trace.json` | Table XII |
| `realdata.py sts`, `sts2` | Similarity vs. meaning on STS-B and Quora pairs | `res5_sts.json`, `res6_sts.json` | Tables S-IV, S-V |
| `figs2.py e1 e1r e2 e3 e4 e5` | Figures from the stored results | `figs/*.pdf` | Figs. 5 to 8, S1, S2 |

The design and parameter choices are explained in `sim/SIMULATION_PLAN.md`.

## Repository contents

| Path | What it is |
|---|---|
| `sim/ucm.py` | The simulator: `run(seed, **params)` simulates one seeded run and returns the metrics |
| `sim/run_ucm.py` | Command-line interface for your own runs, sweeps, CSV output, and plots |
| `sim/experiments2.py` to `sim/experiments6.py` | The experiments of the article and the supplement |
| `sim/realdata.py` | Checks on public data (request trace and sentence-pair benchmarks) |
| `sim/figs2.py`, `sim/tables.py` | Figures and table values from the stored results |
| `sim/res*.json` | Raw per-run results of every point, with means and 95% confidence intervals |
| `figs/` | The simulation figures of the article and the supplement (vector PDF) |
| `review/cited_works.csv` | Every cited work with year, venue, preprint flag, citing sections, and its code for the synthesis of Section VIII (main axis, axes T/P/L/M/R, evidence A/S/D/X/N) |

## Data for the checks on public data

Data are not redistributed. Download them into `sim/data/` before running `realdata.py`:

```
mkdir -p sim/data
curl -sL -o sim/data/ratings.dat https://raw.githubusercontent.com/sidooms/MovieTweetings/master/latest/ratings.dat
curl -sL -o sim/data/stsb-en-test.csv https://raw.githubusercontent.com/PhilipMay/stsb-multi-mt/main/data/stsb-en-test.csv
pip install spacy sentence-transformers
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl
```

- `sim/data/qqp_dev_sample.tsv`: a 1,999-pair sample of the GLUE QQP development set (columns id, qid1, qid2, question1, question2, is_duplicate), from a public copy in Microsoft's NeuronBlocks repository (md5 4a25960ed86e6188213be0b44e2088dd).
- `sim/data/quora_first1000.csv`: the first 1,000 rows of the Quora question-pairs data with columns question1_en, question2_en, is_duplicate. After removing duplicate pairs the two files give 2,995 pairs.
- Encoders: `sentence-transformers/all-MiniLM-L6-v2` and `sentence-transformers/all-mpnet-base-v2`. They are loaded from Hugging Face, or from local folders given by the environment variables `MINILM` and `MPNET`. We checked the weights by reproducing the published STS-B Spearman correlations (0.82 and 0.83).

## Citation and license

Please cite the article above (see `CITATION.cff`). Code: MIT (see `LICENSE`). Results and figures: CC BY 4.0.
