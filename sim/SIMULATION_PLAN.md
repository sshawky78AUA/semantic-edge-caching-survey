# Simulation Plan: UCM Evaluation for Section IX (RQ1-RQ5)

Status (Nov 8, 2026): executed in Python (`experiments2.py`, `figs2.py`, results in `res2_*.json`); results are in Section IX of the paper. MATLAB port (Section 5) remains to do.

Original status: planning document. Targets a MATLAB implementation
that reproduces the equations of Section II (UCM) one to one, matching the
validated Python prototype in this folder, so the numbers that go into the
paper are reproducible and checked against closed forms before anything else
is trusted.

## 1. Objective and mapping to the paper

| Experiment | RQ | Axis | Claim under test |
|---|---|---|---|
| E1 | RQ1 | Time | Value of prediction grows with drift rate; memory-bounded estimators trade hit ratio for footprint |
| E2 | RQ2 | People | SP-side and user-side cooperation substitute for each other; Stackelberg reward has an interior optimum |
| E3 | RQ3 | Place | D2D value collapses above the cutoff speed; mobility-aware placement helps below it |
| E4 | RQ4 | Meaning | Semantic matching raises effective capacity at a fidelity cost; gain saturates at the semantic-class size |
| E5 | RQ5 | Interaction | Which axis dominates hit ratio and SP cost, and which pairs reinforce or substitute |

Each experiment turns on one UCM layer, sweeps the parameters that axis's
literature varies (Section IX already enumerates these), and holds the rest
at the base case below. This is unchanged from the current Section IX text;
what follows grounds the base case in realistic values and specifies how the
numbers get produced.

## 2. Base-case parameters, with grounding

The Table IV defaults are already partly grounded (Zipf exponent against
real web-proxy traces, within-class similarity validated by sampling). The
table below keeps every value that is already justified in Section II and
tightens the few that were picked for convenience rather than realism.

| Symbol | Current default | Plan | Grounding |
|---|---|---|---|
| $F$ | 400 | keep (400) | Representative of a single-operator regional catalog tier (not a full CDN catalog); large enough for a Zipf tail, small enough to run E5's $2^5$ factorial at scale |
| $\gamma$ | 0.8 | keep, add sweep $\{0.6, 0.8, 1.0\}$ as a robustness check in E1 | 0.6-1.0 spans the exponents reported for web and video request traces since Breslau et al.~\cite{breslau1999zipf} |
| $L$ (region side), $B$ (BSs) | 600 m, 4 | keep | 2x2 grid of micro-cells at $\approx$300 m inter-site distance, consistent with 3GPP urban micro-cell deployments (TR 36.872-class spacing) |
| $N$, $C_b$, $C_d$ | 80, 20, 10 | keep | Device-to-BS ratio (20/cell) and cache sizes (5%/2.5% of catalog) are in the range used in FemtoCaching-style numerics |
| $r$, $t_f$ | 100 m, 5 s | keep | 100 m is a representative D2D/WiFi-Direct range; 5 s transfer time matches a few-second content object at D2D rates |
| $v$ | 2 m/s | add explicit pedestrian/vehicular sweep $\{0, 1.4, 2, 5, 10, 15.7^-, 20, 30\}$ m/s in E3 | 1.4 m/s is the standard pedestrian walking speed; 15.7 m/s is the model's own cutoff $\pi r/(4t_f)$; 30 m/s ($\approx$108 km/h) covers vehicular D2D |
| $d_0,\dots,d_4$ | 1, 5, 20, 40, 100 ms | keep order of magnitude, but state explicitly as representative RTTs (on-device cache access, D2D link, radio access to serving BS, inter-BS/X2-class link, core/backhaul) rather than measured figures | Consistent with typical ranges reported for LTE/5G air-interface and backhaul latency budgets; flag in the paper as representative, not measured |
| $\delta$ | 0.05 | keep, with sweep $[0, 0.3]$ already specified in E1 | 5% of a 400-item catalog reordering per epoch (5 s $\times$ 20 slots = 100 s) is a moderate drift; the sweep covers slow to fast churn |
| $\tau$ | 1 | sweep $[0.7, 1]$ in E4, unchanged | Already grounded via the coverage-count table in Section II |
| $c_{bh}, c_{ib}, \kappa$ | 1, 0.3, 4 | keep | Backhaul costlier than inter-BS by construction (3.3x), consistent with inter-BS links usually being shorter, cheaper leased or fiber links than the core backhaul |

No parameter above needs to move to be "realistic" in an absolute sense
(these are stylized values, as in all of this literature), but each one
should carry a one-line justification like the above when it appears in the
paper, and several should become explicit sweeps rather than fixed points, so
the robustness of each RQ's answer can be shown, not just asserted.

## 3. Experimental design (refined E1-E5)

Keep the design in the current Section IX table, with these additions:

- **E1 (time):** add the $\gamma \in \{0.6, 0.8, 1.0\}$ robustness sweep at
  fixed drift, 3 extra curves, same estimators.
- **E2 (people):** add a zero-coordination, zero-D2D baseline (SP-only,
  $\rho=0$) as the reference point for all latency/cost comparisons, so the
  "substitution" claim has an explicit anchor.
- **E3 (place):** add $v=0$ (static) and $v \to \infty$-equivalent (no usable
  contact) as the two theoretical bounds the finite-speed curve must sit
  between.
- **E4 (meaning):** add $C_b \in \{10, 20, 40\}$ crossed with $\tau$, as
  already in the table, plus one run at $\tau=1$ (exact match) per $C_b$ as
  the no-semantics baseline.
- **E5 (interaction):** keep the $2^5$ full factorial (32 cells); add a
  second block with different random seeds (blocking) so block effects can
  be separated from the five main factors.

## 4. Statistical protocol

- 30 independent runs per configuration (up from 20), common random numbers
  (same seed sequence) reused across compared policies within an experiment
  to reduce variance when comparing, e.g., coordinated vs. independent BSs at
  the same $\rho$.
- Report mean $\pm$ 95% CI ($\pm 1.96 \times$ SE) for every curve, as now.
- For E5, fit the $2^5$ factorial with a linear model (Yates' algorithm or
  equivalent regression) to get main effects and two-way interactions with
  their own standard errors, not just visual comparison of cells.
- Pre-register (in the simulator's config file) which three validation
  checks must pass before any experiment's output is trusted:
  1. Oracle, $\delta=0$, $\rho=0$, independent BSs: measured hit ratio
     within $\pm 3$ SE of the closed form $h_{\mathrm{top}}(C)$ (Eq. in
     Section II).
  2. Measured usable-neighbor counts within $\pm 3$ SE of the
     border-corrected analytical value at $v \in \{0, 2, 10\}$ m/s.
  3. D2D share $h_1 \to 0$ as $v \to \pi r/(4 t_f)$ from below.

## 5. Implementation plan (MATLAB)

Port the existing, already-validated Python prototype (`sim/ucm.py`,
`sim/experiments.py`) function-for-function, since its structure already
matches the paper's equations one to one and its outputs already pass the
three validation checks above (measured hit ratio $0.3874\pm0.0005$ vs.
$0.3881$ closed form, etc.). Proposed module layout:

```
sim_matlab/
  ucm_defaults.m      % DEF struct: base-case parameters (Table IV)
  ucm_semantic.m      % semantic() -> similarity matrix S, coverage matrix Cm
  ucm_greedy_cover.m  % greedy_cover() -> submodular placement
  ucm_place_bs.m      % place_bs()  -> independent / coordinated BS placement
  ucm_place_dev.m     % place_dev() -> proportional / mobility-aware device placement
  ucm_run.m           % run()       -> one simulation run, all 13 output metrics
  ucm_avg.m           % avg()       -> R replications, mean + 95% CI (parfor)
  run_e1_time.m .. run_e5_factorial.m   % one script per experiment, writing a .mat/.csv
  validate_ucm.m      % the three closed-form checks in Section 4, run first
  make_figures.m      % regenerates every Section IX/X figure from the .mat/.csv outputs
```

- Use `parfor` in `ucm_avg.m` over replications (direct analogue of the
  Python `multiprocessing.Pool`).
- Use `rng(seed, 'twister')` with substreams (`RandStream` per worker) for
  reproducibility under parallel execution.
- Each `run_eX_*.m` script writes its results table to `res_eX.csv` (or
  `.mat`), mirroring the existing `res_e1.json` ... `res_e5.json`, so the
  LaTeX tables/figures can be regenerated from a flat file without rerunning
  the simulation.
- `validate_ucm.m` must pass before any `run_eX_*.m` script is treated as
  producing citable numbers.

## 6. Deliverables back into the paper

Once the above runs and passes validation:

1. Five result figures/tables for Section IX (one per RQ, as the current
   table of experiments already promises).
2. An updated Section IX with the actual numbers and confidence intervals,
   replacing the current methodology-only text.
3. A correction to Section I's contribution (3) and "Organization" paragraph,
   and to the abstract, so that "we answer RQ1-RQ5 by simulation" is true at
   the point those sentences are published, not aspirational.
4. The MATLAB source and the base-case config released alongside the paper,
   consistent with the "code, parameters, and seeds are released" claim
   already in the text.

## 7. Open decision for the user

The Python prototype already produces the qualitative results quoted earlier
in this project (SP coordination 0.365 to 0.580, semantic cache beating a 4x
larger exact cache, factorial main effects). Two options:

- **(a) Port to MATLAB** per Section 5 above, rerun everything there, and use
  those numbers in the paper (matches the stated MATLAB preference).
- **(b) Use the existing Python results directly**, document that the
  simulator is Python, and skip the port.

(a) costs more time now but keeps the released-code claim consistent with
preference; (b) is faster but means either changing the released-code
language in the paper or carrying a MATLAB preference that the artifact
doesn't reflect.
