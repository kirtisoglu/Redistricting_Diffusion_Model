# TODO: the Forest ReCom kill test

Written 2026-10-21. The previous version of this file (the September audit
tiers and the October status log) is in git history at commit dbd76a3; its
open items are summarised at the bottom.

## Decision

One paper, `paper.tex`. The informed-MEW idea is not written up separately;
it stays as a fallback section if the test below fails.

The paper's sampler claim (contribution 2, Sections 4.7 and 5.3) rests on
the hybrid beating the partition-level tree move of `hybrid.py`. That move
draws its cut edge among all tree edges and accepts under 2 percent on
Iowa, which is not what practice runs. The claim has to be re-established
against the forest-lifted move with balanced-edge selection (Autry,
Herschlag, Mattingly 2023), which is exact and is the baseline a referee
will name. Until that number exists, nothing else in the paper is worth
editing.

## What was built (2026-10-21, cloud session)

All in `diffusion_model/recom/`:

- `forest_recom.py`: the forest-lifted tree move, Chain E lifted to the
  forest state space, their mixture, a runner with the same output format
  as `hybrid.run_hybrid`, and an analytic detailed-balance check on the
  lifted 2x3 fiber. The derivation of the Hastings ratio is in the module
  docstring. One design choice to note in the paper: the adjacent pair is
  drawn uniformly from boundary pairs (as in every other move of the
  paper), not by a uniform cut edge as GerryChain does; both are exact,
  the ratio carries the `|bp|` factor.
- `kill_stats.py`: ESS, TV with noise floor, batch-means Hotelling,
  per-chain summary, shared by the runners.
- `forest_validate_5x5.py`: correctness gate against the 4,006 enumerated
  plans, three targets, three chains (paper's tree move, forest move,
  forest hybrid).
- `forest_iowa.py`: the kill test on Iowa, short and long modes.

Verified in the cloud session (Gurobi restricted licence, GerryChain
0.2.22, geopandas 0.14):

- Analytic check: max relative detailed-balance violation 2.4e-16 (forest),
  1.4e-15 (lifted local), 2.6e-16 (mixture), under a uniform and a
  density kernel; closed-form proposal mass equals brute-force enumeration
  to 1.8e-16.
- 5x5, 20,000 steps, seed 42, p_local 0.2, tau 0.1: gate passed on all
  three targets (every forest and f-hybrid row within 1.3 floor SDs,
  Hotelling p 0.16 to 0.50). Forest move acceptance 0.59 to 0.67 against
  0.06 to 0.08 for the paper's tree move; forest / tree-part ESS per
  second 3.0 to 6.7. Kill ratio f-hybrid / forest 0.44 to 0.72
  (passenger, as the paper's Section 4.6 predicts under exact balance).
- Iowa, 3,000 steps (timing only): forest move acceptance 0.13 to 0.21
  against 0.011 to 0.013 for the paper's tree move; about 1 ms per step
  for the forest move and 1.2 ms for the hybrid, so a 2e6-step run is
  about 40 minutes.

The cloud smoke outputs were deleted; the artifacts the paper will cite
must come from the runs below.

## What to run, in order

Run everything from `diffusion_model/`. Each command writes one `.npz`
into `recom/iowa_hybrid_test_out/` and prints a table; keep the printed
tables (redirect to a log file) because the Results section below is
filled from them.

### Step 0. Environment and kernel check (2 seconds)

    python3 recom/forest_recom.py

Expect every "max rel DB violation" at or below 1e-14 and the closed-form
vs enumeration error below 1e-12. If not, stop: the environment differs
from the one the code was tested in (check GerryChain is 0.2.22 and
`Graph.nodes()` still works as a method).

### Step 1. Correctness gate on enumerated ground truth (about 6 minutes)

    python3 recom/forest_validate_5x5.py 200000 42 0.2 0.1 | tee recom/iowa_hybrid_test_out/forest_validate_5x5.log

The last line must read `GATE PASSED`. If it reads `GATE FAILED`, stop
and report which row failed; do not run Iowa.

The 20,000-step cloud numbers quoted above should be reproduced closely
by the first tenth of this run but need not match bit for bit across
machines.

### Step 2. Iowa short runs, direction and timing (about 20 minutes)

    python3 recom/forest_iowa.py short 200000 42 0.1 | tee recom/iowa_hybrid_test_out/forest_iowa_short_tau0.1_seed42.log

Twelve chains. The ratios in this table are rough (ESS in the tens) and
are not the verdict; they tell you whether the long runs are worth
starting and whether the hybrid at p = 0.1 or p = 0.3 looks better.

### Step 3. Iowa long runs, the verdict (about 40 minutes per command; run the four in parallel)

    python3 recom/forest_iowa.py long forest 2000000 42 | tee recom/iowa_hybrid_test_out/forest_iowa_long_forest_seed42.log
    python3 recom/forest_iowa.py long hybrid 2000000 42 | tee recom/iowa_hybrid_test_out/forest_iowa_long_hybrid_seed42.log
    python3 recom/forest_iowa.py long forest 2000000 43 | tee recom/iowa_hybrid_test_out/forest_iowa_long_forest_seed43.log
    python3 recom/forest_iowa.py long hybrid 2000000 43 | tee recom/iowa_hybrid_test_out/forest_iowa_long_hybrid_seed43.log

Each command runs the perimeter and the density target. The hybrid uses
p_local = 0.1 and tau = 0.1, the paper's locked settings.

### Step 4, optional. The grid mobility optimum on Iowa (about 20 minutes)

    python3 recom/forest_iowa.py short 200000 42 0.3 | tee recom/iowa_hybrid_test_out/forest_iowa_short_tau0.3_seed42.log

Only if Step 3 is close to the threshold and the short runs at tau = 0.1
suggest p = 0.3 is the better mixture.

### Step 5. Commit and push

    git add diffusion_model/recom/iowa_hybrid_test_out/forest_*.npz diffusion_model/recom/iowa_hybrid_test_out/forest_*.log Papers/TODO_Q1.md
    git commit -m "Forest ReCom kill test: results"
    git push

Before committing, paste the printed tables from Steps 1 to 3 into the
Results section below, verbatim.

## Pass / fail, fixed in advance

The kill ratio is ESS per second of the weighted-cut trace, f-hybrid
divided by forest, from the Step 3 long runs.

PASS: ratio >= 1.5 under both the perimeter and the density target on
both seeds, with the quarter means of the two chains on the same level
(the equilibrium check; a ratio from chains at different levels is not
a ratio of anything).

FAIL: ratio < 1.5 on either target on either seed, or quarter means that
disagree between the chains.

In-between (passes on one target, or on one seed): report it as such; the
honest write-up is "decisive under X, marginal under Y", and Section 5.3
is written that way.

## What happens next, by outcome

PASS. The paper keeps its sampler claim with the right baseline. Edits:
Section 4.7 describes the forest move in place of the partition-level
tree move (keep the latter as one paragraph, since its closed form is
why Chain A is only approximate); Section 5.3 and Table "hybrid-iowa"
are rebuilt from the `forest_*.npz` files; contribution 2 and the
abstract quote the new ratios; the forest / tree-part ratio from Step 1
is reported as the correction to the earlier baseline. Then the
precinct-scale run (restore `archive/diffusion_model/mew` for the NC
graph) is the last experiment before submission to Statistics and
Computing or JCGS.

FAIL. The local move does not earn its place next to an exact global
move that practice runs. Sections 4.7 and 5.3 shrink to a short negative
result with the forest numbers, contribution 2 drops the hybrid claim,
and the paper is the theory of the proposal (4.3, 4.6), the exact chain
(4.5, 5.2) and the calibration (5.1). That version goes to a conference
(AISTATS or UAI for the MCMC content, SIAM ACDA for the algorithms crowd)
or a Q2 journal. The informed-MEW idea, as a Boltzmann draw over marked
edges with computable density, becomes the next experiment, inside this
paper's framework, not a second paper.

## Results (fill in)

### Step 0

(paste)

### Step 1: 5x5 gate

(paste the table and the GATE line)

### Step 2: Iowa short

(paste)

### Step 3: Iowa long

(paste the four summary lines; then compute the four ratios here)

| target | seed | ESS/s forest | ESS/s f-hybrid | ratio | quarter means agree? |
|---|---|---|---|---|---|
| perimeter | 42 | | | | |
| density | 42 | | | | |
| perimeter | 43 | | | | |
| density | 43 | | | | |

Verdict:

## Items still open from the previous version of this file

Carried over so they are not lost; none of them should be touched before
the kill test is decided.

- Chain A's acceptance rate is not reported anywhere in the paper.
- `long_run_5x5_seed42.npz` has no alpha, beta or step-count stamp.
- Per-state visit arrays for the 5x5 short runs are not stored.
- `cannon2020reversible` is in `ref.bib` and uncited while the text says
  "RevReCom theory identifies".
- 41 uncited entries in `ref.bib`, several malformed.
- 11 tables and 3 figures; two or three tables should become figures.
- No long Vanilla ReCom run against the pure tree-weighted measure.
- Release tag and DOI on acceptance.
