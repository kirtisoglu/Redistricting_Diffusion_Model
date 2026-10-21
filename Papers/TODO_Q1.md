# TODO: the Forest ReCom kill test

Written 2026-10-21, revised the same day after the first gate run. The
previous version of this file (the September audit tiers and the October
status log) is in git history at commit dbd76a3; its open items are
summarised at the bottom.

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

## What was built

All in `diffusion_model/recom/`:

- `forest_recom.py`: the forest-lifted tree move, Chain E lifted to the
  forest state space, their mixture, a runner with the same output format
  as `hybrid.run_hybrid`, two targets (`Target`, the paper's pi_w, and
  `TreeWeightedTarget`, prod tau(V_d) times pi_w), and an analytic
  detailed-balance check on the lifted 2x3 fiber. The Hastings ratio is
  derived in the module docstring. Two choices to state in the paper: the
  adjacent pair is drawn uniformly from boundary pairs (as in every other
  move of the paper, not by a uniform cut edge as GerryChain does; both
  are exact), and a tree with no balanced edge is redrawn up to 20 times
  (exact because the feasibility probability depends on H only and
  cancels; not true of the local move's rounding, which must stay
  single-shot).
- `kill_stats.py`: ESS, TV with noise floor, batch-means Hotelling with
  low-mass cut bins merged, per-chain summary.
- `forest_validate_5x5.py`: correctness gate against the 4,006 enumerated
  plans, three kernel targets, either target family.
- `forest_iowa.py`: the kill test on Iowa, short and long modes, either
  target family.

## What is already known (cloud session, 2026-10-21)

Verified before handoff:

- Analytic check: detailed-balance violation at or below 1.5e-15 for the
  forest move, the lifted local move and the mixture, under uniform and
  density kernels, under both targets; closed-form proposal mass equals
  brute-force enumeration to 1.8e-16.
- 5x5 gate at 20,000 steps passes under both targets (every forest and
  f-hybrid row within 1.5 floor SDs, Hotelling p 0.20 to 0.70).
- The local agent's full 200,000-step gate under pi passed (p 0.21 to
  0.96) and reproduced the paper's tree-move rows exactly.

Measured facts that shape the test:

1. **The forest move is a different baseline.** Acceptance 0.85 to 0.98
   on the grid and 0.11 to 0.18 on Iowa under pi, against 0.06 to 0.08
   and 0.011 to 0.013 for the paper's tree move; ESS per second 3 to 10
   times the paper's move on the grid. On Iowa under pi it reaches the
   target's equilibrium (uniform kernel, mean cut about 88) within a
   thousand steps; the paper's move needed about 1e5.

2. **What rejects the forest move under pi is the tree-count ratio, not
   the single draw.** Decomposition of log alpha over feasible proposals
   on Iowa, 3,000 steps from the start plan:

   | target | P(tree has balanced edge) | accept given feasible | mean log tau-ratio | mean log pi-ratio | mean log S-ratio |
   |---|---|---|---|---|---|
   | uniform | 0.41 | 0.25 | -5.9 | +0.6 | +0.5 |
   | perimeter | 0.53 | 0.22 | -6.0 | +0.7 | +0.6 |
   | density | 0.41 | 0.13 | -9.8 | +1.6 | +0.7 |

   At equilibrium under pi the acceptance given a feasible tree falls
   further, to about 0.17 (uniform). A spanning-tree proposal favours
   splits with many spanning trees; targeting plain pi_w makes the ratio
   undo that bias every step. Under the tree-weighted target the term
   cancels and the forest move accepts 0.65 to 0.75 on Iowa with ESS
   150 to 180 per 3,000 steps.

3. **Retry is exact and neutral.** Redrawing infeasible trees lifts
   feasibility from 0.44 to 0.95 at 2.5 times the wall clock; accepted
   moves per second are unchanged (158 vs 75 per second over 2.5 vs 6.3
   seconds, same count). It is on by default because it is what practice
   does and it makes per-step acceptance comparable to GerryChain's; it
   does not move the kill ratio.

4. **The Iowa calibration, by a second route.** Under the tree-weighted
   uniform target at lam = 0.1 the forest move equilibrates at mean cut
   41.4; the paper's tempered ReCom sits at 39.8 and pi itself near 88.
   That is Section 5.4's claim with an exact sampler on each side.

5. **Smoke-run kill ratios** (3,000 to 20,000 steps, too short to quote):
   on the grid 0.55 to 1.04 under both targets; on Iowa under the tree
   target 0.6 to 1.0; on Iowa under pi all over the place, ESS in the
   tens. The long runs decide.

Why two targets. Under pi the local move is on home ground (it does not
pay the tree-count ratio); under the tree-weighted target the forest
move is, and the local move pays the ratio instead. Running both answers
"where does the local move help" rather than "does it", and the second
is the hardest fair test the baseline can be given.

The cloud smoke outputs were deleted; the artifacts the paper will cite
must come from the runs below. The local agent's Step 2 output from
commit a3dbc8f (no retry, file `forest_iowa_short_tau0.1_seed42.npz`)
is a valid no-retry reference and should be kept, but the paper's tables
use the runs below.

## What to run, in order

Pull first; Steps 1 to 3 need the revised code. Run everything from
`diffusion_model/`. Each command writes one `.npz` into
`recom/iowa_hybrid_test_out/` and prints a table; keep the printed tables
(tee to a log file) because the Results section below is filled from
them.

### Step 0. Environment and kernel check (5 seconds)

    python3 recom/forest_recom.py

Expect every "max rel DB violation" at or below 1e-14 under both targets
and the closed-form vs enumeration error below 1e-12. If not, stop: the
environment differs from the one the code was tested in (GerryChain
0.2.22, where `Graph.nodes()` is a method; 1.0 breaks `problems.py`).

### Step 1. Correctness gate on enumerated ground truth (about 6 minutes each)

    python3 recom/forest_validate_5x5.py 200000 42 0.2 0.1 pi   | tee recom/iowa_hybrid_test_out/forest_validate_5x5_pi.log
    python3 recom/forest_validate_5x5.py 200000 42 0.2 0.1 tree | tee recom/iowa_hybrid_test_out/forest_validate_5x5_tree.log

Both must end in `GATE PASSED`. If either reads `GATE FAILED`, stop and
report which row failed; do not run Iowa.

### Step 2. Iowa short runs, direction and timing (about 30 minutes each)

    python3 recom/forest_iowa.py short 200000 42 0.1 pi   | tee recom/iowa_hybrid_test_out/forest_iowa_short_tau0.1_seed42_pi.log
    python3 recom/forest_iowa.py short 200000 42 0.1 tree | tee recom/iowa_hybrid_test_out/forest_iowa_short_tau0.1_seed42_tree.log

The ratios in these tables are rough under pi (ESS in the tens for the
forest move) and are not the verdict. Under the tree target the ESS is
large enough that the short run is already informative.

### Step 3. Iowa long runs, the verdict (run in parallel)

Under pi, 2,000,000 steps, about 70 minutes per command:

    python3 recom/forest_iowa.py long forest 2000000 42 pi | tee recom/iowa_hybrid_test_out/forest_iowa_long_forest_seed42_pi.log
    python3 recom/forest_iowa.py long hybrid 2000000 42 pi | tee recom/iowa_hybrid_test_out/forest_iowa_long_hybrid_seed42_pi.log
    python3 recom/forest_iowa.py long forest 2000000 43 pi | tee recom/iowa_hybrid_test_out/forest_iowa_long_forest_seed43_pi.log
    python3 recom/forest_iowa.py long hybrid 2000000 43 pi | tee recom/iowa_hybrid_test_out/forest_iowa_long_hybrid_seed43_pi.log

Under the tree-weighted target, 500,000 steps is enough (ESS about 50 per
thousand steps), about 15 minutes per command:

    python3 recom/forest_iowa.py long forest 500000 42 tree | tee recom/iowa_hybrid_test_out/forest_iowa_long_forest_seed42_tree.log
    python3 recom/forest_iowa.py long hybrid 500000 42 tree | tee recom/iowa_hybrid_test_out/forest_iowa_long_hybrid_seed42_tree.log
    python3 recom/forest_iowa.py long forest 500000 43 tree | tee recom/iowa_hybrid_test_out/forest_iowa_long_forest_seed43_tree.log
    python3 recom/forest_iowa.py long hybrid 500000 43 tree | tee recom/iowa_hybrid_test_out/forest_iowa_long_hybrid_seed43_tree.log

Each command runs the perimeter and the density target. The hybrid uses
p_local = 0.1 and tau = 0.1, the paper's locked settings.

### Step 4, optional. The grid mobility optimum on Iowa (about 30 minutes)

    python3 recom/forest_iowa.py short 200000 42 0.3 pi | tee recom/iowa_hybrid_test_out/forest_iowa_short_tau0.3_seed42_pi.log

Only if Step 3 under pi is close to the threshold and the short runs
suggest p = 0.3 or tau = 0.3 is the better setting.

### Step 5. Commit and push

    git add diffusion_model/recom/iowa_hybrid_test_out/forest_*.npz diffusion_model/recom/iowa_hybrid_test_out/forest_*.log Papers/TODO_Q1.md
    git commit -m "Forest ReCom kill test: results"
    git push

Before committing, paste the printed tables from Steps 1 to 3 into the
Results section below, verbatim.

## Pass / fail, fixed in advance

The kill ratio is ESS per second of the weighted-cut trace, f-hybrid
divided by forest, from the Step 3 long runs. The verdict is taken under
pi, the paper's stated target. The tree-weighted result is reported
alongside as the hardest test and shapes the write-up but does not
decide it.

PASS: under pi, ratio >= 1.5 on both the perimeter and the density
target on both seeds, with the quarter means of the two chains on the
same level (a ratio between chains at different levels is not a ratio of
anything).

FAIL: under pi, ratio < 1.5 on either target on either seed, or quarter
means that disagree between the chains.

In-between (passes on one target, or on one seed): report it as such; the
honest write-up is "decisive under X, marginal under Y", and Section 5.3
is written that way.

## What happens next, by outcome

Either way the paper gains two results that did not exist before: the
right baseline (forest move 3 to 10 times the paper's tree move) and the
mechanism of fact 2 above, which says exactly when a tree proposal pays
and a local one does not.

PASS. The paper keeps its sampler claim with the right baseline. Edits:
Section 4.7 describes the forest move in place of the partition-level
tree move (keep the latter as one paragraph, since its closed form is
why Chain A is only approximate), with the retry remark and the
tree-count-ratio decomposition; Section 5.3 and Table "hybrid-iowa" are
rebuilt from the `forest_*_pi.npz` files, with the tree-weighted rows as
the control; contribution 2 and the abstract quote the new ratios. Then
the precinct-scale run (restore `archive/diffusion_model/mew` for the NC
graph) is the last experiment before submission to Statistics and
Computing or JCGS.

FAIL. The local move does not earn its place next to an exact global
move that practice runs. Sections 4.7 and 5.3 shrink to a short negative
result with the forest numbers and the decomposition, contribution 2
drops the hybrid claim, and the paper is the theory of the proposal
(4.3, 4.6), the exact chain (4.5, 5.2) and the calibration (5.1, now
with fact 4 as its Iowa half). That version goes to a conference
(AISTATS or UAI for the MCMC content, SIAM ACDA for the algorithms crowd)
or a Q2 journal. The informed-MEW idea, as a Boltzmann draw over marked
edges with computable density, becomes the next experiment, inside this
paper's framework, not a second paper.

## Results (fill in)

### Step 0

(paste)

### Step 1: 5x5 gate, pi and tree

(paste both tables and GATE lines)

### Step 2: Iowa short, pi and tree

(paste)

### Step 3: Iowa long

(paste the summary lines; then fill the table)

| target family | kernel | seed | ESS/s forest | ESS/s f-hybrid | ratio | quarter means agree? |
|---|---|---|---|---|---|---|
| pi | perimeter | 42 | | | | |
| pi | density | 42 | | | | |
| pi | perimeter | 43 | | | | |
| pi | density | 43 | | | | |
| tree | perimeter | 42 | | | | |
| tree | density | 42 | | | | |
| tree | perimeter | 43 | | | | |
| tree | density | 43 | | | | |

Verdict under pi:

Reading of the tree-weighted rows:

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
