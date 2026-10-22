# TODO: the Forest ReCom kill test

Written 2026-10-21, revised the same day after the first gate run. The
previous version of this file (the September audit tiers and the October
status log) is in git history at commit dbd76a3; its open items are
summarised at the bottom.


**Status 2026-10-22: the kill test PASSED under the pre-registered criterion** (pi, p_local = 0.1, ratios 1.65 to 2.29 on both kernels and both seeds). Details in Results, Step 3.
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

Run 2026-10-21 on commit e4743a6 (`forest_step0_kernel_check.log`). Passes: largest violation 1.45e-15 (limit 1e-14), closed form vs enumeration 1.8e-16 (limit 1e-12). Note that p_H = 1 on this fiber, so the retry rule is not exercised here; the 5x5 gate below exercises it.

```
lifted 2x3 fiber, target = pi, uniform kernel, lam = 0.10
  forest   max rel DB violation 3.90e-16   max |row sum - 1| 0.0e+00   min diagonal 0.0333
  local    max rel DB violation 1.41e-15   max |row sum - 1| 3.3e-16   min diagonal 0.7525
  mixture  max rel DB violation 4.34e-16   max |row sum - 1| 4.4e-16   min diagonal 0.4496
  (18 labeled plans, 30 lifted states, tau(H) = 15, p_H = 1.000, retry factor c_H = 1.0000, closed-form proposal mass vs enumeration: max rel err 1.8e-16)
lifted 2x3 fiber, target = pi, density kernel (rho = x, sigma = 1), lam = 1.0
  forest   max rel DB violation 2.87e-16   max |row sum - 1| 2.2e-16   min diagonal 0.0333
  local    max rel DB violation 1.09e-15   max |row sum - 1| 2.2e-16   min diagonal 0.7392
  mixture  max rel DB violation 3.56e-16   max |row sum - 1| 4.4e-16   min diagonal 0.4572
  (18 labeled plans, 30 lifted states, tau(H) = 15, p_H = 1.000, retry factor c_H = 1.0000, closed-form proposal mass vs enumeration: max rel err 1.8e-16)
lifted 2x3 fiber, target = tree, uniform kernel, lam = 0.10
  forest   max rel DB violation 3.71e-16   max |row sum - 1| 2.2e-16   min diagonal 0.0556
  local    max rel DB violation 1.45e-15   max |row sum - 1| 2.2e-16   min diagonal 0.6841
  mixture  max rel DB violation 5.23e-16   max |row sum - 1| 3.3e-16   min diagonal 0.4032
  (18 labeled plans, 30 lifted states, tau(H) = 15, p_H = 1.000, retry factor c_H = 1.0000, closed-form proposal mass vs enumeration: max rel err 1.8e-16)
lifted 2x3 fiber, target = tree, density kernel (rho = x, sigma = 1), lam = 1.0
  forest   max rel DB violation 4.09e-16   max |row sum - 1| 1.1e-16   min diagonal 0.0333
  local    max rel DB violation 1.11e-15   max |row sum - 1| 4.4e-16   min diagonal 0.7392
  mixture  max rel DB violation 5.97e-16   max |row sum - 1| 2.2e-16   min diagonal 0.3862
  (18 labeled plans, 30 lifted states, tau(H) = 15, p_H = 1.000, retry factor c_H = 1.0000, closed-form proposal mass vs enumeration: max rel err 1.8e-16)
```

### Step 1: 5x5 gate, pi and tree

Both passed. Run 2026-10-21 on commit e4743a6.

Under pi (`forest_validate_5x5_pi.log`):

```
5x5 fiber, 4006 states, 200000 steps per chain, p_local = 0.2, tau = 0.1, target = pi

          config     chain  TV marg   floor       z  Hotel p  TV full   floor  acc glob  acc loc  mv/loc     ESS  wall s   ESS/s
 uniform lam=0.1 tree-part   0.0097  0.0386    -1.8    0.582   0.6191  0.9076     0.066      nan     nan     384     7.5    51.1
 uniform lam=0.1    forest   0.0055  0.0103    -1.2    0.460   0.1653  0.3500     0.851      nan     nan    5262    24.8   211.8
 uniform lam=0.1  f-hybrid   0.0127  0.0119    +0.2    0.434   0.1923  0.3927     0.851    0.660   0.013    3812    26.5   143.8
                 forest / tree-part ESS/s = 4.15   kill ratio f-hybrid / forest = 0.68

   uniform lam=1 tree-part   0.0284  0.0431    -0.8    0.213   0.4499  0.7781     0.074      nan     nan     330     7.3    45.2
   uniform lam=1    forest   0.0035  0.0081    -1.3    0.800   0.1151  0.2333     0.979      nan     nan    8298    24.5   338.6
   uniform lam=1  f-hybrid   0.0047  0.0098    -1.3    0.827   0.1309  0.2745     0.979    0.673   0.015    6303    26.2   240.3
                 forest / tree-part ESS/s = 7.49   kill ratio f-hybrid / forest = 0.71

   density lam=1 tree-part   0.0283  0.0365    -0.6    0.603   0.5810  0.8693     0.069      nan     nan     418     7.6    55.3
   density lam=1    forest   0.0037  0.0093    -1.5    0.960   0.1621  0.2958     0.876      nan     nan    6751    26.5   254.5
   density lam=1  f-hybrid   0.0050  0.0112    -1.5    0.248   0.1795  0.3610     0.876    0.784   0.005    4606    27.3   168.9
                 forest / tree-part ESS/s = 4.61   kill ratio f-hybrid / forest = 0.66

Saved forest_validate_5x5_p0.2_tau0.1_pi.npz
GATE PASSED: forest and f-hybrid sample their enumerated targets.
```

Under the tree-weighted target (`forest_validate_5x5_tree.log`):

```
5x5 fiber, 4006 states, 200000 steps per chain, p_local = 0.2, tau = 0.1, target = tree

          config     chain  TV marg   floor       z  Hotel p  TV full   floor  acc glob  acc loc  mv/loc     ESS  wall s   ESS/s
 uniform lam=0.1    forest   0.0093  0.0085    +0.2    0.370   0.1022  0.2038     0.944      nan     nan    6617    25.7   257.3
 uniform lam=0.1  f-hybrid   0.0056  0.0090    -0.8    0.778   0.1139  0.2315     0.942    0.675   0.011    5136    27.2   188.8
                 kill ratio f-hybrid / forest = 0.73

   uniform lam=1    forest   0.0026  0.0050    -1.0    0.639   0.0745  0.1282     0.884      nan     nan    6743    25.4   265.9
   uniform lam=1  f-hybrid   0.0047  0.0060    -0.4    0.411   0.0886  0.1474     0.886    0.683   0.011    4933    27.1   182.3
                 kill ratio f-hybrid / forest = 0.69

   density lam=1    forest   0.0072  0.0070    +0.1    0.456   0.0949  0.1774     0.906      nan     nan    6629    25.6   259.3
   density lam=1  f-hybrid   0.0034  0.0081    -1.3    0.699   0.1122  0.2021     0.906    0.789   0.004    5066    30.1   168.6
                 kill ratio f-hybrid / forest = 0.65

Saved forest_validate_5x5_p0.2_tau0.1_tree.npz
GATE PASSED: forest and f-hybrid sample their enumerated targets.
```

The same gate under pi on the earlier commit a3dbc8f (no retry) also passed; its outputs are kept as `forest_validate_5x5_a3dbc8f.log` and `forest_validate_5x5_p0.2_tau0.1_a3dbc8f.npz`.

### Step 2: Iowa short, pi and tree

Run 2026-10-21 on commit e4743a6.

Under pi (`forest_iowa_short_tau0.1_seed42_pi.log`):

```
Iowa, k=4, eps=0.05, 200000 steps per chain, tau = 0.1, seed 42, target = pi

    kernel     lam          chain  acc glob  acc loc  mv/loc  fast  wcut h1  wcut h2     ESS   wall s   ESS/s
   uniform   0.100         forest    0.1626      nan     nan   nan    89.23    87.84      36    224.9   0.161   ratio vs forest 1.00
   uniform   0.100 f-hybrid p=0.1    0.1575    0.249   0.141  0.53    90.07    89.43      56    249.8   0.223   ratio vs forest 1.39
   uniform   0.100 f-hybrid p=0.3    0.1647    0.254   0.140  0.57    89.75    89.27     180    252.3   0.714   ratio vs forest 4.44
   uniform   0.100      tree-part    0.0119      nan     nan   nan    83.17    87.24      14     35.1   0.395   ratio vs forest 2.46

 perimeter   0.343         forest    0.1629      nan     nan   nan    28.09    28.23      43    231.5   0.185   ratio vs forest 1.00
 perimeter   0.343 f-hybrid p=0.1    0.1634    0.530   0.130  0.79    28.70    28.23      79    233.9   0.339   ratio vs forest 1.83
 perimeter   0.343 f-hybrid p=0.3    0.1577    0.526   0.137  0.78    28.00    28.35     116    256.8   0.450   ratio vs forest 2.44
 perimeter   0.343      tree-part    0.0056      nan     nan   nan    25.53    28.16       6     38.5   0.153   ratio vs forest 0.83

   density   0.718         forest    0.1738      nan     nan   nan    18.27    17.88      56    228.7   0.243   ratio vs forest 1.00
   density   0.718 f-hybrid p=0.1    0.1734    0.672   0.134  0.98    18.14    18.09      93    219.1   0.424   ratio vs forest 1.74
   density   0.718 f-hybrid p=0.3    0.1741    0.669   0.133  0.98    18.22    18.08     174    204.5   0.852   ratio vs forest 3.51
   density   0.718      tree-part    0.0068      nan     nan   nan    13.26    14.83       6     30.7   0.204   ratio vs forest 0.84

Saved forest_iowa_short_tau0.1_seed42_pi.npz
```

Under the tree-weighted target (`forest_iowa_short_tau0.1_seed42_tree.log`):

```
Iowa, k=4, eps=0.05, 200000 steps per chain, tau = 0.1, seed 42, target = tree

    kernel     lam          chain  acc glob  acc loc  mv/loc  fast  wcut h1  wcut h2     ESS   wall s   ESS/s
   uniform   0.100         forest    0.7407      nan     nan   nan    41.05    40.93   10035    165.7  60.563   ratio vs forest 1.00
   uniform   0.100 f-hybrid p=0.1    0.7420    0.419   0.157  0.56    40.98    40.94   10677    189.4  56.365   ratio vs forest 0.93
   uniform   0.100 f-hybrid p=0.3    0.7442    0.422   0.152  0.56    41.05    40.90    8010    234.2  34.203   ratio vs forest 0.56

 perimeter   0.343         forest    0.7332      nan     nan   nan    13.04    13.04   10257    164.6  62.307   ratio vs forest 1.00
 perimeter   0.343 f-hybrid p=0.1    0.7319    0.714   0.100  0.81    13.04    13.03    9459    174.6  54.179   ratio vs forest 0.87
 perimeter   0.343 f-hybrid p=0.3    0.7345    0.715   0.103  0.81    13.06    13.04    7686    191.3  40.173   ratio vs forest 0.64

   density   0.718         forest    0.6616      nan     nan   nan     7.30     7.31    9295    173.0  53.717   ratio vs forest 1.00
   density   0.718 f-hybrid p=0.1    0.6619    0.831   0.077  0.99     7.34     7.27    8334    181.3  45.978   ratio vs forest 0.86
   density   0.718 f-hybrid p=0.3    0.6598    0.832   0.077  0.99     7.28     7.29    5927    170.7  34.730   ratio vs forest 0.65

Saved forest_iowa_short_tau0.1_seed42_tree.npz
```

No-retry reference under pi, commit a3dbc8f (`forest_iowa_short_tau0.1_seed42.log`):

```
Iowa, k=4, eps=0.05, 200000 steps per chain, tau = 0.1, seed 42

    kernel     lam          chain  acc glob  acc loc  mv/loc  fast  wcut h1  wcut h2     ESS   wall s   ESS/s
   uniform   0.100         forest    0.0768      nan     nan   nan    93.01    88.70      23     91.6   0.246   ratio vs forest 1.00
   uniform   0.100 f-hybrid p=0.1    0.0768    0.226   0.136  0.52    88.44    91.18      40    121.1   0.327   ratio vs forest 1.33
   uniform   0.100 f-hybrid p=0.3    0.1066    0.262   0.140  0.56    88.73    89.70      69    165.4   0.418   ratio vs forest 1.70
   uniform   0.100      tree-part    0.0119      nan     nan   nan    83.17    87.24      14     37.5   0.370   ratio vs forest 1.50

 perimeter   0.343         forest    0.0950      nan     nan   nan    29.21    28.12      14     88.5   0.161   ratio vs forest 1.00
 perimeter   0.343 f-hybrid p=0.1    0.0935    0.530   0.136  0.78    27.48    28.58      22    106.3   0.211   ratio vs forest 1.31
 perimeter   0.343 f-hybrid p=0.3    0.0933    0.523   0.125  0.79    29.00    28.58      56    133.4   0.422   ratio vs forest 2.62
 perimeter   0.343      tree-part    0.0056      nan     nan   nan    25.53    28.16       6     35.9   0.164   ratio vs forest 1.02

   density   0.718         forest    0.0931      nan     nan   nan    18.64    17.91      32     86.8   0.366   ratio vs forest 1.00
   density   0.718 f-hybrid p=0.1    0.1082    0.675   0.128  0.98    18.01    18.40      57     97.1   0.588   ratio vs forest 1.61
   density   0.718 f-hybrid p=0.3    0.0986    0.685   0.130  0.98    18.34    18.30     116    111.5   1.037   ratio vs forest 2.84
   density   0.718      tree-part    0.0068      nan     nan   nan    13.26    14.83       6     32.2   0.194   ratio vs forest 0.53

Saved forest_iowa_short_tau0.1_seed42.npz
```

The cloud session's recomputation of the no-retry reference from its artifact, with quarter means, and its reading of that run (written before the revised-code runs above existed):

| kernel | chain | acc global | acc local | quarter means of weighted cut | ESS | wall s | ESS/s | ratio vs forest |
|---|---|---|---|---|---|---|---|---|
| uniform | forest | 0.077 | | 94.4 91.7 91.3 86.1 | 23 | 92 | 0.246 | 1.00 |
| uniform | f-hybrid p=0.1 | 0.077 | 0.226 | 89.3 87.6 93.5 88.9 | 40 | 121 | 0.327 | 1.33 |
| uniform | f-hybrid p=0.3 | 0.107 | 0.262 | 89.6 87.9 90.0 89.4 | 69 | 165 | 0.418 | 1.70 |
| uniform | tree-part | 0.012 | | 82.0 84.4 87.0 87.5 | 14 | 38 | 0.370 | 1.50 |
| perimeter | forest | 0.095 | | 28.0 30.5 26.8 29.4 | 14 | 89 | 0.161 | 1.00 |
| perimeter | f-hybrid p=0.1 | 0.094 | 0.530 | 26.8 28.2 28.4 28.8 | 22 | 106 | 0.211 | 1.31 |
| perimeter | f-hybrid p=0.3 | 0.093 | 0.523 | 29.4 28.6 28.7 28.5 | 56 | 133 | 0.422 | 2.62 |
| perimeter | tree-part | 0.006 | | 25.7 25.4 27.2 29.1 | 6 | 36 | 0.164 | 1.02 |
| density | forest | 0.093 | | 19.4 17.9 18.4 17.5 | 32 | 87 | 0.366 | 1.00 |
| density | f-hybrid p=0.1 | 0.108 | 0.675 | 18.5 17.5 18.6 18.2 | 57 | 97 | 0.588 | 1.61 |
| density | f-hybrid p=0.3 | 0.099 | 0.685 | 18.5 18.2 18.4 18.2 | 116 | 112 | 1.037 | 2.84 |
| density | tree-part | 0.007 | | 13.1 13.4 14.9 14.8 | 6 | 32 | 0.194 | 0.53 |

Reading. Direction only: the forest move's ESS is 14 to 32, so its ESS/s
is rough and probably overstated (short runs understate autocorrelation,
as the paper found for the old tree move). Three things are worth
noting. The hybrid at p = 0.3 beats p = 0.1 on every kernel, so the long
runs should include p = 0.3. The hybrid's quarter means are stable
while the forest move's wander (uniform 94 to 86, perimeter 27 to 30),
the same convergence contrast the paper reported against the old tree
move. And on Iowa the forest move is not faster per second than the
paper's tree move (0.16 to 0.37 against 0.16 to 0.37): its steps cost
2.5 times more, since every feasible draw pays the determinants and the
crossing-edge sums, so its order-of-magnitude acceptance advantage is
spent on cost. On the grid, where steps are cheap, it is 4 to 7 times
faster. This matters for the write-up: the forest move is the right
baseline, but it does not by itself demolish the paper's Iowa ratios.

### Step 3: Iowa long

Run 2026-10-21 to 22. The eight runs of the run sheet started together on
commit e4743a6. The two p = 0.3 hybrid runs below were added by commit
af1421f, whose only change to `forest_iowa.py` is the optional p_local
argument, and started about six minutes later. All ten ran at `nice -n 19`
with one thread each on an 11-core machine with 5 performance cores.

The cloud session added, after the short runs, the hybrid at p = 0.3 under
pi on both seeds:

    python3 recom/forest_iowa.py long hybrid 2000000 42 pi 0.3 | tee recom/iowa_hybrid_test_out/forest_iowa_long_hybrid_p0.3_seed42_pi.log
    python3 recom/forest_iowa.py long hybrid 2000000 43 pi 0.3 | tee recom/iowa_hybrid_test_out/forest_iowa_long_hybrid_p0.3_seed43_pi.log

It also wrote that "the criterion is unchanged: the hybrid passes if either
p clears 1.5". That is a change. The criterion fixed in advance named
p = 0.1 only, and a second chance added after seeing the short runs makes
passing easier. Both verdicts are reported below, the pre-registered one
first.

Summary lines, verbatim from the logs:

```
 perimeter lam=0.343 forest 2000000 steps seed 42 target pi: acc forest 0.1575 acc loc nan quarter means 28.18 27.84 28.03 28.39  ESS 421  wall 3438s  ESS/s 0.1226
   density lam=0.718 forest 2000000 steps seed 42 target pi: acc forest 0.1753 acc loc nan quarter means 17.95 18.03 18.31 17.91  ESS 486  wall 2749s  ESS/s 0.1769
 perimeter lam=0.343 forest 2000000 steps seed 43 target pi: acc forest 0.1625 acc loc nan quarter means 28.57 28.23 28.67 27.89  ESS 247  wall 3438s  ESS/s 0.0718
   density lam=0.718 forest 2000000 steps seed 43 target pi: acc forest 0.1750 acc loc nan quarter means 17.88 17.80 17.70 18.18  ESS 585  wall 2740s  ESS/s 0.2135
 perimeter lam=0.343 hybrid p_local=0.3 2000000 steps seed 42 target pi: acc forest 0.1620 acc loc 0.531 quarter means 28.19 28.09 28.08 28.06  ESS 1116  wall 3439s  ESS/s 0.3244
   density lam=0.718 hybrid p_local=0.3 2000000 steps seed 42 target pi: acc forest 0.1744 acc loc 0.675 quarter means 18.08 18.23 18.12 18.10  ESS 1395  wall 2498s  ESS/s 0.5587
 perimeter lam=0.343 hybrid p_local=0.3 2000000 steps seed 43 target pi: acc forest 0.1564 acc loc 0.527 quarter means 28.18 28.27 28.21 28.13  ESS 1101  wall 3457s  ESS/s 0.3183
   density lam=0.718 hybrid p_local=0.3 2000000 steps seed 43 target pi: acc forest 0.1732 acc loc 0.672 quarter means 18.03 18.13 18.02 18.19  ESS 1244  wall 2504s  ESS/s 0.4967
 perimeter lam=0.343 hybrid 2000000 steps seed 42 target pi: acc forest 0.1623 acc loc 0.531 quarter means 28.16 27.99 28.25 28.14  ESS 696  wall 3437s  ESS/s 0.2024
   density lam=0.718 hybrid 2000000 steps seed 42 target pi: acc forest 0.1778 acc loc 0.677 quarter means 17.93 17.99 18.23 17.93  ESS 925  wall 2685s  ESS/s 0.3446
 perimeter lam=0.343 hybrid 2000000 steps seed 43 target pi: acc forest 0.1618 acc loc 0.529 quarter means 28.25 28.56 28.50 28.21  ESS 572  wall 3472s  ESS/s 0.1649
   density lam=0.718 hybrid 2000000 steps seed 43 target pi: acc forest 0.1766 acc loc 0.673 quarter means 18.01 18.04 18.00 18.10  ESS 1167  wall 2692s  ESS/s 0.4336
 perimeter lam=0.343 forest 500000 steps seed 42 target tree: acc forest 0.7340 acc loc nan quarter means 13.05 13.05 13.05 13.02  ESS 27019  wall 615s  ESS/s 43.9195
   density lam=0.718 forest 500000 steps seed 42 target tree: acc forest 0.6601 acc loc nan quarter means 7.30 7.30 7.29 7.32  ESS 21642  wall 671s  ESS/s 32.2667
 perimeter lam=0.343 forest 500000 steps seed 43 target tree: acc forest 0.7325 acc loc nan quarter means 13.01 13.02 13.05 13.07  ESS 27113  wall 614s  ESS/s 44.1422
   density lam=0.718 forest 500000 steps seed 43 target tree: acc forest 0.6596 acc loc nan quarter means 7.30 7.30 7.27 7.30  ESS 22750  wall 672s  ESS/s 33.8653
 perimeter lam=0.343 hybrid 500000 steps seed 42 target tree: acc forest 0.7320 acc loc 0.716 quarter means 13.03 13.04 13.02 13.05  ESS 24558  wall 657s  ESS/s 37.3998
   density lam=0.718 hybrid 500000 steps seed 42 target tree: acc forest 0.6598 acc loc 0.835 quarter means 7.31 7.28 7.27 7.29  ESS 19615  wall 673s  ESS/s 29.1425
 perimeter lam=0.343 hybrid 500000 steps seed 43 target tree: acc forest 0.7349 acc loc 0.719 quarter means 13.03 13.04 13.02 13.07  ESS 23604  wall 659s  ESS/s 35.8328
   density lam=0.718 hybrid 500000 steps seed 43 target tree: acc forest 0.6595 acc loc 0.835 quarter means 7.27 7.29 7.28 7.29  ESS 19647  wall 672s  ESS/s 29.2304
```

Verdict computation from the stored artifacts (`recom/forest_verdict.py`,
output in `forest_verdict.txt`). Beyond the run sheet it adds two checks.
"z" tests whether the two chains sit at the same level: the difference of
post-burn means in standard errors, each from that chain's own ESS, with
|z| < 2 as "same level". "corr ratio" guards against the shared cores: ESS
per step ratio times the Step 2 short run's per-step cost ratio, where both
chains ran in one process.

```
   family    kernel seed |  ESS f  ESS h  ESS/s f  ESS/s h  ratio |  mean f  mean h      z | cost h/f long  short corr ratio
       pi perimeter   42 |    421    696   0.1226   0.2024   1.65 |   28.11   28.14  +0.20 |          1.00   1.01       1.63
                          quarters forest 28.18 27.84 28.03 28.39   hybrid 28.16 27.99 28.25 28.14
       pi perimeter   43 |    247    572   0.0718   0.1649   2.29 |   28.34   28.38  +0.23 |          1.01   1.01       2.29
                          quarters forest 28.57 28.23 28.67 27.89   hybrid 28.25 28.56 28.50 28.21
       pi   density   42 |    486    925   0.1769   0.3446   1.95 |   18.05   18.02  -0.25 |          0.98   0.96       1.99
                          quarters forest 17.95 18.03 18.31 17.91   hybrid 17.93 17.99 18.23 17.93
       pi   density   43 |    585   1167   0.2135   0.4336   2.03 |   17.89   18.04  +1.46 |          0.98   0.96       2.08
                          quarters forest 17.88 17.80 17.70 18.18   hybrid 18.01 18.04 18.00 18.10
     tree perimeter   42 |  27019  24558  43.9195  37.3998   0.85 |   13.04   13.03  -0.64 |          1.07   1.06       0.86
                          quarters forest 13.05 13.05 13.05 13.02   hybrid 13.03 13.04 13.02 13.05
     tree perimeter   43 |  27113  23604  44.1422  35.8328   0.81 |   13.04   13.04  +0.32 |          1.07   1.06       0.82
                          quarters forest 13.01 13.02 13.05 13.07   hybrid 13.03 13.04 13.02 13.07
     tree   density   42 |  21642  19615  32.2667  29.1425   0.90 |    7.30    7.29  -1.12 |          1.00   1.05       0.87
                          quarters forest 7.30 7.30 7.29 7.32   hybrid 7.31 7.28 7.27 7.29
     tree   density   43 |  22750  19647  33.8653  29.2304   0.86 |    7.29    7.28  -0.98 |          1.00   1.05       0.82
                          quarters forest 7.30 7.30 7.27 7.30   hybrid 7.27 7.29 7.28 7.29
  pi p0.3 perimeter   42 |    421   1116   0.1226   0.3244   2.65 |   28.11   28.11  -0.02 |          1.00   1.11       2.39
                          quarters forest 28.18 27.84 28.03 28.39   hybrid 28.19 28.09 28.08 28.06
  pi p0.3 perimeter   43 |    247   1101   0.0718   0.3183   4.43 |   28.34   28.20  -0.87 |          1.01   1.11       4.02
                          quarters forest 28.57 28.23 28.67 27.89   hybrid 28.18 28.27 28.21 28.13
  pi p0.3   density   42 |    486   1395   0.1769   0.5587   3.16 |   18.05   18.13  +0.81 |          0.91   0.89       3.21
                          quarters forest 17.95 18.03 18.31 17.91   hybrid 18.08 18.23 18.12 18.10
  pi p0.3   density   43 |    585   1244   0.2135   0.4967   2.33 |   17.89   18.09  +2.04 |          0.91   0.89       2.38
                          quarters forest 17.88 17.80 17.70 18.18   hybrid 18.03 18.13 18.02 18.19

Pre-registered criterion (pi, p_local = 0.1): 4 of 4 cells meet ratio >= 1.5 with |z| < 2 -> PASS; cost-corrected: 4 of 4.

p_local = 0.3 (added after the short runs): 3 of 4 cells meet ratio >= 1.5 with |z| < 2 -> IN-BETWEEN; cost-corrected: 3 of 4.
Amended criterion (either p clears it): PASS. Report the pre-registered verdict first; the amendment was made after seeing short runs.
```

| target family | kernel | seed | ESS/s forest | ESS/s f-hybrid | ratio | quarter means agree? |
|---|---|---|---|---|---|---|
| pi | perimeter | 42 | 0.1226 | 0.2024 | 1.65 | yes, z = +0.20 |
| pi | density | 42 | 0.1769 | 0.3446 | 1.95 | yes, z = -0.25 |
| pi | perimeter | 43 | 0.0718 | 0.1649 | 2.29 | yes, z = +0.23 |
| pi | density | 43 | 0.2135 | 0.4336 | 2.03 | yes, z = +1.46 |
| tree | perimeter | 42 | 43.92 | 37.4 | 0.85 | yes, z = -0.64; quarters 13.05 13.05 13.05 13.02 vs 13.03 13.04 13.02 13.05 |
| tree | density | 42 | 32.27 | 29.14 | 0.90 | yes, z = -1.12; quarters 7.30 7.30 7.29 7.32 vs 7.31 7.28 7.27 7.29 |
| tree | perimeter | 43 | 44.14 | 35.83 | 0.81 | yes, z = +0.32; quarters 13.01 13.02 13.05 13.07 vs 13.03 13.04 13.02 13.07 |
| tree | density | 43 | 33.87 | 29.23 | 0.86 | yes, z = -0.98; quarters 7.30 7.30 7.27 7.30 vs 7.27 7.29 7.28 7.29 |
| pi, p=0.3 | perimeter | 42 | 0.1226 | 0.3244 | 2.65 | yes, z = -0.02 |
| pi, p=0.3 | density | 42 | 0.1769 | 0.5587 | 3.16 | yes, z = +0.81 |
| pi, p=0.3 | perimeter | 43 | 0.0718 | 0.3183 | 4.43 | yes, z = -0.87 |
| pi, p=0.3 | density | 43 | 0.2135 | 0.4967 | 2.33 | no, just (forest quarters wander 17.70 to 18.18), z = +2.04 |

Verdict under pi: **PASS under the pre-registered criterion** (p_local =
0.1). The ratios are 1.65, 2.29, 1.95 and 2.03, all four at or above 1.5,
and every pair sits on the same level (largest |z| is 1.46). The
cost-corrected ratios are 1.63, 2.29, 1.99 and 2.08, also all four, and the
long runs' per-step cost ratios (0.98 to 1.01) match the short runs' (0.96
to 1.01), so sharing cores did not move the result. At p = 0.3 three of
four cells pass. The fourth, density seed 43, has ratio 2.33 but z = 2.04,
just over the bar, because the forest chain's quarter means wander from
17.70 to 18.18. That is the forest move's slow mixing, not a different
equilibrium. Under the amended either-p criterion the verdict is also PASS,
carried by p = 0.1.

The margin is real but not wide. The smallest pre-registered cell is 1.65
(perimeter, seed 42), and the seeds differ (1.65 against 2.29 under
perimeter), so the paper should quote the range across seeds rather than a
single number. The forest move's ESS is 247 to 585 over 1.8e6 retained
steps, an autocorrelation time of 3,000 to 7,000 steps.

Reading of the tree-weighted rows: under the tree-weighted target, where
the forest move's tree-count ratio cancels, the hybrid is a passenger on
every cell. The ratios are 0.81 to 0.90 (cost-corrected 0.82 to 0.87),
both chains sit on the same level (largest |z| is 1.12), and the ESS is
above 19,000. Even per step, before any cost is counted, the hybrid
decorrelates slightly less: 0.052 to 0.055 effective samples per step
against 0.060 for the forest move on perimeter, and 0.044 against 0.048
to 0.051 on density, so a local step contributes less than the forest
step it replaces (acceptance: forest 0.66 to 0.73, local 0.72 to 0.84;
this point from the cloud session's reading, commit 81d40c9). Together
with the pi rows this answers where the local move helps. It helps when the global move pays a tree-count ratio against the
target, as under the paper's pi, and not when the target absorbs that
ratio. This is fact 2 of the run sheet, now measured at equilibrium on two
seeds.

Step 4 (optional) was not run. Its condition is that Step 3 under pi is
close to the threshold, which the pre-registered result is not, and the
p = 0.3 question it was meant to inform was answered by the two extra long
runs above.

## Step 6. The scaling experiment (added 2026-10-22, after the verdict)

Why. The kill test gives one number on one state. The paper's thesis is
that the exact tree move is an independence sampler from the untempered
spanning-tree measure on balanced splits, so its acceptance falls as the
target's conditional on a merged pair moves away from that measure, and
the distance grows with the pair size at fixed temperature; the local
move's ratio involves a one-unit boundary change and does not. On Iowa
the tree-weighted control at the paper's mild temperature happened to
sit next to the proposal (forest acceptance 0.7, local move a passenger)
while pi_w sat far from it through the tree-count term (acceptance 0.17,
hybrid about 2x). The scaling experiment shows the mechanism directly, on
grids, for four targets, and is the section the student should own.

Prototype (cloud, 20,000 steps, hybrid chain, forest acceptance at
equilibrium against grid side N = 6, 10, 14): pi at lam = 1: 0.79, 0.70,
0.63 with the log tree-count ratio -0.9, -2.0, -3.8; pi at lam = 2:
0.29, 0.12, 0.044; tree-weighted at lam = 1: 0.23, 0.085, 0.028. The
collapse is fastest under the strongly tempered targets, which is the
compactness regime practitioners ask for, and it happens whether or not
the target carries tree weights.

Run (about 6 hours serial on the cloud machine's timings; the three
largest sizes dominate and can run in parallel by passing one N each):

    python3 recom/forest_scaling.py 50000 42 6,8,10,12,16,20 pi:1,pi:2,tree:0.1,tree:1 0.3 --fig | tee recom/iowa_hybrid_test_out/forest_scaling_seed42.log

or, in parallel, one size per process (each writes the same npz name, so
pass different seeds or run sequentially; simplest is three processes:
`6,8,10,12`, `16`, `20`, each with its own seed 42, 43, 44, then merge the
tables by hand; the figure needs one npz, so the serial command is
preferred if time allows).

Expected: panel (a) flat near 0.7 only for tree:0.1, decaying for the
other three, fastest for tree:1 and pi:2; panel (b) the per-proposal
mismatch growing roughly linearly in N; panel (c) the hybrid / forest
ESS ratio growing with N for the three decaying targets and staying
below 1 for the control. If (c) does not rise for pi:2 and tree:1 at
N >= 12, the local move's own acceptance is also collapsing (it did fall
from 0.42 to 0.20 between N = 6 and 14 in the prototype) and the paper's
move-size rule says to lower tau; rerun those two targets with tau = 0.03
(`TAU` at the top of the script) before concluding.

Writes `forest_scaling_seed42.npz` and `Papers/fig/forest_scaling.{pdf,png}`.
Commit both with the log.

### Results

(paste the table; keep the figure)

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
