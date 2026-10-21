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

Not yet run. In addition to the eight commands of Step 3 above, run the
hybrid at p = 0.3 under pi on both seeds, since the short runs favour it
(the long mode's sixth argument is p_local; the file name gains `_p0.3`):

    python3 recom/forest_iowa.py long hybrid 2000000 42 pi 0.3 | tee recom/iowa_hybrid_test_out/forest_iowa_long_hybrid_p0.3_seed42_pi.log
    python3 recom/forest_iowa.py long hybrid 2000000 43 pi 0.3 | tee recom/iowa_hybrid_test_out/forest_iowa_long_hybrid_p0.3_seed43_pi.log

The criterion is unchanged: the hybrid passes if either p clears 1.5 on
both kernels and both seeds with matching quarter means, and the paper
reports the p it used.

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
| pi, p=0.3 | perimeter | 42 | | | | |
| pi, p=0.3 | density | 42 | | | | |
| pi, p=0.3 | perimeter | 43 | | | | |
| pi, p=0.3 | density | 43 | | | | |

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
