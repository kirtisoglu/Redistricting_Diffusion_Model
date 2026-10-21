# TODO: paper.tex to Q1

## 2026-10-21: repository cleaned down to what Paper 1 uses

99 items moved, each listed with its original path in a `MANIFEST.md`.
`Papers/archive/` (gitignored, stays private) holds `paper2/` (shelved, compiles
in place), `backups/`, `earlier_drafts/` and `other_projects/` (the balanced-fibre
and Picard-quotient papers and the torsor notes, which are live work, just not
this paper). Root `archive/` mirrors original paths for Guided-MEW, plots,
scripts, `diffusion_model/mew` (holds the NC, PA, NH dual graphs needed for
precinct scale), the eight optimizer-paper scripts, two notebooks, the old plan,
and 26 shelved or stale artifacts. Verified after the move: all 21 kept scripts
import, all 38 artifacts Paper 1 uses are present, Paper 1 compiles clean at 31
pages. Root `archive/` is NOT gitignored; add it to `.gitignore` before pushing if
it should stay off the public repo.

Note also from 2026-10-14: the user judged that the partition-level tree move is
the wrong baseline (Autry et al.'s forest-lifted move keeps balanced-edge
selection and is exact), and that two papers from this story is not viable. The
recommended next step is to rebuild the hybrid on the lifted move and rerun the
kill test against it before any further writing.

## STATUS 2026-10-13 (evening): Paper 1 is submission-ready

Done in the submission pass, all compiled clean (30 pages, 0 errors, 0 dangling
references, 0 bibliography warnings):

- Full coherence read of Paper 1 after the split; ~20 stale sentences fixed
  (contributions preamble, Section 4.5 duplication, "member of the family",
  "rarely accepted" in Appendix B, balance-audit scope, table captions, L1/L3/L4,
  heading of 6.1, three colons).
- Bibliography: the four flagged entries verified on arXiv/Crossref and corrected
  (Gold, Micah; O'Sullivan, Philip; Proebsting, Todd and Ramsay, Alec);
  GerryChain attributed to the Metric Geometry and Gerrymandering Group (no DOI
  exists); DOIs added for Procaccia–Tucker-Foltz and Dobbs et al.; companion
  paper entry `kirtisoglu2026companion` added and cited; the Tapp journal-version
  lead was checked and does NOT exist, so it stays a preprint.
- Artifact cross-check: every cell of every table reproduces from the stored
  files. Fixes: Chain E on Iowa "averages 55 and ends at 69" (was "reaches 55");
  the 0.82/1.26 Iowa move-size subset stated; "about 7", "a few" wordings;
  ratio-convention notes in both hybrid captions; scripts now store `fast`,
  `flips`, `qp_err`, `n_plans`; `hybrid_iowa_long.py` added and named in App. C.
- **Non-determinism found and documented**: Iowa perimeter-kernel runs are not bit
  reproducible across sessions (identical for 21k/185k steps, then diverge);
  uniform and density runs are. Appendix C says so; the earlier session is kept
  as `hybrid_iowa_tau0.1_earlier_session.npz`. The τ=0.1 short-run table now
  uses the rerun (perimeter short-run ratios 0.90/4.44; text calls them
  unstable); the claim rests on the long runs.
- **Second seed for the long runs** (seed 43): perimeter 3.7x, density 4.7x
  (seed 42: 3.1x, 6.7x). Rows added to the Iowa table; contribution 2 and L4
  updated.
- Code and data availability section added with the repository URL and a
  promise of an archived release on acceptance.

**Only you can do these before submitting:**
1. `git add diffusion_model/recom/*.py diffusion_model/recom/iowa_hybrid_test_out/*.npz`
   and commit + push: 16 scripts, 5 modified sources and the artifacts are all
   untracked, and the availability statement promises them. (Papers/ is
   gitignored, which is fine; the paper itself need not be public.)
2. Confirm the short title "Diffusion Proposals for Redistricting Markov Chains"
   (it was shortened on disk before the split; I kept it).
3. Author block: ORCID / funding / acknowledgments if the venue wants them.
4. Paper 2 stubs (intro, discussion, conclusion) when you turn to it.

## STATUS 2026-10-13 (later): the split is made

User said "Split". Backup: `paper_backup_pre_split_20261013.tex`.

- **Paper 1** = `paper.tex`, 30 pages, main text 16,670 words, conclusion on
  p. 22. Keeps: theory (4.3), deterministic-rounding paragraph (4.4, why it
  blocks the correction), Chain E (4.5), mobility + law (4.6), exact tree move +
  hybrid + ergodicity proposition (4.7), calibration (5.1), 2x3 (5.2), hybrid
  validation and kill test (5.3), a new short Iowa calibration built from the
  hybrid's equilibrium (5.4: the nominal target's mean cut count on Iowa is ~90;
  tempered ReCom sits at 40), fiber diagnostic as Appendix B. Chains table is now
  V, A, E, T (tree), H (hybrid). Compiles clean, zero dangling refs.
- **Paper 2** = `paper2.tex`, a draft skeleton: new title and abstract, intro and
  conclusion stubs marked `%% TODO`, copied background (3.1-3.2) and model
  (4.1-4.2), then the moved blocks verbatim: Cheeger sweep + repair, Chain B +
  ergodicity, Chain C, Iowa compactness + optimizer baselines, NC, kernel
  showcase, least-change, ablations, diversity controls, hyperparameters;
  limitations L1/L5 and the dropped future-work items. Cross-references into
  Paper 1 were rewritten as "Paper~I, Section x". Zero dangling refs. Prose in
  the copied experiments intro still says "six chains" and mentions the 5x5; it
  is a skeleton for you to shape.
- Still yours: the four bibliography author lists, committing the scripts, and
  Paper 2's intro/discussion/conclusion.

## STATUS 2026-10-13: the research bucket, done and decided

Everything below was run today at low priority and is in the paper (new
Sections 4.8 and 5.3, the mobility law in 4.7, and the ripple edits to the
abstract, contributions, limitations, future work and conclusion).

**The reversible hybrid exists and is exact.** `hybrid.py` implements the tree
move of reversible recombination at the partition level (uniform spanning tree
by Wilson, cut edge uniform over ALL tree edges, fair-coin labeling, closed-form
density tau(A)tau(B)|d(A,B)| / (2|bp|tau(H)(n-1))), Metropolized to an arbitrary
kernel-weighted target, plus Chain E with the weighted target and the Lemma 4.1
linear-solve path, plus the mixture. Analytic kernel check on the 18-state fiber:
all three kernels satisfy detailed balance to 1e-15 under a uniform and a density
kernel. On the 4006-state fiber all chains sit within 1.8 floor SDs of their
enumerated targets, Hotelling p 0.21 to 0.88, under uniform and density kernels.

**The kill criterion, answered both ways.**
- 5x5 (exact balance): local kernel is a passenger. ESS/s ratios 0.11 to 0.27 at
  the locked timestep; retuned to tau = 0.3 with p_loc = 0.2 it reaches parity
  (1.07) at the sharp uniform target and 0.39 to 0.63 elsewhere. Never 1.5.
- Iowa (eps = 0.05): local kernel is decisive under weighted targets. Perimeter
  1.59 and 3.90, density 2.91 and 2.12 at the locked timestep; uniform marginal
  (1.56 and 0.38). The tree move alone accepts 0.6 to 1.2 percent and has NOT
  converged in 2e5 steps (half-means differ by 1.5 to 4 units); the hybrid has.
  tau = 0.3 does not transfer: the balance window binds more often, perimeter
  falls to 1.05/1.27, density holds at 2.8/3.4.
- Mechanism, both consistent with Section 4.7: exact balance kills the local
  move's first-order channel; a loose tolerance restores it; and a uniform tree
  edge rarely balances fifty heterogeneous counties.

**The mobility law is derived and confirmed.** Two-swap channel with the
Metropolis factor matches measured mobility at 1.00 (tau 0.1) and 1.02 (tau 0.3),
20,000 steps per tau. Without the Metropolis factor it overpredicts by 20%.
Small-tau form 2 tau^2 S_2 valid only to tau ~ 0.03. `mobility_law.py`.

**Caveats to keep honest.** Tree-only ESS on Iowa is 6 to 14 at 2e5 steps, so
those ratios are rough; the direction and the convergence contrast are solid.
The 2e6-step tree-only run (`hybrid_iowa_treeonly_2e6.npz`) confirms the tree
move reaches the hybrid's equilibrium (quarter means 27.6-29.4 vs 27.9 perimeter;
16.4-18.3 vs 18.2 density) with ESS 28 and 20 over 1.8e6 retained steps, i.e.
IAT ~ 1e5 and ESS/s 0.07 / 0.06, LOWER than the short-run estimates. The matched
2e6-step hybrid run (`hybrid_iowa_hybrid_p0.1_2e6.npz`, p_loc 0.1) gives ESS 126 /
186, ESS/s 0.226 / 0.394, same equilibrium. **Matched ratios: perimeter 3.1x,
density 6.7x.** These are the numbers in the paper. ESS on the 5x5 is hundreds.

**What this does to the Q1 question.** The paper now has the piece that was
missing: an exact sampler for geography-weighted compactness targets whose local
component is shown, on real geography, to be what makes it reach equilibrium.
That is a sampler contribution with no prior art found. Remaining Tier 3: the
tree move's acceptance (lift to forests, Autry-style, to recover balanced-edge
selection), precinct scale, and the downstream partisan analysis. Those are the
next research items; the split (Tier 4) is still your call and the paper is 39
pages with the Chain C block in.

## STATUS 2026-09-21: the rewrite is done, the split is not

**Tally 2026-10-21.** 70 of 86 items ticked below: 64 done and 6 closed as not applicable after the split. Of the 16 still open, 7 are partly done and say what is left. The open work is three kinds: small items (Chain A's acceptance rate, stamping one artifact, 5x5 visit arrays, the uncited RevReCom preprint, bibliography cleanup, the table-to-figure ratio), steps only you can take (commit, release and DOI), and research (the lifted-move rebuild, a Vanilla long run, more states, downstream analysis). The first tally, on 2026-10-13, was 53 of 85.

Backups: `paper_backup_pre_q1_rewrite_20260921.tex` (before today),
`paper_backup_pre_float_change.tex` (before tables were allowed to float),
`ref.bib.pre_fix_20260921`.

**Done today, all compiled clean (no errors, no undefined references, no
bibliography warnings).**

- Bibliography: the four wrong entries fixed (Brous, Najt, Tapp, Chuang), three
  minor ones fixed, the duplicate removed, fourteen prior-art entries added.
  **Verify first names against arXiv for `clelland2021compactness`,
  `gold2026adversarial`, `osullivan2026generalized`, `tapp2025parameter`** before
  submission; the audits supplied surnames only for some coauthors.
- Title, abstract (now ~190 words, two priority claims, both scoped), contributions,
  positioning (Zanella, randomize-then-optimize, Autry, Bonsai), roadmap.
- Related work: Clelland 2021, Procaccia and Tucker-Foltz, Herschlag 2020, CPT
  STOC'24, Gold 2026, generalized SMC, repetition effects, the Atlas, Tapp et al.
  parameter effects, Gurnee and Shmoys, Swamy et al., Dobbs et al. PSPACE claim
  qualified. "Mixing bounds open" qualified.
- New Section 4.3, theory of the diffusion proposal: Lemma (row-stochastic, box
  never binds, linear solve), Proposition (exact move size as a filter norm, with
  the spectral formula), Corollary (small-tau law, monotonicity, ceiling), all with
  proofs, plus verification paragraph and the spectral figure.
- New Section 4.7, mobility: the acceptance-sweep table, the three findings, the
  interior optimum, the mistuning of the locked setting.
- Section 4.6: labeled-plan statement added; "rarely accepted" corrected to
  "rarely feasible"; Remark 4.1 states the tolerance.
- Section 5.1 rewritten around distributions: the TV-with-floors table, the
  calibration figure, "essentially on" replaced by the 6.7-sigma statement.
- New Section 5.2: million-step 2x3 validation with batch means, p = 0.61, TV
  shrinking at the square-root rate, the thinning warning.
- All numeric slips fixed (1.9, 16.04, -4.1%, ±1.0, 44x/48x/40x, 16.75, balance
  range, "over matched compute", timing convention stated in both captions).
  Cohen's d demoted to descriptive. Least-change claim cut to a tie plus
  determinism, with Dobbs et al. named as the exact benchmark.
- Fiber diagnostic and ablations moved to Appendices B and C. Redundancies cut
  (W1/W2 block, Table 7, the three-benefit list, sweep-guarantee elaboration,
  Section 6.1 halved, L1 and L6 shortened, Appendix A.3 deduplicated, Figure 1
  shrunk).

**Length.** 38 pages total, main body ends on page 28. The redundancy cuts are
exhausted. What remains is substance, and almost all of the excess is the Chain C
block: Section 4.8 (about 3 pages) and Sections 5.3 to 5.6 (about 9 pages). That
is the Paper 2 material. Removing it takes the paper to roughly 24 pages with a
single spine. **That decision is yours and is the next thing to decide.**

**Not done, and why.** The reversible hybrid (Tier 3), more states, and the
downstream analysis. Those are research, not editing. The acceptance-rate law is
sketched in Section 4.7 and left to future work in the text.

---

Derived from three audits run 2026-09-18: citation integrity, artifact verification
(every headline number recomputed from stored `.npz`), and a literature survey for
missing prior art.

**Read this first.** Tiers 0-2 make the paper sound, honest and correctly positioned.
They do not make it Q1, because after correct positioning every current contribution
has identifiable prior art. Tier 3 is what Q1 requires. Decide Tier 4 early, because
it changes how you write Tiers 1-2.

---

## CRITIQUE OF THE TIERED PLAN (2026-09-21) AND A BETTER ONE

The tiers below are a fair item bank. As a plan they have eight problems.

1. **They repair the old paper instead of writing the new one.** After the last
   three days the spine is the diffusion theory: a lemma, an exact move-size law, a
   provable ceiling, and a mechanism for Chain E's behavior. The tiers are still
   organized around patching a draft whose spine was the optimizer. Half of Tier 0
   polishes text that should be deleted.
2. **Tier 3 hides the entire Q1 case in one bullet and never asks the question that
   can kill it.** "Build the reversible hybrid" is Chain E's local kernel mixed with a
   tree move Metropolized to the target. Autry, Herschlag, Mattingly (2023) already
   Metropolize forest recombination to a stated target. So a referee will ask what
   the local kernel adds. If the honest answer is "small moves that rarely land",
   the hybrid IS Autry et al. and the paper has no sampler contribution. The plan
   must answer this before any engineering, not after.
3. **No kill criteria.** Nothing says what to do if the hybrid mixes no better than
   the global move alone. That is a live possibility given the 2x3 residual, which
   showed Chain E slow *within* level sets, precisely where a local kernel is
   supposed to help.
4. **The scale target is aspirational.** "Use the 50-State Atlas" with a chain that
   takes 33 minutes for 500 steps on North Carolina. Two or three states at county
   or VTD level is the honest target.
5. **The downstream analysis has no purpose in the current spine.** It is listed
   because the field expects it. It should be there because it answers a question
   the paper poses.
6. **The acceptance-rate derivation is filed as optional.** It is required. You
   cannot tune the hybrid with half a tuning theory.
7. **Tier 0 first wastes effort.** Most of those edits touch prose that the rewrite
   will replace. The exceptions, which must be done today regardless: the four
   bibliography entries, Elle Najt's name, and committing the code.
8. **The split is "decided early". Decide it now.** Every downstream choice depends
   on it, and the answer is yes.

### The better plan: one question, one answer

**Question the paper asks:** what is the right target distribution for a
redistricting sampler, and how do you sample it exactly?

**The reframe that fixes problem 2.** Stop treating the kernel as a property of the
*proposal* and make it a property of the *target*:

    pi_w(P)  ∝  exp( -lambda * w(boundary(P)) )

where w is the perimeter, density, or any nonnegative edge kernel. This costs one
line in the acceptance ratio. Chain E's detailed-balance proof goes through
unchanged, since only the target ratio changes. The Laplacian in the proposal and
the weights in the target are now the SAME object, so the diffusion move is matched
to the geometry of the target by construction, which is exactly what an informed
proposal (Zanella 2020) is supposed to be. The tree move is Metropolized to pi_w
using the RevReCom spanning-tree ratio times the weighted target ratio.

Now the local kernel has an unambiguous job, the kernel flexibility changes the
stationary distribution *on purpose and in a characterized way*, and the claim
becomes "the first exact sampler for geography-weighted compactness targets".
Commissions and courts argue about perimeter and density, not edge counts. That is
a claim with a constituency, and none of the twenty-plus surveyed papers makes it.

**Spine of Paper 1**, in order, each section serving the question:

1. Practitioners want geography-aware compactness. Existing samplers either ignore
   it or shift the target uncharacterizably. Cite Clelland 2021, Procaccia and
   Tucker-Foltz 2022, Zanella 2020, Autry 2023, Bonsai 2026 here as the ground you
   stand on, not as competitors you missed.
2. Even the unweighted target is not what ReCom samples. The distributional
   calibration, z = 220. Chain A versus Chain E on the enumerated fiber.
3. The diffusion proposal and its theory. Linear-solve lemma, exact move-size law,
   small-tau law, the ceiling `2 m f (1-f)`, and the acceptance law once derived.
   This is the section nothing else has.
4. Exact correction. Chain E on labeled plans, the detailed-balance proof, the 2x3
   validation with batch means.
5. The hybrid. Tree move Metropolized to pi_w plus the diffusion move. Validated
   against the enumerated pi_w on the 5x5 for two kernels. ESS per second against
   the global move alone. **This is where the kill criterion is evaluated.**
6. Two or three real states. The choice of kernel changes the ensemble's partisan
   statistics. That is the downstream section with a purpose: the definition of
   compactness is consequential for outlier analysis, and here is a sampler that
   lets you choose it exactly.

Chains B, C and D, the least-change experiment, the ablations and the fiber
community diagnostic are not in this paper. B and D become one paragraph. C and
least-change are Paper 2.

**Kill criterion, stated in advance.** On the 5x5 fiber and on Iowa, at matched
wall-clock, if the hybrid's effective sample size per second on pi_w is not at
least 1.5x the Metropolized tree move alone, the local kernel has no demonstrated
value. Then Paper 1 becomes sections 1 to 4 only, submitted to Statistics and
Computing or a Q2 venue, and it is still a good paper. Do not spend more than
three weeks on section 5 before evaluating this.

**Milestones**

| when | what | gate |
|---|---|---|
| now | fix the four bib entries and the name; commit the ten scripts and four modified sources; decide the split | none |
| week 1 | acceptance-rate law from `acceptance_sweep.py`; weighted-target Chain E (one line) and re-run `db_check_2x3.py` under pi_w | detailed balance still 1e-15 |
| weeks 2-3 | tree move Metropolized to pi_w with the RevReCom ratio; validate against enumerated pi_w on 5x5 | TV inside the ESS floor |
| week 4 | hybrid; ESS per second versus global-only | **kill criterion** |
| weeks 5-7 | Iowa plus two states; kernel changes downstream statistics | effect visible at ten seeds |
| weeks 8-9 | write Paper 1 around the spine; Tier 0 polish last | |
| after | Paper 2 from Chain C, least-change, benchmarks against Dobbs et al. and Validi-Buchanan | |

**Risks named.** The RevReCom ratio at precinct scale is expensive; county and VTD
level is the target. The weighted target invites "which kernel is right"; the
answer is that the paper supplies the tool and the jurisdiction supplies the
criterion, with demographic kernels excluded per the Shaw v. Reno note already in
Section 4.1. The field is moving; three directly relevant papers appeared in 2026.
Nine weeks is the budget, not the aspiration.

---

## RESULTS IN HAND (run 2026-09-18)

### A. The calibration claim, now tested at distribution level

`calibration_distance.py`, on the stored long run. Read the z-scores against the
ESS noise floor, not the raw distances.

| Chain A, cut marginal | TV | floor (ESS) | z |
|---|---|---|---|
| nominal target | 0.6290 | 0.0067 | **+220** |
| tree-weighted | 0.0648 | 0.0060 | +21.7 |
| tree-weighted + tempered | 0.0219 | 0.0059 | +6.7 |

| Chain E, cut marginal | TV | floor (ESS) | z |
|---|---|---|---|
| nominal target | 0.0786 | 0.1405 | **-1.2** |
| tree-weighted + tempered | 0.5318 | 0.1246 | +7.5 |

**What this establishes.** The nominal target is refuted for Chain A at 220 floor
standard deviations, TV 0.63. That is no longer an inference from a mean gap, it is
a distributional fact. Chain E, by contrast, is statistically indistinguishable from
the nominal target it is proved reversible for, and is decisively far from the tree
measure. The two chains separate exactly as theory predicts. **That contrast is the
paper's strongest single result and it did not exist yesterday.**

**What it does NOT establish, and the draft must not claim.** Chain A is not an exact
match to the tree-weighted-tempered measure either. It sits 6.7 floor standard
deviations above it. So the correct sentence is that the tempered chain's cut
distribution is close to but distinguishable from the spanning-tree-weighted
tempered measure, and far from its nominal target. The current wording, "sit
essentially on their tree-count-weighted quasi-targets", is now measurably too
strong. The honest version is more interesting anyway, because vanilla ReCom's
stationary law is not exactly the tree measure, which is why RevReCom exists.

Caveat: on the full 4006-state distribution the ESS floor is 0.15 to 0.23, so that
test cannot separate the tree measures. The cut marginal has the power. Say so.

- [x] Rewrite Section 5.1 around the table above. Figure at `fig/calibration_5x5.pdf`.
- [x] Drop "essentially on"; state the 6.7-sigma residual and why it is expected.

### A2. Chain E on the 18-state fiber, where convergence is checkable

200,000 steps, `long_run_2x3.py`, then `gof_2x3.py` for the test.

| quantity | value |
|---|---|
| coverage | 18/18 |
| acceptance rate | 0.784 |
| mean cuts vs exact target | 2.761400 vs 2.760015, gap 0.0014 |
| TV to the exact target | 0.0250 |
| noise floor at the state-occupancy ESS | ~0.019 |
| Hotelling T-squared, batch means | F(17,129) = 1.60, **p = 0.074, not rejected** |

This is far stronger evidence than the 5x5 section currently offers, where the
chain sits at TV 0.90 having seen 403 of 4006 states. Here it covers the fiber,
the mean matches to four decimals, and the occupancy distribution is not
statistically distinguishable from the exact target.

**A METHODOLOGICAL WARNING, because this nearly produced a false alarm.** The
first test thinned by N/ESS and rejected at p = 0.0001. That test was invalid.
The ESS came from the CUT-COUNT trace, which takes only two values on this fiber,
so it understates the 18-state occupancy correlation time by about 1.5x. As the
dependence was handled properly the p-value moved 0.0001, then 0.045, then 0.074.
Monotone movement toward non-rejection as the diagnostic is fixed is the signature
of a bad diagnostic, not a bad sampler. **Never thin an occupancy test by a coarse
summary statistic's ESS.** The same trap applies to every ESS number in the paper,
all of which are computed from cut traces.

**Honest residual.** States 17, 15 and 10 are persistently under-visited and 7, 13
and 6 over-visited, all with cut count 3, where the target is exactly uniform. So
the residual sits inside a level set, which is where a sticky local chain is
slowest to equilibrate. Detailed balance is verified to 1.5e-15 and coverage is
18/18, so the target is the unique stationary law and the chain must converge. The
residual should therefore shrink like the square root of the run length.

- [x] **Confirm it shrinks.** Rerun at 1e6 steps, about 17 minutes at the measured
      750 steps per second. If TV falls by roughly half, report the 2x3 validation
      as the end-to-end check and retire the weak 5x5 "converges in mean" claim. If
      it does not fall, stop and investigate the sampler.
- [ ] *(partly done: reported for Chains E, T and H; Chain A's acceptance rate is still missing)* Report the acceptance rate, 0.784 here, and add acceptance rates for every
      filtered chain in the paper.
- [x] Use batch means, not thinning, for any distributional test on a chain.

### B. The diffusion connection, made load-bearing

`spectral_move_size.py`. All verified; the Gurobi solve matches the closed form to
1.1e-4 on 120 pairs.

**Lemma (the box constraints never bind).** `I + tau L` is an M-matrix, so its
inverse is entrywise nonnegative, and `L1 = 0` gives `(I + tau L)^{-1} 1 = 1`. The
inverse is therefore ROW-STOCHASTIC, so `x* = (I + tau L)^{-1} x0` is a convex
combination of the entries of `x0` and lies in `[0,1]^n` automatically. Verified:
0 violations in 10,766 solves. **Consequence: whenever the balance constraint is
slack, the QP is one linear solve, not a call to a QP solver.** On the 5x5 fiber
balance is never active, so the proposal is a linear solve at every step.

**Proposition (exact move-size law).** Under single-shot Bernoulli rounding a unit
flips with probability `|x*_v - x0_v|`, so
`E[units flipped] = ||x* - x0||_1 = ||(I + tau L)^{-1} tau L x0||_1`.
A high-pass filter norm, closed form, exact at every tau, no simulation.

**Corollary (small-tau law).** `E[units flipped] = 2 tau |boundary| + O(tau^2)`,
the boundary measured in the proposal's own kernel. Median relative error 2.5% at
tau = 0.01, 7.4% at 0.03, 24% at 0.1.

**Finding (saturation).** Move size is bounded in tau. On the 5x5, raising tau by
1000x raises the move by 76x, and from tau = 1 to tau = 10 by only 2.1x. **Mobility
cannot be bought with alpha/beta.** This is a proof-shaped reason the reversible
hybrid is necessary rather than merely desirable, and it is far stronger than
observing that Chain E happens to be slow.

**Finding (what limits moves on real geography).** The balance constraint is active
for 41% of Iowa merged pairs at the paper's own tau = 0.1, rising to 86% at tau = 10,
and is never active on the uniform grid. On real county geography the population
constraint, not the diffusion, caps the move.

**Finding (Chain E explained quantitatively).** At tau = 0.1 the expected move is
0.42 units on the 5x5 and 0.82 on Iowa. Under one unit per proposal. That is why it
visits 6 of 4006 states in 1000 steps. The stickiness is the timestep, not a tuning
accident and not a mystery.

Write-up items:

- [x] Promote Section 4.2 from one interpretive paragraph to a short subsection with
      the lemma and the proposition, both with proofs. They are a few lines each.
- [x] Replace the Appendix A.2 hand-wave "beta >> alpha keeps x* near-binary" with
      the move-size law. It says exactly how near, and why.
- [x] State the design rule: to propose moves of size m, set
      `alpha/beta = m / (2|boundary|)`, valid in the small-tau regime, with the
      saturation and balance-activity caveats.
- [x] Report the linear-solve shortcut as a cost result. The continuous proposal is
      far cheaper than "solve a QP" implies.
- [x] Use saturation to motivate the hybrid in Section 6.4, replacing the current
      empirical hand-off.
- [x] Figure at `fig/spectral_move_size.pdf`.
- [x] *(done: two-swap mobility law with its Metropolis factor, Section 4.6)* Extend the law to the acceptance rate. `acceptance_sweep.py` below supplies the
      data and the mechanism; the derivation is the week-1 milestone.

### C. Acceptance and mobility against the timestep (2026-09-21)

`acceptance_sweep.py`, 3000 steps per tau on the 5x5 fiber, alpha = tau, beta = 1.

| tau | E[flips] | single-flip share | feasible given single | P(feasible) | accept given feasible | units moved per step | states reached |
|---|---|---|---|---|---|---|---|
| 0.01 | 0.036 | 0.035 | **0.000** | 0.964 | 1.000 | 0.0000 | 1 |
| 0.03 | 0.126 | 0.110 | **0.000** | 0.884 | 1.000 | 0.0033 | 6 |
| **0.10** | 0.385 | 0.268 | **0.000** | 0.683 | 0.999 | 0.0093 | 12 |
| **0.30** | 0.909 | 0.349 | **0.000** | 0.437 | 0.989 | **0.0353** | **38** |
| 1.00 | 1.914 | 0.254 | **0.000** | 0.193 | 0.983 | 0.0313 | 37 |
| 3.00 | 2.672 | 0.168 | **0.000** | 0.074 | 0.986 | 0.0187 | 19 |

**Four findings, each of which changes a sentence in the paper.**

1. **Every single-unit proposal is infeasible, at every tau.** Predicted: exact
   balance means one flip gives sizes 4 and 6. The leading-order move of the
   diffusion proposal is therefore always rejected on this fiber, and the chain
   lives entirely on the second-order coordinated draws. This is the mechanism
   behind "6 of 4006 states", derived rather than observed.
2. **The Metropolis ratio is never the bottleneck.** Given a feasible proposal,
   acceptance is 98 to 100 percent throughout. Chain E's stickiness is an
   infeasibility problem, full stop. Section 4.5's sentence "moves that flip many
   units are rarely accepted" is wrong in an instructive way: they are rarely
   *feasible*, and when feasible they are accepted.
3. **Acceptance rate is a misleading diagnostic for this chain.** At tau = 0.01 the
   acceptance rate is 96 percent and the chain never leaves its starting state,
   because 96 percent of proposals flip nothing. Report units moved per step, not
   acceptance. This matters for the recommendation elsewhere in this file to add
   acceptance rates: add them, but for Chain E pair them with move size.
4. **There is an interior optimum, and the paper is on the wrong side of it.**
   Mobility peaks at tau = 0.3, at 0.035 units per step. The locked setting
   tau = 0.1 delivers 0.009, about 3.8x below optimum. The paper's own theory says
   Chain E is mistuned, and retuning it is free. Even at the optimum, though, a
   thousand steps move about 35 units. The ceiling is real, so the hybrid stays
   necessary. Both statements belong in the paper: E can be 4x better than shown,
   and 4x better is still not enough.

**Shape of the acceptance law, for the week-1 derivation.** Since acceptance given
feasibility is ~1, mobility is essentially `sum_k k * P(k flips) * P(feasible | k)`.
The flip count is Poisson-binomial with parameters `|x*_v - x0_v|`, which the
move-size law gives in closed form. `P(feasible | 1) = 0` under exact balance and
`P(feasible | 2)` is the fraction of cross-boundary unit pairs whose swap preserves
contiguity and balance, a graph quantity computable from the current plan. So the
small-tau mobility is second order in tau with a computable coefficient, and the
interior optimum is where the growth of `P(k >= 2)` meets the decay of
`P(feasible | k)`. Under a loose tolerance single flips become feasible and the
law gains a first-order term, which is why E moves on Iowa and not on the grid.

- [x] *(done: matches measured mobility at 1.00 and 1.02)* Derive the second-order mobility coefficient and check it against the table.
- [ ] *(partly done: hybrid runs at tau = 0.3 exist, but Chain E's own 5x5 long run was not repeated, and the optimum does not transfer to Iowa)* Retune Chain E to the mobility optimum and rerun the 5x5 long run.
- [x] Replace "rarely accepted" with "rarely feasible" in Section 4.5.

---

## Tier 0 — Must fix regardless of venue (~2-3 days)

Nothing here adds novelty. All of it is credibility. A referee who checks will find
these, and two of them are embarrassing.

### Bibliography errors

- [x] `brous2025recom` — three errors in your closest competitor's entry.
      Author is **Daniel Brous**, not "Adam Brous". Real title is
      *Optimizing Districting Plans to Maximize Majority-Minority Districts via IPs
      and Local Search*; the current title matches no published work. Venue is
      ACDA 2025, pp. 264-275, DOI `10.1137/1.9781611978759.20`, arXiv:2508.07446.
- [x] `najt2024empirical` — four errors, the worst entry in the file.
      Correct citation: **Elle Najt**, Daryl DeFord, Justin Solomon,
      *Empirical sampling of connected graph partitions for redistricting*,
      Phys. Rev. E **104**, 064130 (2021), DOI `10.1103/PhysRevE.104.064130`.
      Drop arXiv:1908.08881, which belongs to a different paper. The entry currently
      lists a former name for the first author; please get this one right.
- [x] `tapp2025compactness` — title is *Revisiting Compactness for District Plans*, *(title fixed; whether a journal version exists is still unchecked)*
      not "...for Redistricting". Also check directly whether a journal version now
      exists; an unconfirmed lead points to npj Complexity 3, 23 (2026), but arXiv,
      Crossref and Semantic Scholar all still show preprint only.
- [x] `chuang2025multiscale` — add volume 23, issue 4, pp. 1515-1550,
      DOI `10.1137/24M1635806`. As written it looks like a guessed venue.
- [x] `cannon2023voting` — `number={36}` is an article number and renders as "25(36)".
- [x] `fischetti2003local` — issue is 1-3, not 1.
- [x] `charikar2023complexity` — still preprint-only; canonical arXiv year is 2022.
- [x] *(done: no DOI exists for the Python package; attribution corrected)* `gerrychain2024` — pinned to v0.3.1 (Mar 2024) but v1.0.0 shipped Aug 2026.
      Add a Zenodo DOI or commit hash.
- [x] Delete the `mew` entry; it duplicates `mcwhorter2025marked` exactly.
- [ ] Clean or delete the 41 uncited entries. Several are malformed:
      `esedog2015threshold` (author mangled), `chang2018lov` (title renders as
      `Lov\'asz`), `trevisan2017lecture` (URL in the journal field),
      `duchin2021political` (Springer 2021; the book is Birkhäuser 2022, which
      contradicts `becker2022redistricting` in the same file).
- [x] Re-run bibtex and regenerate `paper.bbl` and `paper.pdf`. The compiled
      bibliography faithfully reproduces every error above.

### Numbers and internal consistency

- [x] **Wall-clock conflict.** Table 9 says Chain C at n=10 takes 57.2 s; Table 12 *(convention now stated in both captions; not re-timed under one convention)*
      says 118 s. Verified to be the *identical deterministic run*, bit-for-bit on
      the cut traces. The headline sums per-step times (57.2 s); the ablation wraps
      one timer around the whole call (117.8 s). Instrumentation explains 0.015 s of
      the 60.6 s gap; the rest is machine-state variation during a 38-config sweep.
      Pick one convention, re-time, state it in both captions. Note the
      "five to ten passes quadruples the runtime" sentence is 3.8x under one
      convention and 1.8x under the other.
- [x] *(closed, not applicable: the experiment moved to the shelved Paper 2)* Save the SA-5000 wall clock (9.7 s). It is printed but never stored, and it
      carries the paper's central "annealing at a sixth of the cost" argument.
- [x] *(closed, not applicable: the experiment moved to the shelved Paper 2)* Save the Vanilla+IQP wall clock (53 s); `vanilla_iqp_cell.py` stores only
      cuts/lam/seed.
- [ ] Save per-state visit arrays for the 5x5 runs. The unique-state and TV columns
      of Table 8 are correct but cannot be regenerated from any stored artifact,
      which contradicts Appendix C.
- [x] "1.6 to 1.9 cut edges below the nominal target" (intro) — actual values are
      1.609 and 1.845. Nothing supports 1.9. Section 5.1 says "1.6 and 1.8".
- [x] 5x5 ten-seed Chain C mean: paper says 16.04, artifact gives 16.0346 → 16.03.
- [x] Same quantity appears as -4.1% (statistical protocol) and -4.0% (§5.1).
      Actual -4.05%.
- [x] State the standard-deviation convention. Iowa Chain B ten-seed spread is
      printed as ±1.0, but `ten_seed_stats.py` reports 0.93 with the population
      convention used for every in-table std. ±1.0 needs the sample convention.
- [x] Per-step ratios: actual 47.4x (C/V) and 45.4x (C/B). Paper says ~48x and ~44x
      in Tables 7 and the trade-off text, and 40x in the discussion.
- [x] Reconcile Chain A's 5x5 mean, which appears as 16.71, 16.73, 16.75 and 17.02
      in four places. Say which is the ten-seed mean, which is single-seed, which is
      the long run.
- [x] Remark 4.1: `db_check_2x3.py` runs at eps = 1/3, not the 0.05 used everywhere
      else, so the 18 states include 4+2 splits. State the tolerance. Also fix the
      script docstring, which claims "3+3 split" while the code enumerates sizes 2-4.
- [x] Balance audit: the stated Iowa range 0.0498-0.0500 does not cover the ablation
      runs, which reach 0.0494-0.0495. No violation of eps = 0.05, but the claim says
      "every step of every run reported here".
- [x] Drop "over matched compute" in §5.2. Restarted SA takes 26 s against Chain C's
      57 s, a 2.2x gap.
- [x] Put the long-run reproduction command in the paper. `long_run_5x5.py` defaults
      to 1e6/5e5 steps; the paper used 2e5/1e5.

### Statistics

- [x] Drop Cohen's d, or explain it. It is not a valid effect size on an
      autocorrelated trace; the denominator is the chain's equilibrium spread, not
      sampling noise.
- [x] *(done: Paper 1 no longer contains single-seed z-tests)* Drop or heavily qualify the single-seed ESS z-tests. Let the ten-seed paired
      tests carry every headline claim; those are sound.
- [x] Do not report Chain C's 5x5 ESS as "in the hundreds". Its trace is nearly
      frozen at 16 with std 0.19; a frozen series cannot be near-independent. The
      estimator is unstable there and it inflates exactly the comparison you headline.
- [ ] *(partly done: reported for Chains E, T and H; Chain A is still missing)* Report acceptance rates for every chain carrying a Metropolis filter. There are
      currently none anywhere in the paper.
- [x] *(done: those comparisons left Paper 1 with the split)* Stop attaching p-values to optimizer-versus-sampler mean comparisons. Chain C's
      mean cut count and Chain A's are not the same kind of quantity.

### Reproducibility

- [ ] *(still open; the list is now the 21 scripts in Appendix C, since the optimizer scripts moved to archive/)* Commit the 10 scripts named in Appendix C. All are currently untracked:
      `least_change.py`, `matrix_tree_targets.py`, `db_check_2x3.py`,
      `ablation_sweep.py`, `ensemble_compare.py`, `sa_baselines.py`,
      `kernel_traj_metrics.py`, `multikernel_figure.py`, `long_run_5x5.py`,
      `ten_seed_stats.py`.
- [ ] Commit the four modified sources: `chains.py`, `qp_model.py`, `problems.py`,
      `diagnostics.py`. The tracked versions do not match what produced the results.
- [ ] *(still open for long_run_5x5_seed42.npz, which has seed and lam but no alpha, beta or step count; the other three files moved to archive/)* Add config and seed stamps to `kernel_traj_metrics_seed42.npz` (currently has
      neither), `long_run_5x5_seed42.npz` (no alpha/beta/steps),
      `vanilla_plus_iqp_seed42.npz`, `sa_baselines_seed42.npz`.
- [x] *(done 2026-10-21: archived with a manifest)* Remove or clearly label stale artifacts: `target_matched_iowa_seed42_v11.npz`
      sits beside the current file, and the May 1 `target_matched_{iowa,5x5}.npz`
      predate Chain E and carry no stamp. A referee cannot tell which backs the paper.
- [ ] *(partly done: the repository URL is in the paper; release and DOI are promised on acceptance)* Tag a release, archive it for a DOI, and put the link in the paper. There is
      currently no repository URL anywhere in the text.

---

## Tier 1 — Free wins from data already on disk (~1 week)

**Three scripts are written and ready. None has been run.** Run them in this order.

- [x] **Run `calibration_distance.py`.** The decisive experiment. Reads
      `long_run_5x5_seed42.npz`, `matrix_tree_targets.npz` and the fiber JSON;
      writes only the new `calibration_distance_seed42.npz`. Compares each chain's
      empirical distribution against all three reference measures, on both the full
      state space and the cut-count marginal, with parametric-bootstrap noise floors
      at the raw step count and at the effective sample size. Upgrades the central
      claim from "two scalars agree to 0.07 cuts" to a distributional result.
      - The headline statistic is the **cut-count marginal**, not full-state TV.
        About 20 distinct values against 1.8e5 draws means a tiny noise floor, so
        the comparison is immune to the coverage problem that makes the paper's
        current TV column uninformative.
      - Read the z-scores against the ESS floor, never the raw TV.
      - **If both distances sit above their floors, that is slow mixing, not
        convergence to the wrong target. Say so and run longer.**
- [x] **Run `fig_calibration.py`.** Builds the paper's missing figure into
      `Papers/fig/calibration_5x5.{pdf,png}`, both new filenames. Three panels:
      Chain A's marginal against the three references, TV against run length with
      the noise floor shaded, and Chain E against its own target. Colors are
      Okabe-Ito, validated for colorblind separation on all pairs, with distinct
      dash patterns so identity survives grayscale printing.
      - Open the output and check it. The palette was validated; the layout was not.
- [x] **Run `long_run_2x3.py`.** The only Tier 1 item needing real compute. Validates
      Chain E on the 18-state fiber where TV can actually reach zero, and yields a
      chi-square goodness-of-fit test against the exactly known target. Replaces
      "converges in mean", which is weak: at 1e5 steps on the 5x5 the chain sits at
      TV 0.90 against a floor of 0.085, having visited 403 of 4006 states.
      - Time 20000 steps first before committing to the full run.
      - It also prints an acceptance rate, of which the paper currently has none.
- [ ] Long Vanilla ReCom run against the pure tree-weighted measure.
      `long_run_5x5.py` only ever ran A and E, so V has no long trace and its
      distributional comparison cannot be done from stored data.
- [x] *(closed, not applicable: those panels are no longer in Paper 1)* Redo the Iowa and 5x5 histogram panels as small multiples or a ridgeline. Six
      translucent overlaid histograms in default colors is unreadable, and this is a
      graphical-statistics venue.
- [x] *(closed, not applicable: the figure belongs to the shelved Paper 2)* Add trajectory distributions to the multikernel figure. The current panel
      annotations contradict Table 12: the density snapshot shows the *shortest*
      boundary while the table claims density produces the longest.
- [ ] *(still open: Paper 1 has 11 tables and 3 figures)* Reconsider 14 tables and 3 figures. That ratio is backwards for JCGS.

---

## Tier 2 — Positioning and rewrite (~2 weeks)

Each contribution has prior art the draft does not engage. Under-positioning reads
as either not knowing the field or hoping the referee will not check.

### Prior art you must cite and distinguish

- [x] **Clelland et al., *Compactness statistics for spanning tree recombination*,
      arXiv:2103.02699 (2021).** Shows empirically that ReCom samples roughly in
      proportion to a decaying exponential in cut-edge count. This is the direct
      precursor of contribution 2. State precisely what is new in your version:
      exact target on an enumerated fiber, identification of the tree-weighted limit,
      quantified gap. Aggravating factor: you already cite this group's Colorado paper.
- [x] **Procaccia & Tucker-Foltz, *Compact Redistricting Plans Have Many Spanning
      Trees*, SODA 2022, pp. 3754-3771, arXiv:2109.13394.** The theorem that links
      your Matrix-Tree quasi-target to the cut-edge statistic. Without it the
      calibration reads as an unmotivated computation. Tucker-Foltz co-authors two
      works already in your bibliography.
- [x] **Zanella, *Informed Proposals for Local MCMC in Discrete Spaces*, JASA 115(530):
      852-865 (2020).** The canonical statistics framework for locally balanced,
      target-aware discrete proposals with exact MH. Chain E is structurally an
      instance. Rescope the "first" claim explicitly to redistricting.
- [x] **Bardsley et al., *Randomize-Then-Optimize*, SIAM J. Sci. Comput. 36(4) (2014).**
      The ancestor of optimize-then-correct-with-MH.
- [x] **Clelland & Tapp, *Bonsai*, arXiv:2603.18347 (Mar 2026).** Independent
      non-MCMC sampling from a specified distribution on partitions. Direct
      competitor to contribution 1.
- [x] **Cannon, Pegden, Tucker-Foltz, STOC 2024, pp. 1676-1687.** Already in
      `ref.bib` as `cannon2024sampling` and never cited. Then qualify the §2 claim
      that mixing-time bounds remain open, and limitation L2.
- [x] **Herschlag et al., Statistics and Public Policy 7(1): 30-38 (2020).** Origin of
      the compactness-tempered target your abstract measures deviation from.
- [x] **Gold, *Adversarial Configurations for the ReCom Transition Function*,
      arXiv:2606.01333 (2026).** Exhibits planar-grid configurations where ReCom needs
      exponentially many expected steps. Qualify "dynamics not topology" in §3.3/§6.2.
- [x] **Dobbs, King, Ludden, Jacobson, INFORMS J. Optimization 6: 214-239 (2024).**
      Exact MILP for minimal-change plans under transfer distance. This is your
      least-change problem, already solved exactly. Compare against it or drop the
      priority claim.

### Literature the draft under-cites

- [x] OR line, currently absent: Gurnee & Shmoys (ACDA 2021), Swamy, King, Jacobson
      (Operations Research 71(2), 2023), Shahmizad & Buchanan (OR 73(2), 2025),
      Belotti, Buchanan, Ezazipour (OR 73(5), 2025), Mehrotra, Johnson, Nemhauser
      (Management Science 44(8), 1998). Column generation and set partitioning appear
      nowhere.
- [x] Sampling line: O'Sullivan, Imai, McCartan generalized SMC (arXiv:2603.22188,
      2026); Cannon, DeFord, Duchin repetition effects (arXiv:2409.19017, 2024);
      McCartan et al. 50-State Redistricting Atlas (Scientific Data 9:689, 2022);
      Tapp, Proebsting, Ramsay parameter effects (arXiv:2505.21326, 2025).
- [x] *(closed, not applicable: least-change moved to the shelved Paper 2)* A legal or applied anchor for least-change, e.g. Wisconsin's least-change
      standard and its treatment in *Allen v. Milligan*. You run the experiment and
      engage none of that literature.
- [ ] `cannon2020reversible` is uncited while the text repeatedly says "RevReCom
      theory identifies" and "Compared with RevReCom".

### Claims to rescope

- [x] Cut the least-change claim to what Table 14 supports. The anchored IQP wins at
      one of six budgets by 0.10 cuts against a baseline with sd 0.30, and loses at
      four. That is a tie. The real advantages are determinism across ten seeds and
      no schedule to tune. Say that instead of "the strongest and most reproducible
      tool".
- [x] Reduce five separate priority claims to one or two.
- [x] Cut the abstract from 286 words to about 180.
- [x] State in §4.5 that plans are **labeled** assignments. The proof needs it, and
      the code and `db_check_2x3.py` both use it. As written with set notation the
      proposal map is two-to-one and the transition probability omits the
      complementary Bernoulli draw.
- [x] Qualify the PSPACE-completeness attribution in §2. It holds only under
      conditions on district count and step size.
- [x] Retitle or restructure so the body matches the title. The title promises
      exactness and calibration; §4 and most of §5 are about the optimizer.

---

## Tier 3 — What Q1 actually requires (~2-4 months)

The audits found no prior art for one thing: a mobile sampler with a known target
that accepts arbitrary geographic kernels. Bonsai samples independently, generalized
SMC does something else, and the grid samplers take no kernels. That gap is the paper.

- [x] *(done at the partition level and validated; superseded by the decision to rebuild it on the lifted move, see the next item)* **Build the reversible hybrid.** Chain E's exact local kernel mixed with a
      RevReCom move Metropolized toward the *same* pi via its computable
      spanning-tree-count Hastings ratio. Your own §6.4 names this as the single most
      promising follow-up. It is also every referee's conclusion.
- [ ] **Rebuild the hybrid on the forest-lifted tree move** of Autry, Herschlag and Mattingly, which keeps balanced-edge selection and stays exact, and rerun the kill test against it before any further writing. *(added: decided 2026-10-14)*
- [x] *(done: every chain within 1.8 floor SDs on 4,006 states, Section 5.3)* Demonstrate exact TV convergence on the enumerated 5x5 fiber, where it is
      checkable, then Iowa, then at least one precinct-scale state.
- [x] *(done: density kernel against its enumerated target, Section 5.3)* Show the hybrid retains a known target under a non-uniform kernel. That
      combination is the contribution.
- [ ] **Scale up.** Use the 50-State Redistricting Atlas for several states. Iowa plus
      a 25-cell grid will not clear Q1 review in 2026 when that benchmark exists.
- [x] *(done for Paper 1: Iowa runs are 2e5 and 2e6 steps over two seeds; NC left with the shelved paper)* **Run longer.** 500 steps with ESS in the tens is 20x to 2000x short of field
      practice. The NC run is single-seed and Appendix L5 already calls it "the
      pending scalability test" while §5.3 presents it as validation.
- [ ] **Add downstream analysis.** Efficiency gap, mean-median, seat share under each
      chain. This was Sprint 4 in `PAPER_PLAN.md` and was dropped. Without it, the
      "defensible null distribution for outlier analysis" framing has no demonstration,
      and that framing is the paper's stated reason to care.

---

## Tier 4 — Split (decide before writing Tier 2)

- [ ] *(partly done: Paper 1 exists without the downstream analysis, and its hybrid must be rebuilt on the lifted move)* **Paper 1, statistics venue.** Calibration, exact correction, reversible hybrid,
      downstream analysis. Target JCGS, then Statistics and Computing, then AoAS.
- [x] *(closed, not applicable: Paper 2 shelved on 2026-10-14)* **Paper 2, OR venue.** MIQP matheuristic, kernels, least-change, benchmarked
      against Validi-Buchanan exact optima, Dobbs et al. midpoints, and annealing,
      across multiple instances. Target INFORMS Journal on Computing or EJOR.
- [x] *(done: Chain B left Paper 1 with the split)* Chain B is a negative result in both. Keep it as one paragraph, not a chain.

---

## What is already solid (do not spend time here)

- Every headline number reproduced exactly from stored artifacts. The full Iowa
  table to four decimals, all 12 multikernel cells, all 18 least-change entries, all
  18 sensitivity cells, the full NC table, the Matrix-Tree targets, and the ten-seed
  paired statistics including p = 6.443e-13.
- No reference is fabricated. All 41 cited works exist and the compiled bibliography
  is in sync with the source.
- The detailed-balance proof is correct as implemented, verified against
  `qp_model.py:324-407`, and the kernel check gives 1.53e-15 on 18 labeled states.
- Writing style is compliant throughout: no em-dashes outside empty table cells,
  American spellings, plain wording.
- The honesty is genuinely rare. Keep the negative results, the self-reported bugs,
  and the baselines that beat you. Just stop letting them be the whole story.
