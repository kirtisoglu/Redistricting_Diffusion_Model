# Research notes

Running notes from the October 2026 sessions. Ideas, open questions and
follow-up projects that are not part of the current paper's run sheet
(`TODO_Q1.md`). Add to the end; date each entry.

## 2026-10-21. Urban vs rural from topology (DeFord's problem)

**The problem as DeFord states it.** How much of the full geographic
information is contained in the dual graph? Urban and rural units are
easy to tell apart geometrically (population, area); given only the
graph, can the same classification be recovered, and if not, what
additional information is necessary?

**Relation to the paper.** Converse, not the same. The density kernel
takes population and area as given attributes and asks how to sample
with them. DeFord asks whether the attributes are already implied by
the topology. We are downstream of his question. The kernel is,
however, exactly a formalization of "additional information", and an
exact sampler for kernel-weighted targets is the instrument that
measures what each piece of information changes in an ensemble.

**Why the topology should know something.** Urban precincts are small,
so the dual graph is dense there: more vertices per unit area, higher
degree, shorter cycles. The spanning-tree measure is sensitive to that
structure (this is part of why ReCom ensembles behave differently in
cities), so even the uniform-kernel target carries some density
information through the graph alone.

**Experiment that answers him with our tools** (needs a precinct
graph; the 5x5 grid is homogeneous by construction and Iowa counties
are too uniform in size):

1. Sample the uniform-kernel target on a precinct graph (topology
   only). Measure alignment of cut edges with the density gradient,
   e.g. the share of cut edges that cross an urban-rural boundary
   against the share expected if boundaries ignored density.
2. Sample the density-kernel target. Measure the same.
3. Reading: alignment already present under the uniform kernel means
   the topology knows where the cities are (partial yes). Alignment
   only under the density kernel means the information is not in the
   topology and the kernel quantifies the missing piece.

Needs: `archive/diffusion_model/mew/data/NC_VTD.json` restored to
`diffusion_model/mew/data/` (the `nc` mode of `problems.py` reads it),
and a timing probe of the forest move at precinct scale first.

**Student project (separate paper, student as first author).** The
classification half of the question: predict urban or rural from
graph features alone. Features: degree, local vertex density,
clustering coefficient, spectral embedding coordinates (leading
Laplacian eigenvectors), distance to high-degree cores. Labels: census
population density thresholded, or an official urban-area indicator.
Model: anything interpretable first (logistic regression, gradient
boosting), with a map of where it fails, since the failures are the
scientific content (suburbs, small rural towns, water boundaries).
Closing section: the sampler experiment above, showing what the
recovered classification changes in an ensemble. Fits the student's
timeline and gives him a self-contained result to present.

## 2026-10-21. Language and cost (why the community uses Julia or Rust)

Where the time goes in the Python implementation on Iowa under pi:
forest step about 1.1 ms (pure-Python Wilson draw, balanced-edge
scan, two determinants, crossing-edge sums, retry multiplies the
Wilson part); the paper's old tree move 0.18 ms (no sums, determinants
only on the one percent of draws that balance); local step about
1.5 ms, half of it Gurobi call overhead on the 20 to 50 percent of
pairs where the balance window binds.

Consequence for the claim: a compiled forest step would be 10x or more
faster, the local step much less so (Gurobi overhead is language
independent), so wall-clock ratios would move against the hybrid. Two
fixes, both for the paper regardless of language:

1. Report ESS per step beside ESS per second. Per step the hybrid at
   p = 0.3 gets 3 to 5 times the forest move's effective samples while
   taking 30 percent fewer forest steps, which is implementation free.
   State the measured cost ratio and let the reader apply their own.
2. Remove Gurobi from the local move. With one active balance
   constraint the QP has a closed form: the unconstrained solve shifted
   by a multiple of the population vector, found from one extra linear
   solve and a scalar root, with the box constraints checked after.
   Then a local step costs a few linear solves, in any language about
   what a forest step costs. Small change to the solve path in
   `hybrid.py`; do it before the journal version, after Step 3, since
   it changes cost but not the proposal law.

## 2026-10-21. Showing that the tree step needs a local move

The structural argument, to pursue after the kill test is decided
(recorded here so it is not lost; see the session discussion):

- Metropolized to a non-tree target, the tree move on a merged pair is
  an independence sampler; its spectral gap is the infimum of proposal
  over target, exponentially small in boundary length under any target
  with a lighter compactness tail than the spanning-tree measure
  (Procaccia and Tucker-Foltz). The penalty grows with pair size. The
  local move's ratio depends only on the boundary change and does not.
- Measured: under the tree-weighted target the forest move's ESS on
  Iowa is about 10,000 per 200,000 steps; under pi, 36 to 56. A factor
  of 200 at identical step counts is the tree-count penalty in one
  number.
- Experiments: (1) scaling with pair size on grids 6x6 to 20x20, k = 2,
  five percent tolerance, read the forest move's acceptance at the
  hybrid's equilibrium, plot log acceptance against unit count, repeat
  under the tree-weighted target as control; (2) exact spectral gaps of
  the partition-level tree kernel alone and in mixture on the 193,128
  state 5x5 fiber at eps = 0.2; (3) the precinct-scale freeze on North
  Carolina. The first is the candidate section for the student.

## 2026-10-21. Venues

Decided: SIAM ACDA 2027 (proceedings, January deadline, student
presents) and JSM 2027 (talk only, February abstract). European
alternatives if preferred: SEA 2027 (February, experimental
algorithms, best European fit) or ESA 2027 Track B (April); talk-only
slots at EURO 2027, CMStatistics 2027, EMS 2027. Journal version to
Statistics and Computing, with the scaling experiment and the
precinct-scale run as the extension beyond the conference paper.
