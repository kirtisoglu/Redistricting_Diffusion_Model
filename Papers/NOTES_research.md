# Research notes

Ideas and follow-up projects outside the current paper. Add to the end;
date each entry. Entries whose content is now in `paper.tex` have been
removed; see git history before 2026-10-22 for them.

## 2026-10-21. Urban vs rural from topology (DeFord's problem)

**The problem.** How much of the full geographic information is
contained in the dual graph? Urban and rural units are easy to tell
apart geometrically (population, area); given only the graph, can the
same classification be recovered, and if not, what additional
information is necessary?

**Relation to the paper.** Converse, not the same. The density kernel
takes population and area as given and asks how to sample with them;
DeFord asks whether they are implied by the topology. The kernel is a
formalization of "additional information", and an exact sampler for
kernel-weighted targets measures what each piece of information
changes in an ensemble. The topology should know something: urban
precincts are small, so the dual graph is dense there, and the
spanning-tree measure is sensitive to that.

**Experiment with our tools** (needs a precinct graph): sample the
uniform-kernel target (topology only) and the density-kernel target on
the same graph; measure the share of cut edges crossing an urban-rural
boundary against the share expected if boundaries ignored density. If
alignment is already there under the uniform kernel, the topology knows
where the cities are; if only under the density kernel, the kernel is
the missing information, quantified.

**Student project (separate paper, student first author).** Predict
urban or rural from graph features alone: degree, local vertex
density, clustering, spectral embedding coordinates, distance to
high-degree cores; labels from census density or an official
urban-area indicator; an interpretable classifier and a map of where
it fails (suburbs, small towns, water boundaries). Closing section: the
sampler experiment above.

## 2026-10-21. Informed marked-edge selection for MEW

The QP-scored marked-edge selection of the earlier `qp_proposal.tex`
is deterministic, so as written it has the same zero-reverse-density
defect as a Cheeger sweep. Replace the argmin with a Boltzmann draw
over marked sets with computable density (sample the k-1 marked edges
sequentially from a softmax of the subtree-population score); the MEW
degeneracy ratio handles the lift and the chain is exactly correctable.
Add the boundary cycle step already proved reversible in
`picard_quotient.tex`. Competitor to beat: BUD, which removes balance
rejection by construction at the same cubic degeneracy cost. Not a
second paper; at most a section of this one if the diffusion move's
case needed a second instance.

## 2026-10-22. Venues

Decided: SIAM ACDA 2027 (proceedings, January deadline, student
presents) and JSM 2027 (talk only, February abstract). European
alternatives: SEA 2027 (February, experimental algorithms, best
European fit) or ESA 2027 Track B (April); talk-only slots at EURO
2027, CMStatistics 2027, EMS 2027. Journal version to Statistics and
Computing, differing from the conference paper by the precinct-scale
run and the per-pair timestep.
