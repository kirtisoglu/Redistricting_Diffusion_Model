# TODO: paper.tex to submission

Updated 2026-10-22. The run sheet and results of the Forest ReCom kill
test and the scaling experiment are in git history (commit 7e44673 and
earlier); everything they established is now in `paper.tex`.

## Status

`paper.tex` is the rebuilt draft: theory, Chain E, calibration, the
exact sampler against the forest move (kill test passed under the
pre-registered criterion, about 2x on Iowa under pi_w, passenger under
the tree-weighted measure), and the scaling experiment. Compiles clean.
Two red placeholders: Section 5.7 (precinct scale) and one item in
Appendix B.

## Before the conference version (ACDA, January)

- [ ] Read the draft end to end and decide whether the framing of 4.7.4
      and 5.5 (the tree move as an independence sampler from the
      untempered tree measure; the local move has a dial, the tree move
      has none) is the one to own.
- [ ] Chain A's acceptance rate, still not reported anywhere.
- [ ] Bibliography: delete or fix the uncited and malformed entries
      (`esedog2015threshold`, `chang2018lov`, `trevisan2017lecture`,
      `duchin2021political`); `cannon2020reversible` is cited nowhere
      while RevReCom is discussed.
- [ ] Figures over tables: the Iowa long-run table and the scaling table
      are candidates for a two-panel figure each.
- [ ] Stamp `long_run_5x5_seed42.npz` with alpha, beta and step count.
- [ ] Cut to the conference page limit: Sections 1 to 4.6 shortened,
      5.3 to one paragraph, 5.5 as the closing section.

## Before the journal version (Statistics and Computing)

- [ ] Precinct scale (Section 5.7): restore `archive/diffusion_model/mew`
      for the North Carolina graph, timing probe, then forest move alone
      and hybrid at p_local = 0.1 with tau from the move-size rule,
      2e5 steps, two seeds, perimeter and density targets.
- [ ] Gurobi-free local solve for the active-balance case (one extra
      linear solve and a scalar root), so a local step costs what a
      forest step costs in any language.
- [ ] Per-pair timestep tau = m / (2 |boundary|) in place of a fixed
      tau, and the N = 20 cells of the scaling table to convergence.
- [ ] ESS per step beside ESS per second in every comparison (done in
      the text for Iowa; add the column to the tables).
- [ ] A long Vanilla ReCom run against the pure tree-weighted measure.
- [ ] Release tag and DOI on acceptance.

## Student's section

The scaling experiment (5.5) and its extension to a per-pair timestep
and to N = 20, plus the precinct-scale run. See `NOTES_research.md` for
the urban-rural classification project as his own paper.
