# TODO: paper.tex to submission

Updated 2026-10-22. The run sheet and results of the Forest ReCom kill
test and the scaling experiment are in git history (commit 7e44673 and
earlier); everything they established is now in `paper.tex`.

## Status

`paper.tex` is the rebuilt draft: theory, Chain E, calibration, the
exact sampler against the forest move (kill test passed under the
pre-registered criterion, about 2x on Iowa under pi_w, passenger under
the tree-weighted measure), and the scaling experiment. Compiles clean
at 25 pages (21 of body text) after the 2026-10-22 shortening pass.
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

- [ ] Precinct scale (Section 5.7). Code is ready: `NC_VTD/NC_VTD.json`
      (2692 precincts, built from the MGGG shapefile by gerrychain) and
      `diffusion_model/recom/forest_nc.py`. Run sheet for the local agent,
      from `diffusion_model/`:

      1. Probe (10 min): `python3 recom/forest_nc.py probe`
         Cloud probe at 30 steps: forest 66 ms/step, local 70 ms/step,
         so a 2e5-step chain is about 3.7 h per kernel. Forest acceptance
         at the tree-partition start is 0.57; it is expected to fall at
         equilibrium (Section 5.5).
      2. Eight chains in parallel (one kernel each, about 4 h on the
         11-core machine), each forest chain at the same time as its
         hybrid partner:
         for k in perimeter density; do for s in 42 43; do
           python3 recom/forest_nc.py run forest --seed $s --kernel $k > recom/iowa_hybrid_test_out/forest_nc_forest_${k}_${s}.log 2>&1 &
           python3 recom/forest_nc.py run hybrid --seed $s --kernel $k > recom/iowa_hybrid_test_out/forest_nc_hybrid_${k}_${s}.log 2>&1 &
         done; done; wait
      3. `python3 recom/forest_nc.py verdict > recom/iowa_hybrid_test_out/forest_nc_verdict.txt`
      4. Push the eight .npz files, the logs and the verdict.
      Defaults: K = 13, eps = 0.05, p_local = 0.1, m = 2 (tau from the
      move-size rule, about 0.037 under perimeter and 0.041 under
      density), target pi. If the probe shows the forest move accepting
      below 0.01 at equilibrium, also run `--target tree` for the same
      eight chains; that is the forest move's home ground and the paper
      reports both. If 2e5 steps leave the forest ESS below 50, extend
      with `--n 1000000` for the forest chains only.
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
