"""
Guided-MEW: Boundary-Cycle + QP-Guided Marked Edge Walk

An improved MEW sampler for redistricting that addresses two structural
inefficiencies in the original MEW:

  (P1) Within-fiber waste — solved by the boundary cycle step
  (P2) Blind marked-edge selection — solved by QP-guided marking

See: picard_quotient.tex, Algorithm 1.
"""
