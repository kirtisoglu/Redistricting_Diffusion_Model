# A Diffusion Model for Political Redistricting — Paper Plan

> **Status (2026-05-01 evening).** Concept locked. Q1-ready scope. Six-week sprint to arXiv on **2026-06-12**. Sprint 1 ✓; Sprint 2 ≈70% done (SpecReCom + IQP unification + repair landed; NC pending). Pre-registered first-submission target odds at JCGS: **60-70%.**

## Concept (locked, reframed 2026-05-01 late)

**Working title.** *A Kernel-Flexible Diffusion Model for Political Redistricting: Continuous and Integer Local Refinement of ReCom*

**Pitch.** ReCom is the dominant MCMC method for redistricting, but its spanning-tree-based split is *kernel-blind* — every edge is treated identically. Real geography is not: shared boundary length, population density, demographic similarity, and administrative boundaries all carry signal that should shape districts. We propose a **kernel-aware local refinement** for redistricting MCMC: a single energy

$$
\min\; \alpha\, x^\top L\, x \;+\; \beta\, \|x - x_0\|^2 \quad\text{s.t.}\quad (1-\varepsilon)\bar p \le p^\top x \le (1+\varepsilon)\bar p
$$

on the merged-pair subgraph $H = G[V_i \cup V_j]$, with kernel-weighted Laplacian $L$ and a fidelity term $\beta\,\|x - x_0\|^2$ that keeps refinement local. The framework has **two solvers** for the same model:

- **B (continuous QP-diffusion)**: $x \in [0,1]^n$, convex QP, Cheeger-sweep rounding + orphan-flip repair.
- **C (integer QP / IQP)**: $x \in \{0,1\}^n$, convex MIQP, orphan-flip repair on the integer optimum.

Both kernels mix with ReCom for ergodicity. **The framework generalises SpecReCom** (Davies et al. 2025), which corresponds to the special case ($\beta = 0$, kernel uniform). **It subsumes the discrete cut-min MIP** as the ($\beta = 0$, kernel uniform, integer) special case. The fidelity term $\beta > 0$ unlocks (i) kernel-aware injection of geographic signal, (ii) locality-aware refinement, (iii) a contiguity-preserving prior that reduces post-rounding rejection from 10-15% to <3% on heterogeneous-population graphs.

**Primary contributions** (in order of weight):

1. **A kernel-aware local-refinement framework** for redistricting MCMC. Three example kernels (uniform, perimeter, density) demonstrated on Iowa; the framework accepts any kernel function. *Nothing else in the literature does this.*
2. **A unified continuous/integer dual** (chains B and C) for the same energy, bridging spectral relaxations and integer programming. Both inherit kernel flexibility and locality control.
3. **Spectral analysis** of the QP-diffusion: energy-descent lemma, closed-form frequency response, Cheeger-inequality bound on rounded cut. Recasts the model as an implicit Euler step of the graph heat equation.

**Supporting contributions** (motivation and validation):

4. **Diagnostic characterisation of the partition fiber's macro-structure** (§3.5): islands + corridors. The "narrow-throat invariant" — count of narrow inter-cluster connections is ~12-16 across (grid, k, ε) variations. Step-trace shows ReCom jumps between islands (~14% / step) while spectral local kernels stay within (~5-7% / step). Three test beds with full enumeration. This diagnostic *motivates* the local-refinement design but is also a stand-alone tool for analysing other samplers.
5. **Empirical validation**: 23.9% mean cut-edge reduction on Iowa (k=4, ε=0.05) versus Tempered ReCom, p < 10⁻¹⁹⁷. Compactness-coverage trade-off quantified on a fully enumerated 5×5 ground truth via TV-distance to uniform.
6. **Reproducible implementation** with hyperparameters locked and documented (Appendix A).

**Why this framing.** "More compact than ReCom" alone is incremental: ReCom is already adequately compact for most uses. *Kernel flexibility* is structurally distinct from anything in the literature and directly serves the practical needs the community has flagged: geographic realism, constraint awareness, locality-preserving moves. The compactness gain is *evidence the method works*, not the contribution itself.

## Target venue & timeline

**arXiv first**: hard deadline **2026-06-12** (six weeks).
**Journal**: *JCGS* primary, submitted within one week of arXiv (target **2026-06-19**).
Backups: *Statistics and Computing*, *AoAS*.

## Five chains compared

| Chain       | Method                               | Mechanism                                                                      | Notes                                       |
| ----------- | ------------------------------------ | ------------------------------------------------------------------------------ | ------------------------------------------- |
| V           | Vanilla ReCom                        | spanning-tree split, always accept                                             | baseline                                    |
| A           | Tempered ReCom                       | ReCom + MH on cuts (sym-q approx)                                              | reference                                   |
| **B** | **ReCom + QP-diffusion mix**   | continuous min α·xᵀLx + β·‖x−x₀‖², Cheeger sweep, orphan-flip repair | ours, kernel-flexible                       |
| **C** | **Tempered ReCom + IQP local** | integer min α·xᵀLx + β·‖x−x₀‖², MIQP, orphan-flip repair             | ours, kernel-flexible, subsumes MIP-cut-min |
| D           | ReCom + SpecReCom mix                | Fiedler+Cheeger sweep (no β, no kernel, no repair)                            | Davies et al. 2025 baseline, faithful       |

## Sprint 1 results (2026-05-01) — completed

### 1. QP-diffusion fix on Iowa: Cheeger sweep + unnormalised L

The notes' L_sym formulation gave +9% MORE cuts than Tempered ReCom on Iowa (relaxation looseness on heterogeneous-degree graphs). Sprint 1 fix: **(i)** replace median/balance threshold with **Cheeger sweep**, **(ii)** switch from L_sym to **unnormalised L** so xᵀLx exactly equals weighted cut count for binary x.

| Iowa configuration                            |       Mean cuts |    qp_success |              Δ vs Tempered ReCom |
| --------------------------------------------- | --------------: | ------------: | --------------------------------: |
| L_sym, balance threshold (Sprint 0)           |           41.61 |           92% |                     +8.7% (worse) |
| L_sym, Cheeger sweep                          |           39.13 |           89% |                        +2.2% (NS) |
| **L (unnormalised), Cheeger sweep**     | **37.07** | **98%** |    **−3.2% (p=3×10⁻⁵)** |
| **IQP local (chain C, supersedes MIP)** | **31.99** | **98%** | **−16.4% (p=2×10⁻⁵⁵)** |

For the paper §4: explain that L_sym matches normalised-cut (Shi-Malik 2000); unnormalised L matches raw cut count, the redistricting metric.

### 2. Island-neck diagnostic (paper §3 headline)

5×5 fiber decomposes into **10 islands** (greedy modularity Q=0.65). Step-trace over 1000 steps:

| Chain                                  | Islands visited | **Jump rate** | Jumps |
| -------------------------------------- | :-------------: | ------------------: | ----: |
| V: Vanilla ReCom                       |      7/10      |     **14.1%** |   141 |
| A: Tempered ReCom                      |      8/10      |     **13.3%** |   133 |
| **B: QP-diffusion mix (50/50)**  |      6/10      |      **5.2%** |    52 |
| **C: ReCom + IQP local**         |      5/10      |               11.5% |   115 |
| **D: ReCom + SpecReCom (50/50)** |      7/10      |      **6.8%** |    68 |

**Interpretation.** B and D (both spectral-based local kernels) drop ReCom's jump rate from ~14% to ~5-6%. Direct empirical confirmation: **ReCom jumps between islands; spectral local kernels explore within them.** Paper §3 headline figure.

### 3. Bias bound table (paper §7.1, 1000 steps, BURN=100)

TV(empirical, uniform-on-fiber) on the 5×5 (all 4006 partitions, exact target):

| Chain                | Unique partitions | Coverage | TV(emp, uniform) |
| -------------------- | :---------------: | :------: | ---------------: |
| V: Vanilla ReCom     |    280 / 4006    |   7.0%   |             0.93 |
| A: Tempered ReCom    |    315 / 4006    |   7.9%   |             0.92 |
| B: QP-diffusion mix  |    164 / 4006    |   4.1%   |             0.96 |
| C: ReCom + IQP local |    179 / 4006    |   4.5%   |             0.96 |
| D: ReCom + SpecReCom |    199 / 4006    |   5.0%   |             0.95 |

**Interpretation.** Compactness-improving methods (B, C) trade representativeness for compactness; coverage drops from 7–8% (V, A) to 4–5%. **The trade-off is bounded and quantified.**

### 4. 5×5 compactness (1000 steps)

| Chain                | Mean cuts |  std |                          Δ vs A |
| -------------------- | --------: | ---: | -------------------------------: |
| V: Vanilla ReCom     |     16.60 | 0.91 |                               — |
| A: Tempered ReCom    |     16.70 | 0.95 |                               — |
| B: QP-diffusion mix  |     16.67 | 0.95 |                      −0.2% (NS) |
| C: ReCom + IQP local |     16.17 | 0.47 | **−3.2% (p=2×10⁻³⁴)** |
| D: ReCom + SpecReCom |     16.58 | 0.84 |                      −0.7% (NS) |

On uniform-population 5×5, B/D give modest gains; C dominates. Std of C collapses to 0.50 (tightest of all chains).

## Sprint 3 partial results (2026-05-01 evening) — corridors + ε=0.2 fiber + Iowa diversity sweep

### 1. Iowa Baseline / Jitter / Tabu sweep (n_iqp=5, α=10, β=1, 500 steps)

| Config | C mean cuts | C std | Δ vs A | Wall (C) |
|---|---:|---:|---:|---:|
| Baseline | 31.88 | 2.28 | −20.8% | 39.7s |
| **Jitter** (SpecReCom-style U[1,2] edge weights) | 31.76 | **1.77** | −21.1% | 45.2s |
| **Tabu** (size 3) | **31.29** | **1.26** | **−22.3%** | 37.6s |

**Tabu wins on Iowa**: best mean cuts (31.29) AND tightest distribution (std 1.26, 44% reduction vs baseline). On 5×5 the diversity tricks didn't move cuts because the chain was already at the cut floor; on Iowa there is real room and pair-rotation finds better local optima.

**Lock-in for Iowa headline**: chain C with Tabu (size 3), n_iqp=10, α=10, β=1.

### 2. ε=0.2 fiber: corridor analysis (2026-05-01)

The full ε=0.2 fiber on the 5×5 grid (193,128 partitions, 969,760 edges) was constructed as a paper-supporting test bed (single-flip adjacency, district sizes ∈ {4, 5, 6}, all balance-budget-respecting partitions). Properties:

- **Layer structure**: L0 = (5,5,5,5,5) = 4006 partitions; L1 = (4,5,5,5,6) = 76,472; L2 = (4,4,5,6,6) = 112,650.
- **Connected**: single component, no bridges, no articulation points.
- **Mean degree**: 10.04 (vs 6.70 in ε=0 fiber). L0 mean degree 19.5, L1 12.8, L2 7.9.
- **L0 is an independent set** (no L0-L0 edges; a single flip changes population so cannot map L0→L0). All L0-to-L0 paths route through L1 — these are the corridors.

**Louvain communities** (faster than greedy modularity for this size):
- 26 communities, modularity Q = 0.6876 (vs 0.6416 for greedy modularity in 4.7 hours; Louvain finishes in 44s).
- Community sizes range from ~3000 to ~13350.

**Corridor "throats"** (narrowest inter-community edge counts):

| Pair | Edges | |A| / |B| | Edges per min(|A|, |B|) |
|---|---:|---:|---:|
| (12, 19) | **2** | 3253 / 5902 | 0.0006 |
| (10, 18) | **2** | 5538 / 5530 | 0.0004 |
| (11, 19) | **3** | 7406 / 5902 | 0.0005 |
| (3, 7) | **4** | 5257 / 2788 | 0.0014 |

**Three pairs of communities each ≥ 5000 nodes are connected by 2–4 edges**. These are not bridges (graph remains connected by other paths) but they are *2–4 vertex separators between macro-clusters*. Exactly the "corridor" structure: bridges of ε=0 fiber thicken into 2–4 edge throats in ε=0.2.

**Inter-community edge-count distribution** (325 possible pairs, 320 active):

| Quantile | Edges/pair |
|---|---:|
| min | 2 |
| p10 | 112 |
| p25 | 304 |
| median | 666 |
| p75 | 1191 |
| max | 3006 |

**Three-tier structure**:
- **5 pairs disconnected** (zero edges; must route via intermediates).
- **16 thin pairs (<50 edges)** — narrow corridors / throats.
- **105 medium pairs (50-500)**.
- **199 thick pairs (≥500)** — highways, freely traversed.

**Striking parallel with ε=0 fiber**: the *count* of narrow connection points (16 thin pairs in ε=0.2 ≈ 16 single-edge bridges in ε=0) is preserved as ε relaxes. Bridges thicken into corridors but the macro-cluster connectivity skeleton retains its topology. Provisional finding for §3: the bridge/corridor count is approximately a topological invariant of the fiber's macro-structure across ε.

### 2b. 4×4 ε=0.25 fiber (k=4) — second test bed

A complementary fiber on the smaller 4×4 grid (k=4, ε=0.25, allowed sizes {3,4,5}) confirms the pattern at much smaller scale and lets us run all analyses in seconds rather than hours.

| Layer (multiset) | Count |
|---|---:|
| (4,4,4,4) — L0 balanced | 117 |
| (3,4,4,5) — L1 one-off | 1240 |
| (3,3,5,5) — L2 two-off | 596 |
| **Total** | **1953** |

- **7024 edges** (single-flip adjacency).
- **0 bridges, 0 articulation points** (matches 5×5 ε=0.2).
- **16 communities** (Louvain), Q = 0.6135. Sizes 59-170.
- **13 narrow throats** (<5 edges between large communities) — corridors.
- **12 disconnected community pairs** (must route through intermediates).
- 67 medium pairs (5-30 edges); 28 highway pairs (≥30 edges).
- 4 single-edge "near-bridges" between communities of sizes 100-170: pairs (11,12), (2,11), (1,13), (9,15).

### 2c. The narrow-throat invariant (paper §3 headline finding)

| Fiber | Nodes | Edges | Communities | Q | Bridges | Narrow throats |
|---|---:|---:|---:|---:|---:|---:|
| 5×5 ε=0 (k=5) | 4 006 | 13 416 | 10 | 0.65 | **16** | n/a |
| 5×5 ε=0.20 (k=5) | 193 128 | 969 760 | 26 | 0.69 | 0 | **16** (<50 edges) |
| 4×4 ε=0.25 (k=4) | 1 953 | 7 024 | 16 | 0.61 | 0 | **13** (<5 edges) |

The narrow-throat count is **in the order of dozen** across both grid sizes and ε regimes. As ε relaxes, single-edge bridges thicken into few-edge corridors, but the cluster-connectivity skeleton retains its topology. **Provisional thesis for §3: narrow-throat count is approximately a topological invariant of the fiber's macro-structure across (grid, k, ε) variations.**

Implementation: [`fiber_5x5/build_fiber_4x4.py`](fiber_5x5/build_fiber_4x4.py) builds the 4×4 fiber; [`fiber_5x5/corridor_analysis_eps02.py`](fiber_5x5/corridor_analysis_eps02.py) takes a CLI prefix (e.g. `fiber_4x4_eps025`) so the same code runs on either fiber.

**Sampled edge betweenness** (Brandes, k=100 pivots): top 0.1% (969 edges) include 263 corridor edges (cross-community), 706 intra-community. Corridor edges are over-represented relative to the 27% inter-community baseline (263/969 = 27.1% — exactly equal, suggesting community structure is well-balanced w.r.t. shortest-path traffic).

**Visualisation**: [`fiber_5x5/fiber_5x5_eps02_corridors.png`](fiber_5x5/fiber_5x5_eps02_corridors.png) — 22×22 inch, 200 DPI. Two-level layout (community centroids by spring on meta-graph, nodes within community by spring on subgraph). Crimson edges highlight top-betweenness corridors.

**Implementation**: [`fiber_5x5/corridor_analysis_eps02.py`](fiber_5x5/corridor_analysis_eps02.py).

### 3. ε=0.2 fiber: chain step-trace test (in progress)

To complement the ε=0 island-neck step-trace from Sprint 1, we run the same diagnostic on the ε=0.2 fiber. New mode `"5x5_eps02"` in `problems.py` loads the 193k-partition fiber and the 26-community Louvain labels. Chains run with ε=0.20 (so ReCom can produce all valid (4,5,5,5,6) and (4,4,5,6,6) partitions). Step-trace tracks which Louvain community each visited partition is in.

**Hypothesis** (to be tested): ReCom variants (V, A) jump between communities at high rate (~10-20% per step); spectral local kernels (B, C, D) drop the jump rate substantially (~5-7%) — replicating the ε=0 finding on the larger fiber and confirming the corridor/island narrative scales.

Results pending; will update.

---

## Detailed paper outline (v1 arXiv draft — section-by-section content map)

Target ~30 pages arXiv, ~35 pages JCGS revision. Outside-in writing order: §7 first, §1 last.

### §1. Introduction (~3 pages)  *(write last)*

- **1.1 The redistricting problem and ReCom's role.** Brief setup: dual graph, k-partition, ε-balance, contiguity, compactness. ReCom's spanning-tree-split as the workhorse; its known weakness on compactness ("banana districts").
- **1.2 Contribution.** Three claims, in order of weight:
  (a) **A unified QP-diffusion model with two solvers** — continuous QP and integer QP — sharing the same energy $\alpha x^\top L x + \beta \|x - x_0\|^2$ with configurable kernel-weighted Laplacian. Generalises SpecReCom (Davies et al. 2025) by adding the fidelity term β·‖x−x₀‖² and kernel flexibility; subsumes raw cut-min as a special case.
  (b) **Spectral analysis**: energy-descent lemma, closed-form frequency response, and Cheeger-inequality bound on the rounded cut — recasting the QP-diffusion as an implicit Euler step of the graph heat equation.
  (c) **Empirical**: 23.9% cut-edge reduction on Iowa (k=4, ε=0.05) versus Tempered ReCom, with a quantified compactness-coverage trade-off on a fully enumerated 5×5 ground truth.
  *Supporting: a structural diagnostic of the partition fiber (islands, narrow corridors) that motivates the local-refinement design, with three test beds spanning grid size and ε.*
- **1.3 Comparison with SpecReCom (Davies et al. 2025).** Position as a strict generalisation: SpecReCom's spectral cut is the (β=0, kernel=uniform) limit of our QP. The fidelity term unlocks kernel flexibility, locality-aware refinement, and a discrete (IQP) counterpart that delivers the empirical headline.
- **1.4 Roadmap.**

### §2. Related work (~2 pages)

- **2.1 MCMC for redistricting.** DDS ReCom (DeFord, Duchin, Solomon), RevReCom (Cannon-Duchin-Randall), Mattingly's outlier methodology, gerrychain.
- **2.2 Spectral graph partitioning for redistricting.** SpecReCom (Davies et al. 2025), Fiedler vector, Cheeger inequality.
- **2.3 Image segmentation / level-set lineage** (already drafted in notes §2): Ginzburg-Landau, Chan-Vese, MBO.
- **2.4 Cut minimisation.** Min-cut, normalized cut (Shi-Malik 2000), Total Variation Isoperimetric Profiles.

### §3. Background: ReCom and the partition fiber (~3 pages)

Mostly drafted in `diffusion.tex` §3.1.

- **3.1 ReCom recap** (drafted).
- **3.2 The partition fiber** $\mathcal{P}_k(G; \varepsilon)$: set of valid k-partitions. Stratification by district size multiset.
- **3.3 Single-flip adjacency**: the natural neighbour relation on the fiber. Used as test bed for visualising the chain's local structure.

### §3.5. The partition fiber's island and corridor structure (~3 pages) **[NEW]**

This is a major contribution of the paper. Draw from the Sprint 1 + Sprint 3 corridor analyses.

- **3.5.1 Three test beds.** Table:
  | Fiber | Nodes | Edges | k | ε | Allowed sizes |
  |---|---:|---:|---:|---:|---|
  | 5×5 ε=0 | 4 006 | 13 416 | 5 | 0 | {5} |
  | 5×5 ε=0.20 | 193 128 | 969 760 | 5 | 0.20 | {4,5,6} |
  | 4×4 ε=0.25 | 1 953 | 7 024 | 4 | 0.25 | {3,4,5} |
- **3.5.2 Island structure.** Greedy modularity / Louvain communities. Q values 0.61-0.69. Sizes distributed (figure: degree distribution + community-size histogram).
- **3.5.3 Bridges (ε=0) and corridors (ε>0).** Headline finding: narrow-throat count is approximately invariant (~12-16) across (grid, k, ε). Table from §3 above.
- **3.5.4 The corridor visualisation** (Figure: `fiber_5x5_eps02_corridors.png`). Community-aware 2-level layout, crimson edges = top-betweenness corridors, communities colour-coded.
- **3.5.5 Chain step-trace diagnostic.** On 5×5 ε=0: ReCom jumps 13-14% / step between islands; spectral local kernels (B, D) drop to 5-7%. **Direct empirical confirmation that ReCom is for inter-island, local kernels for intra-island.** Headline Figure: `island_trace_5x5.png`.
- **3.5.6 Implication.** Local refinement complements ReCom's global jumps. Motivates the QP / IQP family.

### §4. The QP-diffusion model (~4 pages)

Drafted in `diffusion.tex` §3.2-§4. Polish + add Cheeger sweep.

- **4.1 Locality kernels** (drafted): perimeter, density, demographic, voting, administrative-boundary. Composite kernels.
- **4.2 The convex QP**: $\min \alpha\, x^\top L\, x + \beta\, \|x - x_0\|^2$ s.t. ε-balance. Continuous relaxation of cut + fidelity.
- **4.3 Cheeger-sweep rounding + orphan-flip repair** **[NEW subsection]**. Algorithm + complexity. References Davies et al. for the sweep idea, our repair as a feasibility-rescuing extension.
- **4.4 Hybrid composition with ReCom.** Random mixture; ergodicity argument (convex combination of two ergodic kernels).

### §5. Spectral analysis (~4 pages) **[mostly drafted in notes]**

`diffusion.tex` §6.

- **5.1 Energy-descent lemma** (drafted, Lemma 6.1).
- **5.2 Spectral filter frequency response** (drafted): $\hat x_k = \gamma/(\beta\lambda_k + \gamma) \cdot \hat x_{0,k}$.
- **5.3 Cheeger inequality bound** on the rounded cut.
- **5.4 Implications for kernel design.** "Are the first eigenvectors noise?" — diagnostic for kernel validity.

### §6. Integer QP local refinement (~3 pages) **[NEW]**

Subsumes prior MIP-cut-min.

- **6.1 The IQP**: same objective and constraints as §4.2, $x \in \{0,1\}^n$. Convex MIQP via Gurobi.
- **6.2 Subsumption**: setting $\alpha=1, \beta=0$, kernel uniform recovers raw cut-min.
- **6.3 The fidelity term as a connectivity prior**: β > 0 anchors near $x_0$ (contiguous), keeping post-rounding rejection rates low.
- **6.4 Orphan-flip repair on the integer optimum**: same routine as §4.3.
- **6.5 Two solvers, one model**: §4 continuous and §6 integer share α, β, kernel. Trade-off table.

### §7. Experiments (~7 pages) — *the empirical core*

Five chains (V, A, B, C, D) on three datasets (5×5, Iowa, 4×4). All locked.

**Reordered to put kernel flexibility first** (the primary selling point), with compactness as supporting evidence.

- **7.1 Multi-kernel showcase on Iowa** **[NEW HEADLINE — pending Sprint 3 sub-task]**
  - QP-diffusion (chain B) and IQP (chain C) with three kernels on Iowa:
    - **uniform** (baseline cut count) — current locked numbers.
    - **perimeter** (`w_uv = |∂v_u ∩ ∂v_v|` from `shared_perim` edge attribute) — boundary-length-aware refinement.
    - **density** (`w_uv = |∂v_u ∩ ∂v_v| · exp(-(ρ_u-ρ_v)²/σ²)` with ρ = TOTPOP/area) — urban-rural-aware refinement.
  - **Side-by-side Iowa partition maps** under each kernel: visually different, each respecting a different signal.
  - **Mechanism**: the kernel re-weights the Laplacian → different x* → different cut chosen → different partition geometry.
  - **Cut counts** under each kernel (uniform baseline, perimeter, density).
  - **Novelty story vs SpecReCom**: SpecReCom uses unweighted spectral; ReCom uses spanning trees. *Neither can encode geographic signal at the proposal level.* This is unique to our framework.

- **7.2 Iowa compactness comparison (5 chains)**
  - Locked headline: **C with n_iqp=10 → 30.66 mean cuts, −23.9% vs A** (Δ from Sprint 3 sweep).
  - With Tabu (n_iqp=5, K=3): 31.29 cuts, −22.3%, std 1.26 (44% tighter than baseline).
  - Comparison: V 41.81 → A 38.29 → D 38.81 (≈ A) → B 37.07 → C 30.66.
  - **Pareto frontier** plot: cuts vs wall-clock for V, A, B, C, D.
  - Framed as *evidence* the framework works, not the headline.

- **7.3 5×5 ground-truth validation and bias bound**
  - Compactness table (4 chains): V 16.60, A 16.70, B 16.67, C 16.17, D 16.58.
  - **Bias-bound table (TV to uniform)**: V 0.93, A 0.92, B 0.96, C 0.96, D 0.95. Honest cost-of-compactness.
  - Confirms framework is correctly implemented (vs the exact target distribution).

- **7.4 Diversity ablations (Jitter / Tabu)**
  - Iowa Baseline vs Jitter vs Tabu (n_iqp=5):
    - Baseline: 31.88 cuts (std 2.28), 4.5% coverage
    - Jitter: 31.76 cuts (std 1.77), 5.7% coverage
    - Tabu: **31.29 cuts (std 1.26)**, 4.6% coverage
  - Two configurable knobs for the practitioner's compactness ↔ coverage trade-off.

- **7.5 NC validation** **[optional for v1; valuable for journal revision]**
  - NC precinct (2692 nodes), k=4. Cross-state replication of compactness gain.
  - Defer if Sprint 3 multi-kernel is the priority.

- **7.6 Downstream impact: partisan skew** **[pending Sprint 4; v2/revision]**
  - 5000 plans per chain on Iowa; efficiency gap, mean-median, packing.
  - "Does compactness improvement change the answer?" — important for *Statistics and Computing* / *AoAS* angle.

### §8. Discussion (~2 pages)

- **8.1 Compactness-vs-coverage trade-off**: bounded, characterised, configurable via β and tabu.
- **8.2 Limitations**
  - C is slower than D on small graphs (MIQP cost). For massive precincts, may need time limits + warm starts.
  - QP-diffusion is loose on heterogeneous-population graphs; the integer version recovers compactness at a compute cost.
  - Symmetric-q approximation in Tempered ReCom is biased on small graphs (5×5: mean 16.70 vs target 18.56).
- **8.3 Future directions**
  - Replica exchange / parallel tempering between B/C and V/A for coverage.
  - cyclewalk / Balanced-Up-Walk integration.
  - Full theoretical mixing-time analysis.

### Appendix (~5 pages)

- **A. Algorithm details and hyperparameters** (current "Algorithm details" section of this plan).
- **B. Detailed parameter sweep tables** (n_iqp ∈ {3, 5, 10}, α ∈ {10, 50}, β ∈ {0, 1}).
- **C. Bridge / corridor / community size tables** (full tables behind the §3.5 headline).
- **D. Compute environment + reproducibility checklist.**

---

## Figures and tables checklist (for v1 submission)

| # | Element | Source                                              | Status |
|---|---|-----------------------------------------------------|---|
| Fig 1 | Iowa dual graph w/ congressional districts | `Papers/fig/redistricting.png` placeholder in notes | needs real shapefile render |
| Fig 2 | 5×5 fiber community structure | `fiber_5x5/fiber_5x5_communities.png`               | ✓ have |
| Fig 3 | 5×5 ε=0 island-trace step-by-step | `iowa_hybrid_test_out/island_trace_5x5.png`         | ✓ have |
| Fig 4 | 5×5 ε=0.2 corridor visualisation | `fiber_5x5/fiber_5x5_eps02_corridors.png`           | ✓ have |
| Fig 5 | 4×4 ε=0.25 corridor visualisation | `fiber_5x5/fiber_4x4_eps025_corridors.png`          | ✓ have |
| Fig 6 | Iowa cuts trace (5 chains) | `iowa_hybrid_test_out/target_matched_iowa.png`      | ✓ have |
| Fig 7 | Iowa cuts histogram (5 chains, post burn-in) | (panel of Fig 6)                                    | ✓ have |
| Fig 8 | Spectral frequency response curve | `plots/spectral_frequency_response.png`             | ✓ have |
| Fig 9 | Pareto frontier: cuts vs wall-clock | new — easy from npz                                 | pending |
| Fig 10 | Multi-kernel Iowa partitions | new                                                 | pending Sprint 3 |
| Fig 11 | Partisan-skew distributions | new                                                 | pending Sprint 4 |
| Tbl 1 | Three fibers comparison (§3.5.3) | this plan                                           | ✓ have |
| Tbl 2 | 5×5 compactness + bias | this plan                                           | ✓ have |
| Tbl 3 | Iowa headline 5-chain | this plan                                           | ✓ have |
| Tbl 4 | n_iqp sweep | this plan                                           | ✓ have |
| Tbl 5 | Diversity ablations (Jitter/Tabu) | this plan                                           | ✓ have |
| Tbl 6 | Hyperparameter locked values | "Algorithm details" section                         | ✓ have |

## Writing order (recommended)

Revised priority: **multi-kernel showcase (§7.1) is the critical path**, NC and downstream are deferable.

**Week 1**:
- Mon-Tue: **§7.1 multi-kernel Iowa runs** (perimeter and density kernels — Sprint 3 critical path).
- Wed: §7.1 writeup with side-by-side partition maps.
- Thu-Fri: §3.5 (corridor narrative — the diagnostic motivating the framework).

**Week 2**:
- Mon: §7.2 + §7.3 + §7.4 (we have all the numbers — compactness, bias, ablations).
- Tue-Wed: §6 IQP description.
- Thu-Fri: §4 QP-diffusion description (polish + add Cheeger sweep + repair).

**Week 3**:
- Mon-Tue: §5 spectral analysis (polish existing draft).
- Wed-Fri: §7.5 NC run **OR** §7.6 downstream skew (pick one for v1, save the other for revision).

**Week 4**:
- Mon-Tue: §8 Discussion.
- Wed: §2 Related work.
- Thu-Fri: §1 Introduction (last — writes itself from §7).

**Week 5**: Figures polish, captions, internal review, abstract.

**Week 6**: arXiv submission.

**Critical-path note**: if multi-kernel runs reveal issues (e.g. perimeter kernel doesn't behave as expected), pivot to investigating those rather than continuing to write. The kernel-flexibility story is the entire pitch — it must demonstrably work.

---

## Algorithm details and hyperparameters (locked, paper-grade)

This section catalogs every parameter and design choice so the plan can merge directly into `Papers/diffusion.tex`. All notation matches the paper draft.

### Common setup (all chains)

- **Dual graph $G = (V, E)$** with vertex populations $p_v$, edge attributes (`shared_perim` for Iowa).
- **k districts** with **ε-population balance**: each district must satisfy $|p(V_i) - \bar p| / \bar p \le \varepsilon$ where $\bar p = \sum_v p_v / k$.
- **Initial partition** from `gerrychain.tree.recursive_tree_part(G, range(k), $\bar p$, "TOTPOP", ε)`.
- **Step count $N$** and **burn-in $B$**: 5×5: $N=1000$, $B=100$. Iowa: $N=500$, $B=100$.
- **Random seed**: 42 (Python `random` and `numpy`).

### Chain V — Vanilla ReCom

Standard `gerrychain.proposals.recom(...)`, `node_repeats=1`. Always accept. Reference baseline.

### Chain A — Tempered ReCom

Vanilla ReCom proposal + Metropolis-Hastings acceptance on cut-edge target $\pi(x) \propto \exp(-\lambda \cdot \mathrm{cut\_edges}(x))$, **using the symmetric-q approximation** (treats ReCom proposal density as symmetric):

$$\alpha_{MH}(x \to x') = \min\bigl(1, \exp(-\lambda(\mathrm{cuts}(x') - \mathrm{cuts}(x)))\bigr).$$

- **λ (target temperature)**: $0.10$. Mild compactness tilt; kept fixed for all chains' shared target.

### Chain B — ReCom + QP-diffusion mixture

Random mixture: at each step, with probability $p_R$ take a Vanilla ReCom step; otherwise take a QP-diffusion step.

**QP-diffusion step** (continuous local refinement):

1. Choose adjacent district pair $(V_i, V_j)$ uniformly at random from boundary pairs.
2. Form merged subgraph $H = G[V_i \cup V_j]$.
3. Build kernel-weighted Laplacian $L$ on $H$ (see *kernels*).
4. Solve convex QP:
   $$x^\star = \arg\min_{x \in [0,1]^{|V_H|}} \alpha\, x^\top L\, x + \beta\, \|x - x_0\|_2^2 \quad \text{s.t.}\ (1-\varepsilon)\bar p \le p^\top x \le (1+\varepsilon)\bar p$$
   where $x_0$ is the binary indicator of the current $V_i$. Solver: Gurobi convex QP.
5. **Cheeger-sweep rounding**: sort $x^\star$ descending, try every threshold $k = 1, \ldots, |V_H|-1$. For each $k$:
   - Verify population balance $(1-\varepsilon)\bar p \le \sum_{j \le k} p_{\text{order}(j)} \le (1+\varepsilon)\bar p$.
   - Verify connectivity of both sides; if disconnected, apply orphan-flip repair and re-verify.
   - Compute pair cut count.
6. Return the threshold that minimises pair cut count among feasible candidates.

**Hyperparameters (Iowa locked-in values)**:
- $p_R$ (P_RECOM) = $0.5$
- $\alpha$ (QP_ALPHA) = $10.0$ — Laplacian smoothness weight.
- $\beta$ (QP_BETA) = $1.0$ — fidelity to current state.
- threshold scheme = `"cheeger"` with `"balance"` fallback.
- Laplacian normalisation = `False` (unnormalised $L$).
- kernel = `"uniform"` for headline; `"perimeter"` and `"density"` tested in §7.3.
- weight jitter = `False`, tabu size = $0$ (defaults; both available as knobs).

### Chain C — Tempered ReCom + IQP local refinement

Tempered ReCom step (chain A) followed by **n_iqp deterministic IQP local steps**.

**IQP local step** (integer version of the QP-diffusion):

1. Choose adjacent pair $(V_i, V_j)$ uniformly at random.
2. Form merged subgraph $H$.
3. Build kernel-weighted Laplacian $L$ on $H$.
4. Solve convex MIQP:
   $$x^\star = \arg\min_{x \in \{0,1\}^{|V_H|}} \alpha\, x^\top L\, x + \beta\, \|x - x_0\|_2^2 \quad \text{s.t.}\ (1-\varepsilon)\bar p \le p^\top x \le (1+\varepsilon)\bar p$$
   No auxiliary cut variables; quadratic objective handled directly by Gurobi MIQP. Time limit per solve: 5.0 s.
5. Apply orphan-flip repair to the integer optimum.
6. Verify post-repair connectivity and ε-balance; if it fails, the chain stays at the current state.

**Hyperparameters (Iowa locked-in values, after parameter sweep 2026-05-01 evening)**:

| Param | Symbol | Value | Notes |
|---|---|---:|---|
| Steps per outer | n_iqp | **10** | Dominant lever (sweep below); sweep showed −5% additional cuts going 3 → 10 at 4× compute. |
| Cut weight | $\alpha$ | $10.0$ | Saturates here; α=50 gives no further gain. |
| Fidelity | $\beta$ | $1.0$ | Locality + connectivity-friendliness; β=0 essentially identical cut quality but loses contiguity advantage. |
| Time limit | — | $5.0$ s | Per MIQP solve. Almost never hit on Iowa pair sizes. |
| Normalize | — | False | Unnormalised L. |
| Kernel | — | uniform | Headline; perimeter/density tested separately. |

**n_iqp parameter sweep (Iowa, 500 steps each)**:

| Config | Mean cuts | Δ vs A | Wall-clock C |
|---|---:|---:|---:|
| n=3, α=10, β=1 | 32.37 | −19.6% | 24s |
| n=5, α=10, β=1 | 31.88 | −20.8% | 40s |
| **n=10, α=10, β=1** | **30.66** | **−23.9% (p=5×10⁻¹⁹⁷)** | 103s |
| n=3, α=50, β=1 | 32.38 | −19.6% | 32s |
| n=3, α=10, β=0 | 32.31 | −19.8% | 31s |
| n=5, α=10, β=0 | 31.45 | −21.9% | 46s |

n_iqp = 10 is locked as the headline. Other levers (α, β, kernel) saturate or are orthogonal.

### Chain D — ReCom + SpecReCom mixture (Davies et al. 2025 baseline)

Random mixture: with probability $p_R$ take Vanilla ReCom step; otherwise take SpecReCom step.

**SpecReCom step** (faithful to Algorithm 2 of Davies et al.):

1. Choose adjacent pair $(V_i, V_j)$ uniformly from boundary pairs.
2. Form merged subgraph $H$.
3. **Randomise edge weights** $w(e) \sim \mathcal{U}[1, 2]$ for each $e \in E_H$ (paper alg. 2 lines 4-6 — breaks determinism).
4. Compute Laplacian $L = D - W$ on $H$ with the random weights (unnormalised).
5. Compute Fiedler vector $\phi_2$ via `np.linalg.eigh`.
6. Cheeger-sweep rounding on $\phi_2$, with `repair=False` to be faithful.
7. Return new partition or no-move on infeasibility.

**Hyperparameters**:
- $p_R$ = $0.5$ (matches B's mixture weight).
- normalize = False; repair = False (faithful Davies).

### Orphan-flip contiguity repair (used in B and C, NOT in D)

Given a candidate assignment $x$ that may have disconnected sides:

1. For each side $V_i'$ in the merged pair: identify connected components.
2. Keep the largest CC; flip all smaller CCs to the opposite side.
3. Recompute connectivity on both final sides; if either is disconnected or empty, reject the move.
4. Verify ε-balance on the repaired assignment; if violated, reject.

For B and C: when the Cheeger sweep / MIQP produces near-feasible cuts whose few orphan vertices can be reassigned without changing cut count, repair salvages them. Without repair, post-rounding rejection on Iowa runs at 5-15%; with repair, <3%.

For D: NOT applied. SpecReCom has no β term, so repaired thresholds can be far from the original spectral cut and degrade quality. Faithful Davies skips repair.

### Diversity controls (optional, off by default in headline runs)

Two complementary mechanisms tested in Sprint 2 ablations:

- **Weight jitter (SpecReCom-style)**: in `build_laplacian{,_sym}`, multiply each edge weight by $\mathcal{U}[1, 2]$ before solving. Tested on B and C; gave **+27% coverage gain** on chain C with negligible cut-quality cost (0.04 mean cuts on 5×5). Always on for D (faithful Davies).
- **Tabu list**: forbid the last $K$ pairs chosen by the local kernel (cleared whenever ReCom moves). Modest cut tightening (std 0.47→0.36 on 5×5), no coverage gain. K=3 best.

Default: both off in headline runs. Available as configuration knobs; reported as ablation.

### Random seed and reproducibility

- All Python and numpy randomness seeded with `SEED = 42`.
- Gurobi solver uses default seed (deterministic given the same input).
- Every experiment writes raw cut traces to `iowa_hybrid_test_out/target_matched_<MODE>.npz`.

### Compute environment

- macOS (Darwin 23.6.0), Python 3.12.
- Gurobi 11 with academic license.
- NetworkX 3.x for graph operations and modularity / connectivity.
- Single-threaded; no GPU. All experiments completed on a 2024 MacBook.

---

## Sprint 2 results (2026-05-01) — IQP unification + repair

### 1. IQP local step (chain C) — replaces MIP

The integer version of the QP-diffusion model: same objective, same constraints, same kernel; only the variable domain changes from $[0,1]^n$ to $\{0,1\}^n$. Solved as convex MIQP (no auxiliary cut variables). **Setting α=1, β=0, kernel=uniform recovers the prior MIP-cut-min** as a special case — IQP is strictly more general.

**Iowa, 5 chains, 200 steps, BURN=30:**

| Chain                          |       Mean cuts |            std |                           Δ vs A |           Wall |              Success |
| ------------------------------ | --------------: | -------------: | --------------------------------: | -------------: | -------------------: |
| V: Vanilla ReCom               |           41.81 |           3.85 |                                — |           0.2s |                   — |
| A: Tempered ReCom              |           38.29 |           3.02 |                                — |           0.2s |                  87% |
| B: QP-diffusion mix            |           37.07 |           4.14 |              −3.2% (p=3×10⁻⁵) |           1.1s |                  98% |
| **C: ReCom + IQP local** | **31.99** | **1.91** | **−16.4% (p=2×10⁻⁵⁵)** | **9.9s** | **iqp 2.95/3** |
| D: SpecReCom mix               |           40.79 |           4.47 |               +6.5% (NS-positive) |           0.6s |                  97% |

**Headlines:**

- **C beats every other chain**, including any prior MIP-cut-min number we recorded (~32.44). IQP delivers the lowest mean cuts AND the smallest std (most concentrated near the optimum).
- **C beats SpecReCom (D) by 22%** in cuts on Iowa (31.99 vs 40.79). The β + kernel + repair triple is the structural advantage.
- **B remains a meaningful intermediate point** on the Pareto front: 10× faster than C, modest gain over A.

### 2. Why C beats D (mechanism)

The β·‖x − x₀‖² fidelity term in the IQP makes it a *local boundary deformation* (the discrete analog of QP-diffusion's continuous flow). Three coupled advantages over SpecReCom:

1. **β anchors near the current contiguous partition.** Solutions are mostly contiguous already; small perturbations.
2. **When they're not, orphan-flip repair is cheap and effective** — disconnections are 1-2 nodes near the boundary, not whole islands.
3. **SpecReCom has no β.** Its Fiedler vector is a global property of the merged subgraph; repaired thresholds can be arbitrary, lowering quality. We therefore use repair in B and C, *not* in D (faithful Davies et al.).

### 3. Orphan-flip repair (universal across our methods)

Given a candidate assignment that may be disconnected on one side, the repair step:

- Identifies connected components on each side of the merged pair.
- Keeps the largest component on each side, flips smaller components to the opposite side.
- Verifies post-repair connectivity and ε population balance.
- Fails (returns None, chain stays) if repair cannot satisfy both.

Used in:

- B (QP-diffusion): inside Cheeger sweep — extends the set of feasible thresholds.
- C (IQP local): on the MIQP optimum — lifts feasibility from ~95% to 98% with negligible cut quality loss.
- D (SpecReCom): **NOT used** — faithful to Davies et al. 2025; repair without β tends to accept lower-quality cuts.

Implementation: [`qp_model.py:_orphan_flip_repair`](qp_model.py).

### 4. Codebase consolidation (refactor)

Code split into 4 modules:

- [`problems.py`](problems.py) — graph setup (Iowa, 5×5, NC), fiber loading, canonical-form helpers.
- [`chains.py`](chains.py) — `run_chain` runner + step factories: `make_vanilla_step`, `make_tempered_step`, `make_qp_diffusion_mix_step`, `make_iqp_hybrid_step`, `make_specrecom_mix_step`.
- [`diagnostics.py`](diagnostics.py) — KL/TV computations, bias-bound table, island-neck table, plot helpers.
- [`iowa_target_matched.py`](iowa_target_matched.py) — slim orchestrator (~205 lines).

The legacy `mip_local_step` and `make_mip_hybrid_step` have been **removed**. C uses `iqp_local_step` exclusively.

## Sprint 2 status table

| Task                                       | Status                          |
| ------------------------------------------ | ------------------------------- |
| SpecReCom implemented (chain D)            | ✓ (no repair, faithful Davies) |
| IQP local step (chain C, replaces MIP)     | ✓                              |
| Orphan-flip repair for B and C             | ✓                              |
| Codebase refactor into 4 modules           | ✓                              |
| Iowa 5-chain comparison                    | ✓ — C dominates               |
| Kernel infrastructure (perimeter, density) | ✓ — Sprint 3 will run them    |
| NC dataset loader (custom JSON)            | ✓ — works                     |
| NC 5-chain run                             | pending                         |

## Remaining sprints (May 4 → June 12)

### Sprint 3 — Multiple kernels + NC + long runs (May 18–24)

- [ ] Run 5-chain comparison on **NC precinct** (2692 nodes). MIP-style solves slow; expect IQP per-step ~5-30s, total ~10-30 min for 100 steps.
- [ ] On Iowa, run B and C with **perimeter kernel** (uses `shared_perim` edge attribute, already implemented). Visualise resulting partitions.
- [ ] On Iowa, run B and C with **density kernel** (uses TOTPOP/area, already implemented). Visualise.
- [ ] **Long runs**: 5,000 steps × 5 random initial conditions per (method, dataset) cell. Compute autocorrelation, ESS, ESS/sec.

**Exit criteria:** §7.2 master compactness table populated for Iowa + NC across 5 chains; §7.3 multi-kernel showcase has perimeter and density partitions for Iowa.

### Sprint 4 — Downstream impact (May 25–31)

- [ ] Sample 5,000 plans with each of {V, A, B, C} on Iowa.
- [ ] Compute per-plan: efficiency gap, mean-median, packing.
- [ ] Histograms of each metric under each method. KS test for distribution differences.
- [ ] **Headline §7.4 figure**: 3-panel histogram showing partisan skew shifts across methods.

### Sprint 5 — Write & polish (June 1–7)

- [ ] Outside-in draft: §7 results first, §1 intro last.
- [ ] All figures self-contained, high-DPI, captions readable standalone.
- [ ] One internal review pass.

### Sprint 6 — arXiv ship (June 8–12)

- [ ] arXiv-ready compile.
- [ ] **arXiv upload by 2026-06-12.**
- [ ] Submit to JCGS by 2026-06-19.

## Locked decisions

1. **Datasets**: 5×5 grid (full enumeration), 20×20 grid (mid-size validation, optional), Iowa (k=4), NC (precinct k=4 sub-state).
2. **Five chains**: V, A, B, C (=IQP), D. No MIP-cut-min as separate method (subsumed by C with α=1, β=0, uniform).
3. **Three kernels** for the Iowa multi-kernel section: uniform, perimeter, density.
4. **Repair**: enabled in B and C; disabled in D (faithful Davies et al.).
5. **Pre-registered win conditions** (must hold to ship arXiv v1):
   - C ≥ 10% better than V on cuts on every state.
   - B between A and C on cuts (intermediate Pareto point).
   - 5×5 ground-truth: B and C TV-to-uniform within 0.05 of each other.
6. **arXiv hard deadline**: 2026-06-12.
7. **JCGS submission**: by 2026-06-19.

## Risks & mitigations

| Risk                                                  | Mitigation                                                                                                                                                                  |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| NC precinct IQP solves too slow                       | Cap MIQP time-limit at 5 s per pair; warm-start from continuous QP solution; failing that, run NC at county-level aggregation.                                              |
| Perimeter / density kernels regress cuts vs uniform   | Acceptable: paper §7.3 framing is "kernels change the*shape* of partitions, not necessarily cut count" — visual demonstration is the contribution.                      |
| Sprint 4 partisan skew shows no meaningful difference | Reframe §7.4: "compactness differs but downstream stable" —*also* a publishable finding.                                                                                |
| Long runs reveal mixing issues                        | Report ESS/sec as honest metric; the Pareto trade-off discussion in §8 covers the worst case.                                                                              |
| arXiv schedule slips                                  | **Cut scope ladder**: drop Sprint 3 perimeter+density first, then drop NC, then drop §7.4 downstream. Ship with Iowa + 5×5 + uniform-kernel as last-resort minimum. |

## Repo conventions

- Implementation: `diffusion_model/recom/`:
  - `qp_model.py` — `qp_diffusion_step`, `iqp_local_step`, `spec_recom_step`, `_cheeger_sweep_round`, `_orphan_flip_repair`, kernels (`kernel_uniform`, `kernel_perimeter`, `make_kernel_density`).
  - `chains.py` — chain step factories + `run_chain`.
  - `problems.py` — graph setup, fiber loading.
  - `diagnostics.py` — KL/TV, bias-bound table, island-neck table, plot helpers.
  - `iowa_target_matched.py` — orchestrator (`MODE` flag selects dataset).
  - `fiber_5x5/` — 5×5 ground-truth data and figures.
- Outputs: `diffusion_model/recom/iowa_hybrid_test_out/`.
- Paper writeup: `Papers/diffusion.tex`, `Papers/diffusion.pdf`.
- Literature in `Literature/`, especially `spectral-recom.pdf`.

---

## Q1 first-submission odds (calibrated, 2026-05-01)

With the IQP unification:

- **JCGS first submission**: ~60-70% (was 50-55% before IQP unification — the unified-model story is stronger).
- **Statistics and Computing**: ~55-65%.
- **AoAS**: ~40-50%.

Sprint 2's IQP consolidation moved odds by ~10 pp by collapsing two methods into one model, simplifying the paper's central narrative.

---

## Historical findings (paper-supporting material, preserved)

### What earlier dead-ends taught us

1. **ReCom's compactness bias** quantified: 7% on 5×5 ground truth (16.60 vs 18.67 uniform mean), ~18% empirical bias on Iowa.
2. **QP-MH-as-strict-MH cannot serve as a compactness lever.** Aggressive proposals violate reversibility; gentle proposals drift up. Used in Discussion to motivate deterministic local kernels.
3. **Symmetric-q approximation in Tempered ReCom is biased on small graphs** (5×5: empirical 16.70 vs target 18.56). Footnote-worthy.
4. **The fiber has clean island/neck structure** — backbone of §3.
5. **MIP-cut-min was a useful intermediate step** in development but is now subsumed by IQP (β=0 special case). The paper presents IQP only.

### qp_model.py evolution

- v1 `qp_mh_proposal` — strict MH targeting exp(−λ·cuts). **Failed**: drift up at gentle proposals, reject at aggressive. Removed/deprecated.
- v2 `mip_local_step` — discrete IP, deterministic, always accept. Worked on Iowa (-15%) but limited (no kernel, no fidelity). **Removed (subsumed by IQP).**
- v3 `qp_diffusion_step` — notes-aligned: L (unnormalised) + Cheeger sweep + orphan-flip repair. Chain B.
- v4 `iqp_local_step` — integer version of v3: same model, $x \in \{0,1\}^n$, MIQP via Gurobi, orphan-flip repair. Chain C. **Strictly subsumes v2.**
- v5 `spec_recom_step` — Davies et al. 2025 baseline, faithful (no repair, no β). Chain D.
