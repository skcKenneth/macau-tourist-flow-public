# 02 — Literature Review

> Annotated bibliography. Every paper we cite in the final report must appear here with a 1-paragraph note on (a) what it does, (b) how we use it, (c) limitations relevant to our setting. Add new entries with `## [year] author — title` as a heading.

## Reading priorities for first 2 weeks
1. Hughes (2002) — continuum PDE foundation
2. Bagagiolo & Pesenti (2017) — MFG-on-network for tourists (closest precedent)
3. Achdou & Capuzzo-Dolcetta (2010) — numerical methods for MFG
4. Helbing & Molnár (1995) — social force microscopic baseline
5. Chen et al. (2018) — Neural ODE for parameter learning

---

## Pedestrian flow — continuum models

### Hughes, R. L. (2002). A continuum theory for the flow of pedestrians. *Transportation Research B*, 36(6), 507–535.
- **What**: Treats pedestrians as a continuum density satisfying conservation + an eikonal equation for desired direction. Closes the system with a velocity-density relation.
- **Use**: Conceptual foundation. Our graph MFG is the network analog.
- **Limitations**: Continuum approximation breaks down in narrow corridors with discrete decision points (which is exactly Macau's setting). Motivates moving to a graph formulation.

### Helbing, D., & Molnár, P. (1995). Social force model for pedestrian dynamics. *Physical Review E*, 51(5), 4282.
- **What**: Microscopic model where each pedestrian is a point mass with attractive + repulsive social forces.
- **Use**: Baseline comparison and validation. We'll use a published implementation (PedPy or similar) to cross-check our graph model in a single-corridor case.
- **Limitations**: Computationally expensive for city-scale; harder to fit to aggregate data than density-based models.

## Mean field games

### Lasry, J.-M., & Lions, P.-L. (2007). Mean field games. *Japanese Journal of Mathematics*, 2(1), 229–260.
- **What**: Original MFG framework. Coupled HJB + FP system characterizes Nash equilibrium of $N \to \infty$ symmetric agents.
- **Use**: Cite for mathematical foundation. Do not derive — just state.

### Bagagiolo, F., & Pesenti, R. (2017). Mean field game for tourists' flow on a network. [find exact venue]
- **What**: Models tourists choosing paths in a small network (station + 2 attractions) as MFG. Closed-form for special cases, numerical for general.
- **Use**: **Closest precedent.** Our work extends this to a 10–15 node graph with real data calibration and ML-based parameter learning. Cite extensively.
- **Limitations**: Their networks are tiny (≤3 attractions), no real data calibration, no optimization for intervention design — these are our gaps to fill.

### Achdou, Y., & Capuzzo-Dolcetta, I. (2010). Mean field games: numerical methods for the planning problem. *SIAM J. Numer. Anal.*, 48(3), 1136–1162.
- **What**: Finite-difference schemes for solving MFG PDEs.
- **Use**: Algorithmic basis for our solver, adapted to graph setting.
- **Limitations**: Designed for continuous space; we adapt to discrete graph (cite, adapt, document differences).

## Crowd dynamics — applied / data-driven

### Corbetta, A., et al. (2018). Continuous measurements of real-life bidirectional pedestrian flows... (or similar Eindhoven station papers)
- **What**: Long-term Lagrangian measurements + macroscopic models for crowd forecasting.
- **Use**: Inspiration for calibration approach. Their bi-directional macroscopic model is a calibration target.

### Crociani, L., et al. (2019). Multidestination pedestrian flows... (museum studies)
- **What**: Clustering trajectories in museums, building transition matrices, stochastic simulator.
- **Use**: The transition-matrix approach is our discrete baseline. Compare against full MFG.

## Macau tourism — context

### Lu, H., Zhou, X., Chen, Z., & Cheng, J. (2026). Multi-source data integration for safety evaluation of walking tourism routes: coupling spatial analysis of attractiveness and carrying capacity in Macao. *Sustainability*, 18(4), 1984. https://doi.org/10.3390/su18041984
- **What**: Integrates social-media and spatial data on MGTO walking routes; couples node attractiveness with a spatial carrying-capacity index and diagnoses “high attractiveness–low capacity” hotspots (including the Ruins of St. Paul’s corridor).
- **Use**: **Macau-specific empirical context** for why peak-density management matters and for the core–periphery imbalance narrative. Complements our MFG intervention framing (they assess static risk; we model dynamic congestion and policy levers).
- **Limitations**: Static / index-based capacity, not a forward-looking equilibrium model; no intervention optimisation; data are route-level proxies rather than the fine-grained 13-node graph used here.

### Macao Government Tourism Office (MGTO) — visitor statistics and walking-route materials
- **What**: Official tourism statistics, walking-route promotions, and (where available) per-attraction visitor information used in this project’s MGTO proxy (`data/raw/mgto/`).
- **Use**: **Data provenance** for attraction-side targets and qualitative validation of which nodes are bottlenecks. Cite as institutional source, not as a peer-reviewed model.
- **Limitations**: Per-attraction time series are often incomplete in the public record; this repo uses proxy counts where official series are missing (see `data/README.md`).

### Statistics and Census Service (DSEC) — Visitor Arrivals / Tourism Statistics
- **What**: Monthly and quarterly official arrival statistics by mode of transport (land, sea, air) for Macao.
- **Use**: **Primary data source** for the arrival side of EXP-05 calibration (`data/processed/arrivals_monthly.parquet`). Document acquisition in `docs/04_data_sources.md`.
- **Limitations**: Monthly aggregates only — **no within-day profile** — so intra-day `g(t)` remains an assumed shape in our MFG (see `docs/08_validity_scope.md`).

## Machine learning

### Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural ordinary differential equations. *NeurIPS 2018*.
- **What**: Neural networks parameterizing ODE/PDE dynamics, trainable end-to-end via adjoint method.
- **Use**: Parameterize unknown components of MFG (e.g., congestion cost shape) and learn from data.
- **Limitations**: Training instability; we need careful initialization and possibly classical optimization as fallback.

### Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks (PINNs). *Journal of Computational Physics*, 378, 686–707.
- **What**: NNs constrained to satisfy known PDEs.
- **Use**: Alternative approach to our calibration; document if we end up using.

## Graph theory / network science

### Newman, M. E. J. (2010). *Networks: An Introduction*. Oxford University Press.
- **What**: Standard textbook on network science.
- **Use**: Centrality measures, robustness analysis — used in our network characterization section.

---

## How to add a new reference
1. Find the paper, read at least the abstract + intro + conclusion.
2. Add an entry under the appropriate section above.
3. Write the 3-part note: what / how we use / limitations.
4. Add the BibTeX to `report/references.bib`.
5. Cross-check that any quantitative claim appears in `docs/05_experiment_plan.md` or the relevant experiment writeup (`docs/08`–`12`).

## Open questions for further reading
- [ ] Does anyone publish per-attraction visitor counts for Macau heritage sites? Need to dig MGTO Yearbook of Statistics.
- [ ] Is there published MFG work on UNESCO sites? (Venice, Cinque Terre, etc.)
- [ ] What's the state of the art for differentiable MFG simulators (post-2022)?
