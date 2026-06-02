# Mean Field Game Modeling of Tourist Flow at Macau Heritage Attractions

A differentiable **mean field game (MFG)** on a graph of Macau's UNESCO heritage
attractions, calibrated against public tourism data and used to design congestion
interventions (entrance metering, routing recommendations) with explicit
validity-scope, compliance, and robustness analysis.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)

## What this is (and what it is not)

Tourists in the Macau historic centre concentrate into a few narrow corridors
(Largo do Senado, Rua de São Paulo), where crowd density routinely exceeds comfort
and safety thresholds. We model each visitor as an agent choosing a path that
maximises sightseeing utility net of congestion and walking cost; at Nash
equilibrium this is a coupled Hamilton–Jacobi–Bellman (HJB) + Fokker–Planck (FP)
system on the graph, solved by a damped fixed-point iteration. Both **calibration**
and **intervention design** are posed as end-to-end gradient problems through the
differentiable forward simulator (PyTorch + Adam).

**Honest scope.** This MFG is **not** offered as the most accurate predictor of the
static visitor distribution — we show that a simple gravity model fits that
(nearly time-invariant) distribution far more tightly (held-out MAE **0.0003** vs the
MFG's **0.018**; see `docs/09_baselines.md`). The MFG's held-out spatial fit is a
**consistency check**, not the headline. Its value is that it is the only model here
that represents **intra-day congestion dynamics** and can therefore evaluate
**interventions** — which gravity/logit models cannot even express.

## Key results

| Result | Value | Source |
|---|---|---|
| Held-out spatial calibration MAE | 0.018 (consistency check) | EXP-05 |
| Best simple baseline (gravity) vs MFG | 0.0003 vs 0.018 | EXP-10 / `docs/09` |
| Entrance metering peak reduction | 5–8% (single terminal ≈12% of arrivals) | EXP-07 |
| Routing peak reduction (full compliance) | ~71% **upper bound** | EXP-08 |
| Routing under partial compliance φ | 6.7% (φ=0.1) → 70.9% (φ=1) | EXP-09 / `docs/10` |
| Combined metering+routing | 72.5% system-peak ↓, 94% visits kept | EXP-09 |
| Calibration vs assumed arrival profile | MAE 0.0182–0.0184 (profile-invariant) | EXP-11 / `docs/08` |
| Damped fixed point (λ=0.5) contraction | factor ≈0.5 (~20 iters); unique equilibrium | EXP-12 / `docs/11` |
| One-step calibration gradient | directionally exact (cos 1.0), β magnitude-biased | EXP-12 / `docs/12` |

Figures are in `figures/`; the full table is `figures/results_table.md`.

## Method (brief)

Discrete-time MFG on graph `G=(V,E)` with walking-distance matrix `D` (step `Δt`).
A tourist at node `v` earns reward `α_v·Δt − β·ρ_v(t)·Δt` and chooses the next node
`w` (or to exit) paying `γ·D_vw`:

- **HJB:** `u_v(t) = α_v·Δt − β·ρ_v(t)·Δt + max(max_w{−γ·D_vw + u_w(t+Δt)}, 0)`
- **Policy / FP:** `π_vw ∝ exp((−γ·D_vw + u_w(t+Δt))/ε)`,
  `ρ_w(t+Δt) = Σ_v ρ_v(t)·π_vw + g_w(t)·Δt`
- **Equilibrium:** damped fixed point `ρ ← (1−λ)ρ + λ·FP(HJB(ρ))`.
- **Routing** adds a perceived bonus `η_vw` inside the max/softmax; **compliance**
  blends complier/ignorer policies by a fraction `φ`.

See `docs/03_methodology.md` and `docs/ARCHITECTURE.md`.

## Repository structure
```
src/            implementation (solver, calibration, baselines, interventions, runners EXP-01..12)
configs/        per-experiment YAML configs
tests/          unit tests (toy/synthetic; no data download needed)
docs/           methodology + research writeups (validity scope, baselines, interventions, convergence, gradient)
figures/        generated figures + results_table.md
data/           skeleton + acquisition guide (raw data NOT redistributed)
```

## Installation
```bash
conda env create -f environment.yml
conda activate macau-tourist-flow
pip install -e .
pytest -q -m "not slow"     # unit tests (toy data; no downloads)
```

## Reproduce
Data (DSEC / MGTO / OpenStreetMap) is not redistributed — see `data/README.md`
(a cached OSM walking graph is included). Acquire the source data, then build the
processed datasets:
```bash
python -m src.ingest_data --source all   # writes data/processed/*.parquet
```
Then run **the whole pipeline with one command**:
```bash
python main.py                 # EXP-01..12 + figures, in order
python main.py --list          # show the steps
python main.py --only exp08    # run a single experiment
python main.py --from exp07    # run from this step to the end
```
`main.py` **auto-wires** EXP-05's calibrated parameters into the downstream
experiments (EXP-07–11), so you never edit a config path by hand. Each step seeds
all RNGs and writes a config snapshot + `summary.txt` to `experiments/<date>_<EXP>/`;
figures + `results_table.{md,csv}` are written to `figures/`.

Individual steps can still be run directly, e.g. `python -m src.run_exp08`.

## Limitations (read these)
- Calibration data is **monthly**; the within-day arrival profile is **assumed**, so
  intra-day peak magnitudes are model outputs under that assumption (we report
  *relative* reductions, shown robust across profiles in EXP-11).
- The per-attraction target is an **MGTO/proxy** estimate; a stronger test needs real
  time-varying per-attraction counts.
- Routing's ~71% is a **perfect-compliance upper bound**; the deployable effect is the
  φ-band (EXP-09).
- Single-agent rationality, static topology, exogenous uniform compliance.

## License & attribution
MIT (see [LICENSE](LICENSE)). Map data © OpenStreetMap contributors (ODbL).
Author identities are withheld from this public mirror; available on request.
