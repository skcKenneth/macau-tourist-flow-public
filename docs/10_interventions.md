# 10 — Combined intervention + compliance

> **Goal D of the research-hardening program.** Completes the intervention story
> credibly: combines the two levers, replaces the perfect-compliance assumption
> with a realistic compliance model, and reports robustness as ranges. Source:
> `src/run_exp09.py`, `MFGSolver.fixed_point_iteration_compliance`, outputs in
> `experiments/20260601_EXP-09_combined_compliance/`.

## Part 1 — Combined metering × routing (Pareto)
System-wide peak over the two bottlenecks {Ruins of St. Paul's, Senado Square} vs
total preserved attraction-visits (`pareto_combined` figure):

| Strategy | System-peak reduction | Visits preserved |
|---|---|---|
| Entrance metering only (best) | 5.5% | 93% |
| **Combined (metering 10% cap + routing)** | **72.5%** | **94%** |

Metering alone is limited (~5–6%) because the metered terminal is only ~12% of
arrivals (EXP-07). Routing dominates the combined gain; adding metering on top
preserves the gain while shaving the residual peak. **Combined ≥ either lever
alone**, as hypothesised, while keeping ~94% of attraction-visits.

## Part 2 — Compliance: the deployable range (the key honesty fix)
The routing result assumes **everyone** follows the recommendation. We replace this
with a **two-type mean field game**: a fraction `φ` perceive the routing bonus and
best-respond; the remaining `1−φ` ignore it; both share one congestion field
(`MFGSolver.fixed_point_iteration_compliance`). Sweeping `φ`
(`compliance_phi` figure):

| Compliance `φ` | Peak reduction at Ruins |
|---|---|
| 1.0 (everyone complies) | **70.9% — upper bound, not deployable** |
| 0.10 (one in ten complies) | 6.7% |

The benefit rises monotonically with compliance. **We therefore report a band, not
a single number**: real signage/app compliance is partial, so the *deployable*
effect sits between these ends (e.g. a plausible 25–50% compliance yields an
intermediate reduction — see the figure/`compliance_phi.csv`). The headline ~71% is
explicitly the **perfect-compliance ceiling**.

### Part 2b — heterogeneous / uncertain compliance band
A single deterministic `φ` still treats the whole population as homogeneous. We
additionally draw compliance from a **distribution** `φ ~ Beta(mean, κ)` (a
population with heterogeneous, uncertain willingness to follow the nudge) and
propagate every draw through the two-type equilibrium
(`interventions.compliance_robustness_band` + `sample_beta_compliance`). This yields
a **p5–p95 robustness band** on the routing peak-reduction (overlaid on the
`compliance_phi` figure; full stats in `compliance_distribution.csv`), so the
deployable claim is an interval rather than a point even before fixing a mean
compliance level. Configure via the `compliance.distribution` block in
`configs/exp09_combined.yaml`.

## Part 3 — Robustness
Re-evaluating the **deployed** routing policy (the η optimised once, under full
compliance) under the four assumed within-day profiles (Goal A) crossed with
**±20% misspecification of β**:

- Ruins peak-reduction range: **[70.2%, 71.0%]**, every equilibrium converged.

The policy's effect is essentially invariant to both the assumed arrival profile and
±20% error in the congestion parameter — it is not an artefact of a particular
assumption.

## Honest headline for the report
> "An optimised routing policy can cut peak congestion at the Ruins of St. Paul's by
> ~71% **if fully complied with** — an upper bound. Under partial compliance the
> effect scales down (≈7% at 10% compliance), so the *deployable* benefit is a band
> set by realistic compliance. Combining routing with entrance metering preserves
> the gain while keeping ~94% of attraction-visits, and the result is robust to the
> assumed within-day arrival profile and to ±20% error in the congestion parameter."

## Caveats
- η is optimised under full compliance (the planner's design problem); we then
  *evaluate* under partial compliance. Jointly optimising η for a target φ is future
  work.
- Compliance `φ` is exogenous and uniform; heterogeneous/endogenous compliance
  (e.g. tourists more likely to comply when a site is visibly crowded) is not modelled.

## Reproduce
```
python -m src.run_exp09     # optimises eta, then Pareto + phi sweep + robustness
```
Artifacts: `pareto.csv`, `compliance_phi.csv`, `robustness.csv`,
`pareto_combined.{png,pdf}`, `compliance_phi.{png,pdf}`, `summary.txt`.
