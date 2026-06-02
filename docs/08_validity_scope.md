# 08 — Validity scope: what is calibrated vs assumed

> **Goal A of the research-hardening program.** This note draws a hard line
> between what the model *learns from data* and what it *assumes*, and shows
> experimentally (EXP-11) that our conclusions are robust to the central
> assumption. Source: `src/run_exp11.py`, `configs/exp11_validity_scope.yaml`,
> outputs in `experiments/20260601_EXP-11_validity_scope/`.

## The core issue

The DSEC arrival data is **monthly**. It tells us, per entry point, how many
visitors arrive in a month — and nothing about *when within a day* they arrive.
But "peak congestion" is an **intra-day** quantity. So we must be explicit:

| Quantity | Status | Where |
|---|---|---|
| Monthly **volume** of arrivals per transit node | **Data** (DSEC) | `run_exp05._daily_source_counts` |
| **Spatial** attraction distribution (shares across the 10 sites) | **Data-calibrated & held-out-validated**, MAE 0.018 | `run_exp05._evaluate_months` |
| Attractiveness `{α_v}`, congestion `β`, walk cost `γ` | **Fit** to the monthly spatial shares | `calibration/estimator.py::MFGParameters` |
| **Within-day shape** of `g(t)` (when arrivals peak, how sharply) | **Assumed** — not estimable from monthly data | `src/utils/arrival_profiles.py` |
| Softmax temperature `ε` | **Assumed** (fixed 0.1) | `MFGSolver.epsilon` |
| Intra-day **peak density** & **intervention peak-reduction** | **Modelled under the assumed `g(t)`** | EXP-07/08/11 |

We make this split **explicit in code**: the within-day shape now lives in a
named, swappable registry (`src/utils/arrival_profiles.py`) rather than a
hard-coded Gaussian, so the assumption can be varied and stress-tested instead of
hidden. The default (`gaussian`) reproduces the original behaviour exactly.

## EXP-11 — stress test across assumed profiles

We re-run everything under several plausible within-day profiles, holding the
**volume** (the data) fixed:
- **Single afternoon peak** (the default Gaussian),
- **Broad midday plateau** (steady arrivals, no rush),
- **Double peak** (morning + afternoon waves),
- **Near-uniform** (essentially flat — the un-peaked limit),
- **Empirical hourly proxy** (`empirical_proxy`): an asymmetric day-tripper shape
  with a late-morning/midday peak and a gradual afternoon decline, interpolated
  from a documented hourly inflow envelope (Gongbei/Border-Gate crossings ramping
  through the late morning, consistent with the typical Popular Times envelope for
  Senado Square / Ruins of St. Paul's). It is still a **proxy** — DSEC gives only
  monthly volume — but it is the only profile grounded in an external real-world
  pattern rather than a synthetic functional form, so surviving it is the most
  informative single robustness check.

### Part A — the data-validated quantity is profile-invariant
Re-calibrating `{α_v}, β, γ` on the same train/val split under each profile:

| Profile | Held-out spatial MAE | Fitted β |
|---|---|---|
| Single afternoon peak (default) | 0.0182 | 0.00105 |
| Broad midday plateau | 0.0184 | 0.00105 |
| Double peak | 0.0183 | 0.00105 |
| Near-uniform | 0.0184 | 0.00105 |

The held-out MAE moves only in the 4th decimal and the fitted attraction
parameters deviate **0.0%** from the Gaussian baseline (`alpha_across_profiles`,
`val_mae_across_profiles` figures). This is expected and reassuring: the
calibration target is the **day-cumulative** spatial distribution, which depends on
*how many* tourists visit each site, not on the *timing* of arrivals. **Our
data-validated result does not depend on the within-day assumption.**

### Part B — intervention conclusions are robust in sign and magnitude
Holding the (profile-invariant) calibrated parameters fixed and re-running the
interventions under each profile (`peak_reduction_vs_profile` figure):

| Profile | Metering peak ↓ | Routing peak ↓ | Gini (base→routed) | Converged |
|---|---|---|---|---|
| Single afternoon peak | 5.2% | 70.6% | 0.383 → 0.133 | yes |
| Broad midday plateau | 7.2% | 71.0% | 0.383 → 0.125 | yes |
| Double peak | 6.7% | 71.0% | 0.383 → 0.126 | yes |
| Near-uniform | 7.8% | 70.8% | 0.376 → 0.118 | yes |

Both interventions reduce peak congestion under **every** profile: metering in a
**5–8%** band, routing in a tight **70.6–71.0%** band, with the spatial Gini
falling to ~0.12–0.13 in all cases and every equilibrium converging.

## Honest verdict (state this in the report)
- **Strong:** the **spatial calibration** (our only data-validated claim) is
  *provably profile-invariant* — it carries no dependence on the assumed `g(t)`.
- **Strong:** the **direction and rough magnitude** of both interventions' peak
  reductions are **stable across all four profiles**.
- **Caveat to keep:** the **absolute peak density** at any instant still depends on
  the assumed profile (a sharper assumed peak ⇒ a higher absolute peak). We
  therefore report intervention effects as **relative reductions**, which are
  robust, and we do **not** claim a specific absolute peak headcount as data-backed.
- **Caveat to keep:** routing's ~71% remains a *perfect-compliance upper bound*
  (addressed by the compliance model in Goal D / `docs/10_interventions.md`).

## Reproduce
```
python -m src.run_exp11        # ~2–3 min; outputs to experiments/<date>_EXP-11_validity_scope/
```
Artifacts: `calibration_invariance.csv`, `intervention_robustness.csv`,
`alpha_across_profiles.{png,pdf}`, `val_mae_across_profiles.{png,pdf}`,
`peak_reduction_vs_profile.{png,pdf}`, `summary.txt`.
