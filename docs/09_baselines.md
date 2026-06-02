# 09 — Fair baselines + ablation: does the MFG earn its complexity?

> **Goal C of the research-hardening program.** Honest, apples-to-apples
> comparison of the MFG against simpler models on the **same real-DSEC held-out
> split** as the headline MAE. Source: `src/run_exp10.py`,
> `src/models/baselines.py`, outputs in `experiments/20260601_EXP-10_baselines/`.

## Result (held-out spatial MAE, lower is better)

| Model | Held-out MAE | Congestion? | Dynamics? |
|---|---|---|---|
| **Gravity** (distance-decay) | **0.0003** | no | no |
| **Multinomial logit** (static discrete choice) | **0.0004** | no | no |
| Full MFG (EXP-05) | 0.0182 | yes | yes |
| MFG, β = 0 (ablation) | 0.0189 | no | yes |
| Random walk (non-strategic) | 0.0791 | no | no |

(The random-walk number ≈ the original EXP-02 0.079, now confirmed on the real
protocol — it is the genuinely poor baseline.)

## The honest reading — the MFG does NOT win the spatial-prediction task

On predicting the spatial attraction distribution, **simple static models beat the
MFG by ~50×**. We report this plainly. Two reasons, both important:

1. **The task is easy for static models and the target barely varies.** Our
   per-attraction target is the MGTO/proxy distribution from `annual_visitors_est`
   — effectively *constant across months*. A gravity or logit model has ~10 free
   attraction parameters fit directly to ~10 near-constant target shares, so it
   essentially **memorizes** the distribution; "held-out" months share the same
   proxy, so this metric tests fit, not temporal generalization. The MFG's
   prediction must instead pass through the congestion + dynamics + graph, which
   *constrains* it away from an exact match.
2. **Congestion is almost irrelevant to the *equilibrium spatial shares*.** The
   β = 0 ablation (0.0189) is barely worse than the full MFG (0.0182): turning
   congestion off costs only **0.0007 MAE**. This is consistent with the EXP-06
   sensitivity finding that β is a minor driver of the spatial distribution.

## So why use the MFG at all? (the defensible reframing)

The comparison clarifies — rather than weakens — the MFG's role, **provided we
state its contribution correctly**:

- The MFG's value is **not** as a better spatial *predictor*. Gravity/logit win
  there, and we say so.
- The MFG is the **only model here that represents intra-day congestion dynamics**,
  and therefore the only one that can even *pose* the questions this project is
  about: how does the **intra-day peak** form, and how do **entrance metering** and
  **routing recommendations** (and partial **compliance**) change it? Gravity and
  logit have no time axis, no congestion, no peaks, and no notion of an
  intervention — they cannot answer the research question at all.
- Its spatial fit being **comparable in order** (MAE 0.018, well within the 0.05
  bar) is enough as a **consistency check** that the mechanistic model is not
  wildly off the observed popularity pattern. We do **not** claim it is the most
  accurate spatial fit.

In short: **the MFG earns its complexity on the *intervention-design* task, not the
spatial-prediction task.** Honest framing for the report and oral defense:
> "A gravity model predicts the static visitor distribution far more accurately
> than our MFG, because that distribution is nearly time-invariant and has few
> degrees of freedom. But a gravity model cannot represent congestion or evaluate
> an intervention. The MFG is justified as a *mechanistic simulator for congestion
> and policy*, validated to be consistent with the observed spatial pattern, not as
> a spatial-distribution predictor."

## Limitations / what would strengthen this
- The spatial target is the MGTO **proxy** (`confidence="estimate"`). A genuine
  temporal-generalization test needs real per-attraction counts that vary month to
  month; until then, low spatial MAE is weak evidence and should not be the
  headline claim.
- A fairer "earns its complexity" test would compare models on a task only the MFG
  can do — but then there is no baseline, by construction. We therefore lean on the
  consistency check + the intervention results (EXP-07/08/09) for the MFG's value.

## Reproduce
```
python -m src.run_exp10     # ~3 min (the beta=0 re-fit dominates runtime)
```
Artifacts: `model_comparison.csv`, `model_comparison.{png,pdf}`, `summary.txt`.
