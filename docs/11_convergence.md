# 11 — Convergence of the damped fixed-point solver

> **Goal B (part 1) of the research-hardening program.** Numerically-verified
> convergence behaviour of the MFG solver, with an explainable sketch. Source:
> `src/run_exp12_convergence.py`, outputs in
> `experiments/20260601_EXP-12_convergence_1/`. *Proven* claims and *numerically
> supported* claims are separated below.

## The map and the iteration
The Nash equilibrium density `ρ*` is a fixed point of `S = FP ∘ HJB`: given a
density we solve the backward HJB for the cost-to-go, derive the softmax policy,
and push the density forward (`src/models/mfg_solver.py`). We iterate the **damped**
map
```
T_λ(ρ) = (1 − λ) ρ + λ S(ρ),      ρ_{k+1} = T_λ(ρ_k),
```
which has the **same fixed points** as `S` for any `λ ∈ (0, 1]` (if `ρ = T_λ(ρ)`
then `ρ = S(ρ)`), but better contraction behaviour.

## Contraction sketch (what we can argue, not a full proof)
`HJB` is built from a `max`/soft-max and `FP` from a row-stochastic policy matrix,
so `S` is Lipschitz. Two terms control its Lipschitz constant `L_S`:
- the **congestion coupling** enters only through `−β ρ_v dt` in the HJB reward, so
  the sensitivity of the policy to `ρ` scales with `β` (and with the demand level,
  i.e. the magnitude of `ρ`);
- the **softmax temperature** `ε` smooths the policy: larger `ε` ⇒ a flatter,
  less reactive policy ⇒ smaller `L_S`.
When `L_S < 1`, `S` is a contraction and Banach's theorem gives a **unique** fixed
point with geometric convergence. When `L_S ≥ 1` (large `β·‖ρ‖` and small `ε`), the
undamped iteration can limit-cycle. Damping helps because
`Lip(T_λ) ≤ (1−λ) + λ L_S`: mixing in the identity pulls the effective constant
toward 1 from above when `L_S > 1` is mild, and below 1 when `L_S` is near 1,
trading a slower rate for a wider convergent region.

## Numerical verification (4-node toy, demand ≈ 200)
Empirical contraction factor `c ≈ median r_{k+1}/r_k` of the residual sequence
(`contraction_sweep.csv`, `contraction_lambda_*` heatmaps):

| Setting | λ = 1 (undamped) | λ = 0.5 (damped) |
|---|---|---|
| β = 0.001, ε = 0.1 | **diverges** (c ≈ 1.0) | **c ≈ 0.51**, converges in 21 it |
| β = 0.01, ε = 0.1 | diverges | **c ≈ 0.52**, converges in 20 it |
| β = 0.001, ε = 0.2 | c ≈ 0.04 (converges) | c ≈ 0.52 (converges) |
| β = 0.05–0.5, ε = 0.1 | diverges | diverges (c ≈ 1.0) |

Reading:
1. **Damping justifies λ = 0.5.** At the calibrated-scale β (0.001–0.01) with the
   project's ε = 0.1, the *undamped* iteration oscillates, while λ = 0.5 converges
   with `c ≈ 0.5` — i.e. the residual halves each step (`0.5²⁰ ≈ 1e-6`), matching
   the ~17–22 iterations observed across EXP-05/07/08/09/11 on the real graph.
2. **Larger ε also helps** (ε = 0.2 converges even undamped at low β), consistent
   with the smoothing argument.
3. **A critical β exists.** Beyond `β ≈ 0.05` *at this toy demand scale* (where
   `β·‖ρ‖` is large) even λ = 0.5 oscillates. The **fitted** β on real data
   (≈ 0.001) sits far inside the convergent region, which is why every real-graph
   run converges. This is the quantitative basis for the long-standing note that
   high β oscillates.

## Existence / uniqueness (numerical)
From **24 random initial densities** at (β = 0.01, ε = 0.1, λ = 0.5 — inside the
convergent region) the iteration reaches the **same** fixed point: the maximum
pairwise distance between the 24 converged densities is **1.1 × 10⁻⁴** (small
relative to a peak density of order tens). Combined with the contraction argument, this supports a **unique**
equilibrium in the convergent regime (numerically; not a closed-form proof).

## Honest scope
- *Argued (sketch):* `T_λ` and `S` share fixed points; `S` is Lipschitz; a
  contraction (`L_S < 1`) implies a unique, geometrically-attained equilibrium.
- *Numerically supported:* the contraction factor map, the λ = 0.5 justification,
  the critical-β boundary, and uniqueness from random restarts.
- *Not claimed:* a closed-form bound on `L_S` in terms of (β, ε, γ, graph). A
  rigorous bound is future work; the empirical map is sufficient to justify the
  solver settings actually used.

## Reproduce
```
python -m src.run_exp12_convergence
```
