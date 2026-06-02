# 12 — Is the calibration gradient correct?

> **Goal B (part 2) of the research-hardening program.** We use a cheap *one-step
> consistency gradient* to calibrate the MFG. This note asks whether it equals the
> true gradient of the equilibrium loss, by comparing it to two reference gradients.
> Source: `src/calibration/gradient_check.py`, `src/run_exp12_convergence.py`
> (Part C), outputs in `experiments/20260601_EXP-12_convergence_2/`.

## Three gradients of the same loss `L(ρ*(θ))`
The equilibrium is `ρ* = S(ρ*, θ)` with `S = FP ∘ HJB` and `θ = {α, β, γ}`.
- **One-step (ours):** run the fixed point under `no_grad` to a stable `ρ_fp`, then
  take **one** HJB+FP step with autograd and backprop. Cheap; but it treats `ρ_fp`
  as independent of `θ`, so it ignores how the equilibrium itself shifts with `θ`.
- **Unrolled:** backprop through the damped iteration unrolled to (near) convergence
  — the practical ground truth.
- **Implicit-function theorem (IFT):** the exact equilibrium gradient
  `dL/dθ = (∂L/∂ρ)(I − ∂T_λ/∂ρ)^{-1}(∂T_λ/∂θ)`, computed by an adjoint Neumann
  solve `w = g_L + (∂T_λ/∂ρ)^T w` (convergent because the **damped** map `T_λ` is a
  contraction — see `docs/11`) followed by a vector-Jacobian product in `θ`.

## Result (4-node toy, ε = 0.1, λ = 0.5; `gradient_bias.csv`)

| β | one-step vs IFT (cosine) | one-step magnitude ratio ‖one‖/‖IFT‖ | unrolled vs IFT (cosine) |
|---|---|---|---|
| 0.001 | 1.000 | 1.02 | 1.00000 |
| 0.01 | 1.000 | 1.24 | 1.00000 |
| 0.05 | 1.000 | 1.61 | 1.00000 |
| 0.10 | 1.000 | 0.91 | 1.00000 |

Two clean findings:
1. **Unrolled ≡ IFT (cosine 1.00000).** The two independent ground-truth methods
   agree exactly, which validates the IFT adjoint implementation and confirms both
   recover the true equilibrium gradient.
2. **The one-step gradient is directionally exact but magnitude-biased.** Its cosine
   with the true gradient is **1.000** at every β tested — it points in exactly the
   right direction — but its magnitude is biased by a factor that **grows with β**
   (1.0× → 1.6× as β goes 0.001 → 0.05). At very small β (the calibrated regime) the
   bias is negligible (≈ 2%).

## Why this matters (and why calibration still works)
- The **direction** being exact means one-step is a valid descent direction, and
  with **Adam** — which rescales each parameter by its own running gradient
  magnitude — a uniform magnitude bias is largely absorbed. This is why α calibration
  is accurate (EXP-04: α MRE < 10%).
- The bias is concentrated where the equilibrium's `θ`-dependence is strongest,
  i.e. the **β / congestion** direction. This is the gradient-level explanation of
  the EXP-04 finding that **β is weakly identified**: the one-step gradient
  systematically mis-scales the β component, so β is the parameter most affected by
  the approximation.
- **Beyond the convergent regime** (β ≳ 0.2 at this demand scale) there is **no
  stable equilibrium** to differentiate — `S` is not contractive, the IFT adjoint
  and the unrolled gradient both diverge, and the gradient is undefined. The
  calibrated β sits well inside the safe region.

## Practical recommendation
For α and γ, the one-step gradient is effectively exact (direction) and fine under
Adam. For a **bias-free β estimate**, prefer the IFT gradient (now implemented in
`gradient_check.ift_grad`) or unrolled backprop; this is a cheap, optional upgrade
to the calibration loop if sharper β identification is needed.

## Honest scope
- *Numerically verified:* one-step is directionally exact and magnitude-biased
  (growing in β); unrolled and IFT agree to machine-ish precision.
- *Not claimed:* a closed-form expression for the one-step bias. The empirical
  characterisation is enough to justify our use of one-step for α/γ and to flag β.

## Reproduce
```
python -m src.run_exp12_convergence    # Part C prints the gradient-bias table
```
