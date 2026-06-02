# 03 — Methodology

> Canonical specification of the mathematical model and the ML calibration pipeline. **Single source of truth** for notation. If anything here changes, update it here first, then propagate.

## Notation

| Symbol | Meaning | Units |
|---|---|---|
| $G = (V, E)$ | Graph of attractions and corridors | — |
| $v \in V$ | Node (attraction or transit point) | — |
| $(u,v) \in E$ | Directed edge with weight $w_{uv}$ | meters |
| $t \in [0, T]$ | Time | hours, $T = 14$ (8am–10pm) |
| $\rho_v(t)$ | Mass of tourists at node $v$ at time $t$ | dimensionless, $\sum_v \rho_v(t) + \rho_{\text{out}}(t) = 1$ |
| $g_v(t)$ | Exogenous arrival rate at node $v$ | hr⁻¹ |
| $u_v(t)$ | Cost-to-go for a tourist at $v$ at $t$ | utility units |
| $\alpha_v$ | Intrinsic attractiveness of node $v$ | utility |
| $\beta$ | Congestion sensitivity | utility per density |
| $\gamma$ | Walking cost per meter | utility per meter |
| $m^*_{uv}(t)$ | Equilibrium movement rate from $u$ to $v$ | hr⁻¹ |

## Phase 1 — Graph construction

### Nodes
Heritage attractions (initial set, refine after lit review):
1. Ruins of St. Paul's (大三巴牌坊)
2. Senado Square (議事亭前地)
3. A-Ma Temple (媽閣廟)
4. St. Dominic's Church (玫瑰堂)
5. Mount Fortress (大炮台)
6. Lou Kau Mansion (盧家大屋)
7. St. Lawrence's Church (聖老楞佐教堂)
8. St. Joseph's Seminary (聖若瑟修院)
9. Lilau Square (亞婆井前地)
10. Mandarin's House (鄭家大屋)

Transit/source nodes:
- Outer Harbour Ferry Terminal
- Macau Border Gate
- Hotel-belt aggregate node

### Edges
Walking corridors as the pedestrian-accessible street network, extracted via `osmnx` with filter `walk`. Edge weight = shortest-walking-distance.

## Phase 2 — Mean field game formulation

### Setup
Each tourist starts at a source $s$, has a time budget $T$, and accumulates utility:
$$
J(\text{path}) = \sum_v \alpha_v \cdot \mathbb{1}[\text{visited } v] - \int_0^T \left[\beta \rho_{v(t)}(t) + \gamma \dot{\ell}(t)\right] dt
$$

### Backward HJB on graph
For each $v$, the cost-to-go $u_v(t)$ satisfies:
$$
-\partial_t u_v(t) = \max_{w: (v,w) \in E} \left\{ \alpha_w - \beta \rho_w(t) - \gamma w_{vw} + u_w(t + \tau_{vw}) \right\} - u_v(t)
$$
with terminal $u_v(T) = 0$.

(The $\max$ is the optimal next-edge choice; $\tau_{vw}$ is the walking time along edge.)

### Forward Fokker-Planck
Movement rate from $u$ to $v$ at time $t$ under the optimal policy:
$$
m^*_{uv}(t) = \rho_u(t) \cdot \pi^*_{uv}(t)
$$
where $\pi^*_{uv}(t)$ is the softmax of the HJB Q-values (smoothed for differentiability):
$$
\pi^*_{uv}(t) = \frac{\exp\left( \frac{1}{\epsilon}[\alpha_v - \beta\rho_v - \gamma w_{uv} + u_v(t+\tau_{uv})]\right)}{\sum_{w} \exp(\cdots)}
$$

Density evolves as:
$$
\dot{\rho}_v(t) = g_v(t) + \sum_{u: (u,v) \in E} m^*_{uv}(t - \tau_{uv}) - \sum_{w: (v,w) \in E} m^*_{vw}(t)
$$

### Equilibrium
Nash equilibrium = simultaneous solution of HJB (backward) and FP (forward). Solved by **fixed-point iteration**:
1. Initialize $\rho^{(0)}_v(t)$ uniform.
2. Solve HJB backward given $\rho^{(k)}$.
3. Solve FP forward using policy from HJB.
4. Update $\rho^{(k+1)}$, repeat until $\|\rho^{(k+1)} - \rho^{(k)}\|_\infty < \delta$.

### Discretization
- Time step $\Delta t = 5$ minutes ($t = 0, \ldots, 168$ steps for 14-hour day).
- All quantities stored as `torch.Tensor` of shape `(T, V)` for vectorization.

## Phase 3 — Parameter calibration

### Unknowns to learn
- $\alpha_v$ for each node (10–15 parameters)
- $\beta$ congestion sensitivity (scalar)
- $\gamma$ walking cost (scalar)
- $g_v(t)$ arrival profile (parameterized as piecewise linear or small NN)

### Observations
- Average $\rho_v(t)$ during peak vs off-peak hours, from MGTO published attraction counts.
- Total daily visitors per attraction (DSEC).

### Loss function
$$
\mathcal{L}(\theta) = \sum_{v, t \in \text{obs}} \left( \hat{\rho}_v(t) - \rho_v(t; \theta) \right)^2 + \lambda \cdot \text{Reg}(\theta)
$$

where $\rho_v(t; \theta)$ is the MFG equilibrium under parameters $\theta$, and $\text{Reg}$ keeps $\alpha_v \ge 0$, $\beta \ge 0$, etc.

### Optimization
- Differentiable forward simulator (PyTorch).
- Adam optimizer, lr $10^{-3}$, gradient clipping at 1.0.
- Warm-start with classical least-squares fit on simplified model.

### Validation
- Hold out one month of data; check MAE on held-out attraction densities.
- Sensitivity analysis: vary each parameter ±20%, observe effect on equilibrium.

## Phase 4 — Intervention optimization

### Intervention types
1. **Entrance scheduling**: Modify $g_v(t)$ by metering arrivals at the ferry terminal.
2. **Routing recommendation**: Bias the policy $\pi^*$ at certain nodes via informational nudges, modeled as additive bonus $\eta_{uv}(t)$.
3. **Combined**.

### Objective
Minimize peak density at bottleneck nodes $B \subset V$:
$$
\min_{\text{intervention}} \max_{v \in B, t} \rho_v(t)
$$
subject to "fairness" (total attractions visited per tourist not degraded by more than 10%).

### Method
Gradient descent through the differentiable simulator on intervention parameters; Pareto front of (peak density, average attractions visited).

## Fallback plans

If the full MFG proves too computationally expensive within 12 weeks:

**Fallback A**: Replace MFG with **best-response simulation** — agents myopically choose the highest-utility next attraction given current density (no full HJB backward solve). Loses theoretical "Nash equilibrium" framing but keeps quantitative results.

**Fallback B**: Replace MFG with **time-inhomogeneous Markov chain** on graph, calibrated to match observed flows. Loses individual optimization but is well-defined and fast.

Both fallbacks share calibration pipeline and intervention framework.

## Open methodological questions
- [ ] Should arrivals be deterministic or stochastic ($g_v(t)$ as expected rate vs Poisson process)?
- [ ] How to handle return trips (tourists leaving the heritage area)?
- [ ] What's the right time horizon for HJB? (Single visit ≈ 4 hr, full day ≈ 14 hr.)
- [ ] How to handle group dynamics (tourists travel in clusters, not individuals)?
