"""Within-day arrival profiles g(t) for the MFG simulator.

IMPORTANT — what is data-derived vs assumed (see docs/08_validity_scope.md):
    The DSEC data is *monthly*. It fixes only the daily *volume* of arrivals at
    each transit node (see ``run_exp05._daily_source_counts``). It carries **no
    intra-day information**. The within-day *shape* of g(t) — when arrivals peak
    and how sharply — is therefore an **assumed modelling choice**, not something
    estimated from data. Every intra-day quantity (peak density, intervention
    peak-reduction magnitudes) inherits this assumption.

This module makes that assumption explicit and swappable: it provides a small
registry of plausible within-day profiles so that downstream code can test
whether conclusions are robust to the choice of profile (EXP-11), instead of
silently hard-coding a single Gaussian.

Each profile returns a 1-D tensor ``w`` of length ``T_steps`` of non-negative
within-day weights, normalised so that ``sum(w) * dt == 1`` (a discrete density
over the operating day). Callers scale it by the daily arrival count per node.

Time convention: ``t`` runs from 0 at the start of the operating day
(08:00) to ``(T_steps-1)*dt`` hours; the default horizon is 14 h (08:00–22:00).
"""

from __future__ import annotations

from typing import Callable

import torch


def _t_vec(T_steps: int, dt: float) -> torch.Tensor:
    """Time grid in hours since the start of the operating day."""
    return torch.linspace(0.0, (T_steps - 1) * dt, T_steps)


def _normalize(w: torch.Tensor, dt: float) -> torch.Tensor:
    """Normalise non-negative weights so that ``sum(w) * dt == 1``."""
    w = w.clamp(min=0.0)
    return w / (w.sum() * dt + 1e-12)


def gaussian(
    T_steps: int,
    dt: float,
    peak_time_hours: float = 2.0,
    sigma_hours: float = 1.5,
) -> torch.Tensor:
    """Single sharp peak (the project's original / default assumption).

    Models a crowd that builds to one peak (e.g. a sharp early-afternoon rush).
    Reproduces the historical hard-coded shape used in EXP-05/07/08 exactly.

    Args:
        T_steps: Number of discrete time steps in the operating day.
        dt: Time step in hours.
        peak_time_hours: Hour (since day start) of the arrival peak.
        sigma_hours: Gaussian standard deviation in hours (peak sharpness).
    """
    t = _t_vec(T_steps, dt)
    w = torch.exp(-0.5 * ((t - peak_time_hours) / sigma_hours) ** 2)
    return _normalize(w, dt)


def broad_midday_plateau(
    T_steps: int,
    dt: float,
    center_hours: float | None = None,
    half_width_hours: float = 4.0,
    edge_sigma_hours: float = 1.0,
) -> torch.Tensor:
    """Broad, flat-topped midday window (a difference of two sigmoids).

    Models steady arrivals across the middle of the day rather than one rush —
    a deliberately *un*-peaked alternative to the Gaussian.

    Args:
        T_steps: Number of discrete time steps.
        dt: Time step in hours.
        center_hours: Plateau centre; defaults to the middle of the day.
        half_width_hours: Half-width of the flat top in hours.
        edge_sigma_hours: Softness of the plateau edges in hours.
    """
    t = _t_vec(T_steps, dt)
    total_hours = (T_steps - 1) * dt
    center = center_hours if center_hours is not None else total_hours / 2.0
    left = torch.sigmoid((t - (center - half_width_hours)) / edge_sigma_hours)
    right = torch.sigmoid(((center + half_width_hours) - t) / edge_sigma_hours)
    w = left * right
    return _normalize(w, dt)


def double_peak(
    T_steps: int,
    dt: float,
    peak1_hours: float = 2.5,
    peak2_hours: float = 8.0,
    sigma_hours: float = 1.2,
    weight1: float = 0.5,
) -> torch.Tensor:
    """Two peaks: a morning and an afternoon rush separated by a midday lull.

    Models tour-group dynamics with distinct morning and afternoon waves.

    Args:
        T_steps: Number of discrete time steps.
        dt: Time step in hours.
        peak1_hours: Hour of the first (morning) peak.
        peak2_hours: Hour of the second (afternoon) peak.
        sigma_hours: Shared peak sharpness in hours.
        weight1: Mass fraction in the first peak (in [0, 1]).
    """
    t = _t_vec(T_steps, dt)
    g1 = torch.exp(-0.5 * ((t - peak1_hours) / sigma_hours) ** 2)
    g2 = torch.exp(-0.5 * ((t - peak2_hours) / sigma_hours) ** 2)
    w = weight1 * g1 + (1.0 - weight1) * g2
    return _normalize(w, dt)


def near_uniform(T_steps: int, dt: float) -> torch.Tensor:
    """Near-constant arrivals across the operating day (the un-peaked limit).

    A deliberate stress case: if the arrival rate is essentially flat, peaks are
    driven purely by the MFG dynamics rather than the arrival timing.

    Args:
        T_steps: Number of discrete time steps.
        dt: Time step in hours.
    """
    w = torch.ones(T_steps, dtype=torch.float32)
    return _normalize(w, dt)


def _interp_hours(t: torch.Tensor, anchor_hours, anchor_weights) -> torch.Tensor:
    """Piecewise-linear interpolation of (hour -> weight) anchors onto grid ``t``.

    ``t`` is clamped to the anchor support before interpolation so the curve is
    flat (held at the endpoint value) outside the anchored range.
    """
    xs = torch.as_tensor(anchor_hours, dtype=torch.float32)
    ys = torch.as_tensor(anchor_weights, dtype=torch.float32)
    tc = t.clamp(min=float(xs[0]), max=float(xs[-1]))
    idx = torch.searchsorted(xs, tc, right=True).clamp(1, xs.numel() - 1)
    x0, x1 = xs[idx - 1], xs[idx]
    y0, y1 = ys[idx - 1], ys[idx]
    frac = (tc - x0) / (x1 - x0 + 1e-12)
    return y0 + frac * (y1 - y0)


# Hourly proxy for a typical Macau heritage-core day-tripper inflow, expressed as
# relative weights at each hour since the 08:00 day start (t = 0 at 08:00). This is
# a PROXY shape, not fitted DSEC data: it follows the well-documented day-tripper
# pattern (Gongbei/Border-Gate crossings ramp through the late morning, plateau
# around midday-early afternoon, then taper toward the evening) and is consistent
# with the typical "Popular Times" envelope for Senado Square / Ruins of St. Paul's.
# It exists so EXP-11 can test conclusions against a realistic *asymmetric* shape,
# not just symmetric/synthetic ones. See docs/08_validity_scope.md.
_EMPIRICAL_PROXY_HOURS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
_EMPIRICAL_PROXY_WEIGHTS = [
    0.30, 0.60, 0.90, 1.00, 1.00, 0.95, 0.85, 0.75, 0.65, 0.50, 0.40, 0.30, 0.20, 0.12, 0.08,
]


def empirical_proxy(
    T_steps: int,
    dt: float,
    anchor_hours: list[float] | None = None,
    anchor_weights: list[float] | None = None,
) -> torch.Tensor:
    """Asymmetric within-day shape from an hourly day-tripper *proxy* curve.

    Interpolates a documented hourly inflow envelope (late-morning ramp, midday
    plateau, gradual afternoon decline) onto the simulation grid. This is the only
    profile grounded in an external real-world pattern rather than a synthetic
    functional form, so it is the most informative single robustness check: if a
    conclusion survives under this proxy as well as the synthetic shapes, it is not
    an artefact of the (symmetric) Gaussian assumption.

    The shape is still a PROXY (DSEC gives only monthly volume), so intra-day
    *magnitudes* remain assumption-dependent; only *relative* intervention effects
    are reported in EXP-11.

    Args:
        T_steps: Number of discrete time steps in the operating day.
        dt: Time step in hours.
        anchor_hours: Optional hour offsets (since day start) for custom anchors.
        anchor_weights: Optional relative weights at each ``anchor_hours`` point.
    """
    t = _t_vec(T_steps, dt)
    hours = anchor_hours if anchor_hours is not None else _EMPIRICAL_PROXY_HOURS
    wts = anchor_weights if anchor_weights is not None else _EMPIRICAL_PROXY_WEIGHTS
    if len(hours) != len(wts):
        raise ValueError("anchor_hours and anchor_weights must have equal length.")
    w = _interp_hours(t, hours, wts)
    return _normalize(w, dt)


# Registry of named profiles. All are ASSUMED shapes; ``empirical_proxy`` is
# grounded in an external hourly pattern, the others are synthetic functional forms.
PROFILES: dict[str, Callable[..., torch.Tensor]] = {
    "gaussian": gaussian,
    "broad_midday_plateau": broad_midday_plateau,
    "double_peak": double_peak,
    "near_uniform": near_uniform,
    "empirical_proxy": empirical_proxy,
}


def weights(name: str, T_steps: int, dt: float, **params) -> torch.Tensor:
    """Return the normalised within-day weight vector for a named profile.

    Args:
        name: Profile name; one of ``PROFILES``.
        T_steps: Number of discrete time steps in the operating day.
        dt: Time step in hours.
        **params: Profile-specific parameters (see each profile function).

    Returns:
        Float32 tensor of shape (T_steps,), non-negative, with ``sum * dt == 1``.

    Raises:
        ValueError: If ``name`` is not a registered profile.
    """
    if name not in PROFILES:
        raise ValueError(
            f"Unknown arrival profile {name!r}. Available: {sorted(PROFILES)}"
        )
    return PROFILES[name](T_steps, dt, **params)
