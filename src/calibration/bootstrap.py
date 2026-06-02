"""Bootstrap confidence intervals for monthly MFG calibration (EXP-05).

Resamples training months with replacement, refits the model on each replicate,
and summarises uncertainty in the fitted parameters and held-out spatial MAE.
This quantifies how tightly the calibration is pinned by the limited monthly
training window (not a claim of external predictive validity).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import torch

logger = logging.getLogger(__name__)


def resample_train_months(
    train_data: list[dict[str, Any]],
    n_bootstrap: int,
    seed: int = 0,
) -> list[list[dict[str, Any]]]:
    """Draw ``n_bootstrap`` training sets by resampling months with replacement.

    Args:
        train_data: List of per-month dicts (``g``, ``obs``, ``month``, ...).
        n_bootstrap: Number of bootstrap replicates.
        seed: RNG seed.

    Returns:
        List of length ``n_bootstrap``, each element a resampled training list
        of the same length as ``train_data``.
    """
    if not train_data:
        raise ValueError("train_data must be non-empty for bootstrap.")
    n = len(train_data)
    gen = torch.Generator().manual_seed(seed)
    samples: list[list[dict[str, Any]]] = []
    for _ in range(n_bootstrap):
        idx = torch.randint(0, n, (n,), generator=gen).tolist()
        samples.append([train_data[i] for i in idx])
    return samples


def summarize_bootstrap(
    replicates: list[dict[str, Any]],
    attraction_count: int = 10,
    ci_percent: tuple[float, float] = (5.0, 95.0),
) -> dict[str, Any]:
    """Summarise bootstrap replicates into percentile CIs.

    Args:
        replicates: List of dicts, each with ``final_params`` (alpha, beta, gamma),
            ``val_mae``, and ``val_pred_share`` (length ``attraction_count``).
        attraction_count: Number of heritage attraction nodes in ``alpha``.
        ci_percent: Lower and upper percentiles for the CI band.

    Returns:
        Dict with scalar CIs for ``beta``, ``gamma``, ``val_mae`` and per-attraction
        arrays ``val_pred_p_low`` / ``val_pred_p_high`` for figure error bars.
    """
    lo, hi = ci_percent
    betas = np.array([r["final_params"]["beta"] for r in replicates])
    gammas = np.array([r["final_params"]["gamma"] for r in replicates])
    val_maes = np.array([r["val_mae"] for r in replicates])
    alpha_mat = np.array(
        [r["final_params"]["alpha"][:attraction_count] for r in replicates]
    )
    pred_mat = np.array([r["val_pred_share"] for r in replicates])

    def pct(x: np.ndarray, p: float) -> float:
        return float(np.percentile(x, p))

    def pct_vec(mat: np.ndarray, p: float) -> np.ndarray:
        return np.percentile(mat, p, axis=0)

    return {
        "n_bootstrap": len(replicates),
        "ci_percent": ci_percent,
        "beta_mean": float(betas.mean()),
        "beta_p_low": pct(betas, lo),
        "beta_p_high": pct(betas, hi),
        "gamma_mean": float(gammas.mean()),
        "gamma_p_low": pct(gammas, lo),
        "gamma_p_high": pct(gammas, hi),
        "val_mae_mean": float(val_maes.mean()),
        "val_mae_p_low": pct(val_maes, lo),
        "val_mae_p_high": pct(val_maes, hi),
        "alpha_p_low": pct_vec(alpha_mat, lo),
        "alpha_p_high": pct_vec(alpha_mat, hi),
        "val_pred_p_low": pct_vec(pred_mat, lo),
        "val_pred_p_high": pct_vec(pred_mat, hi),
    }


def run_bootstrap_calibration(
    train_data: list[dict[str, Any]],
    val_data: list[dict[str, Any]],
    fit_fn: Callable[[list[dict[str, Any]]], dict[str, Any]],
    n_bootstrap: int = 20,
    seed: int = 0,
    ci_percent: tuple[float, float] = (5.0, 95.0),
    attraction_count: int = 10,
    log_every: int = 5,
) -> dict[str, Any]:
    """Run month-resampling bootstrap and return a CI summary.

    Args:
        train_data: Full training-month list (resampled with replacement).
        val_data: Held-out validation months (fixed across replicates).
        fit_fn: Callable ``fit_fn(resampled_train) -> dict`` that must return
            ``final_params``, ``val_mae``, and ``val_pred_share`` (mean predicted
            attraction share on validation months, length ``attraction_count``).
        n_bootstrap: Number of bootstrap replicates.
        seed: RNG seed for resampling.
        ci_percent: Percentile band (default 5th-95th).
        attraction_count: Heritage attraction count.
        log_every: Log progress every this many replicates.

    Returns:
        Output of :func:`summarize_bootstrap` plus the raw ``replicates`` list.
    """
    samples = resample_train_months(train_data, n_bootstrap, seed=seed)
    replicates: list[dict[str, Any]] = []
    for i, td in enumerate(samples):
        rep = fit_fn(td)
        # Mean validation predicted share (for Fig02 error bars).
        if "val_rows" in rep:
            preds = torch.stack(
                [torch.as_tensor(r["pred"], dtype=torch.float32) for r in rep["val_rows"]]
            )
            rep["val_pred_share"] = preds.mean(dim=0)[:attraction_count].numpy()
        replicates.append(rep)
        if (i + 1) % log_every == 0 or i + 1 == n_bootstrap:
            logger.info(
                "Bootstrap %d/%d: val_mae=%.4f beta=%.5f",
                i + 1, n_bootstrap, rep["val_mae"], rep["final_params"]["beta"],
            )

    summary = summarize_bootstrap(
        replicates, attraction_count=attraction_count, ci_percent=ci_percent
    )
    summary["replicates"] = replicates
    return summary
