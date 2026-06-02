"""Tests for calibration bootstrap utilities (no full EXP-05 run)."""

from __future__ import annotations

import numpy as np

from src.calibration.bootstrap import (
    resample_train_months,
    run_bootstrap_calibration,
    summarize_bootstrap,
)


def _fake_train(n: int = 5):
    return [{"month": f"2024-{i:02d}", "g": None, "obs": None} for i in range(1, n + 1)]


def test_resample_train_months_same_length_and_reproducible():
    data = _fake_train(6)
    a = resample_train_months(data, 10, seed=0)
    b = resample_train_months(data, 10, seed=0)
    assert len(a) == 10 and all(len(s) == 6 for s in a)
    assert [m["month"] for m in a[0]] == [m["month"] for m in b[0]]


def test_summarize_bootstrap_percentiles_ordered():
    reps = [
        {
            "final_params": {"alpha": [1.0, 2.0], "beta": 0.01 + i * 0.001, "gamma": 1e-5},
            "val_mae": 0.02 + i * 0.001,
            "val_pred_share": np.array([0.2, 0.3]) + i * 0.01,
        }
        for i in range(20)
    ]
    s = summarize_bootstrap(reps, attraction_count=2, ci_percent=(5.0, 95.0))
    assert s["beta_p_low"] <= s["beta_mean"] <= s["beta_p_high"]
    assert s["val_mae_p_low"] <= s["val_mae_mean"] <= s["val_mae_p_high"]
    assert len(s["val_pred_p_low"]) == 2


def test_run_bootstrap_calibration_with_toy_fit_fn():
    calls = []

    def fit_fn(train):
        calls.append(len(train))
        return {
            "final_params": {"alpha": [1.0] * 10, "beta": 0.01, "gamma": 1e-5},
            "val_mae": 0.02,
            "val_rows": [{"pred": np.full(10, 0.1)}],
        }

    summary = run_bootstrap_calibration(
        _fake_train(4),
        [],
        fit_fn=fit_fn,
        n_bootstrap=8,
        seed=1,
        attraction_count=10,
        log_every=100,
    )
    assert len(calls) == 8
    assert summary["n_bootstrap"] == 8
    assert "val_mae_p_low" in summary
