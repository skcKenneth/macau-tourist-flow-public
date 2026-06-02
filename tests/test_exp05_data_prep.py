from __future__ import annotations

import pandas as pd
import pytest
import torch

from src.run_exp05 import (
    _build_real_arrival_tensor,
    _daily_source_counts,
    _days_in_month,
    _map_dsec_transit_counts,
    _observed_distribution_for_year,
    _select_months,
)


def _arrivals_df() -> pd.DataFrame:
    rows = []
    for month in ["2024-01", "2024-02", "2024-03"]:
        for transit_point, count in {
            "total": 1000,
            "border_gate": 300,
            "by_land": 500,
            "by_sea": 400,
            "outer_harbour": 100,
            "taipa_ferry": 200,
            "inner_harbour": 50,
            "by_air": 100,
        }.items():
            rows.append({
                "year_month": pd.Period(month, freq="M"),
                "transit_point": transit_point,
                "count": count,
            })
    return pd.DataFrame(rows)


def test_dsec_transit_mapping_does_not_double_count_aggregates():
    df = _arrivals_df()
    mapped = _map_dsec_transit_counts(
        df[df["year_month"] == pd.Period("2024-01", freq="M")]
    )
    assert mapped["border_gate"] == 300
    assert mapped["ferry_outer"] == 400
    assert mapped["hotel_belt"] == 300
    assert sum(mapped.values()) == 1000


def test_monthly_arrivals_convert_to_daily_scale():
    daily = _daily_source_counts(_arrivals_df(), "2024-02", population_scale=0.5)
    assert daily["border_gate"] == pytest.approx((300 / 29) * 0.5)
    assert _days_in_month("2024-02") == 29


def test_arrival_tensor_preserves_daily_total_after_integration():
    node_order = ["a0", "a1", "border_gate", "ferry_outer", "hotel_belt"]
    counts = {"border_gate": 10.0, "ferry_outer": 20.0, "hotel_belt": 5.0}
    g = _build_real_arrival_tensor(
        node_order=node_order,
        T_steps=12,
        dt=0.5,
        daily_source_counts=counts,
        peak_time_hours=2.0,
        sigma_hours=1.0,
    )
    assert float((g.sum(dim=0).sum() * 0.5).item()) == pytest.approx(35.0)


def test_month_split_is_deterministic_and_available_only():
    months = _select_months(_arrivals_df(), "2024-01", "2024-12")
    assert months == [
        pd.Period("2024-01", freq="M"),
        pd.Period("2024-02", freq="M"),
        pd.Period("2024-03", freq="M"),
    ]


def test_observed_distribution_sums_to_one_and_excludes_transit():
    node_order = ["a", "b", "c", "d", "t1", "t2"]
    df = pd.DataFrame({
        "node_id": ["a", "b", "c", "d", "t1", "t2"],
        "year": [2024] * 6,
        "annual_visitors": [10, 20, 30, 40, 999, 999],
        "confidence": ["estimate"] * 6,
    })
    # Override module-level attraction count expectation for this focused test.
    import src.run_exp05 as exp05

    old_count = exp05.ATTRACTION_COUNT
    exp05.ATTRACTION_COUNT = 4
    try:
        obs = _observed_distribution_for_year(df, node_order, 2024)
    finally:
        exp05.ATTRACTION_COUNT = old_count
    assert obs.shape == (4,)
    assert float(obs.sum().item()) == pytest.approx(1.0)
    assert torch.allclose(obs, torch.tensor([0.1, 0.2, 0.3, 0.4]))
