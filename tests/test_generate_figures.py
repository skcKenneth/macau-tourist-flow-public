"""Tests for the report figure-generation pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluation.generate_figures import (
    FIGURE_MAP,
    _build_results_table,
    _latest_exp_dir,
    generate_all_figures,
)

EXP_ROOT = Path("experiments")

# Experiment dirs needed to regenerate the headline figures asserted below. The
# public checkout commits only the synthetic EXP-01..04 summaries (EXP-05+ outputs
# are gitignored and require the full data pipeline), so this test must skip rather
# than fail when those source dirs are absent.
_REQUIRED_FIGURE_SOURCES = ("EXP-01", "EXP-05", "EXP-07", "EXP-09", "EXP-10", "EXP-11", "EXP-12")


def _figure_sources_present() -> bool:
    return EXP_ROOT.exists() and all(
        _latest_exp_dir(EXP_ROOT, pat) is not None for pat in _REQUIRED_FIGURE_SOURCES
    )


def test_latest_exp_dir_picks_newest(tmp_path):
    (tmp_path / "20260101_EXP-99_demo").mkdir()
    (tmp_path / "20260601_EXP-99_demo").mkdir()
    (tmp_path / "20260315_EXP-99_demo").mkdir()
    latest = _latest_exp_dir(tmp_path, "EXP-99")
    assert latest is not None and latest.name == "20260601_EXP-99_demo"


def test_latest_exp_dir_none_when_missing(tmp_path):
    assert _latest_exp_dir(tmp_path, "EXP-404") is None


def test_generate_all_figures_produces_key_outputs(tmp_path):
    """Over the real experiment dirs, the high-value figures + table are produced."""
    if not _figure_sources_present():
        pytest.skip(
            "headline figure source experiments (EXP-05+) not present; "
            "run the full pipeline to regenerate (see README 'Reproduce')"
        )

    out = tmp_path / "figures"
    result = generate_all_figures(EXP_ROOT, out, dpi=300)

    # The headline figures every report needs (present in the committed experiment set).
    for name in (
        "Fig01_macau_graph", "Fig02_calibration_fit", "Fig05_metering_pareto",
        "Fig08_combined_pareto", "Fig09_compliance", "Fig10_baselines",
        "Fig11_validity_peak_reduction", "Fig14_contraction_damped",
    ):
        assert (out / f"{name}.pdf").exists(), f"missing {name}.pdf"
        assert (out / f"{name}.png").exists(), f"missing {name}.png"

    # Master results table.
    assert (out / "results_table.md").exists()
    assert (out / "results_table.csv").exists()
    table = (out / "results_table.md").read_text(encoding="utf-8")
    assert "Gravity" in table and "0.018" in table

    # Most of the inventory should resolve (allow a couple of missing if a dir was pruned).
    assert len(result["copied"]) >= len(FIGURE_MAP) - 2


def test_results_table_standalone(tmp_path):
    if not EXP_ROOT.exists():
        pytest.skip("no experiments/ directory present")
    _build_results_table(EXP_ROOT, tmp_path)
    assert (tmp_path / "results_table.csv").exists()
    rows = (tmp_path / "results_table.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0] == "experiment,metric,value"
    assert len(rows) >= 10
