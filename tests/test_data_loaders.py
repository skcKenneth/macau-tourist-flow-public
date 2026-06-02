"""Unit tests for DSEC/MGTO data parsers and loader stubs.

All tests use synthetic in-memory data — no real downloaded files required.
This means the full test suite passes even before the user has downloaded
any DSEC or MGTO data.

Covers:
- ``src.utils.parse_dsec``: DSEC Excel wide-format parsing
- ``src.utils.parse_mgto``: MGTO manual CSV and synthetic fallback
- ``src.utils.data_loader``: schema validators + FileNotFoundError behaviour
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dsec_wide_xlsx(
    years: list[int],
    transit_point: str = "total",
    seed_count: int = 1_000_000,
) -> Path:
    """Write a minimal DSEC-style wide Excel file to a temp path and return it.

    Layout:
        Row 0: Title row ("Visitor Arrivals")
        Row 1: blank
        Row 2: "Year" | "Jan" | "Feb" | ... | "Dec"
        Row 3+: data rows (one per year, counts = seed_count + year*1000 + month*100)
    """
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    rows = [
        ["Visitor Arrivals"] + [""] * 12,  # title
        [""] * 13,                          # blank
        ["Year"] + months,                  # header
    ]
    for year in years:
        row = [str(year)] + [seed_count + year * 1000 + (m + 1) * 100 for m in range(12)]
        rows.append(row)

    df = pd.DataFrame(rows)
    tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
    tmp.close()
    df.to_excel(tmp.name, index=False, header=False)
    return Path(tmp.name)


def _make_mgto_manual_csv(rows: list[dict], extra_comments: bool = False) -> Path:
    """Write a minimal attractions_manual.csv to a temp path and return it."""
    lines = []
    if extra_comments:
        lines.append("# This is a comment line")
    lines.append("node_id,year,annual_visitors,source,confidence")
    for r in rows:
        lines.append(
            f"{r['node_id']},{r['year']},{r.get('annual_visitors', '')},"
            f"{r.get('source', 'MGTO Test')},{r.get('confidence', 'direct')}"
        )
    content = "\n".join(lines)
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8")
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# parse_dsec tests
# ---------------------------------------------------------------------------


class TestParseDsecXlsx:
    """Tests for src.utils.parse_dsec.parse_xlsx."""

    def test_wide_format_returns_correct_columns(self):
        """parse_xlsx returns DataFrame with year_month, transit_point, count."""
        from src.utils.parse_dsec import parse_xlsx

        path = _make_dsec_wide_xlsx([2022, 2023])
        df = parse_xlsx(path)

        assert "year_month" in df.columns
        assert "transit_point" in df.columns
        assert "count" in df.columns

    def test_wide_format_correct_row_count(self):
        """3 years × 12 months = 36 rows (single transit_point sheet)."""
        from src.utils.parse_dsec import parse_xlsx

        path = _make_dsec_wide_xlsx([2021, 2022, 2023])
        df = parse_xlsx(path)

        # One transit_point group × 3 years × 12 months
        assert len(df) == 36

    def test_strips_provisional_markers(self):
        """Numeric strings with 'p', 'r', '*' suffixes are parsed to int."""
        from src.utils.parse_dsec import _clean_number

        assert _clean_number("1234567p") == 1234567
        assert _clean_number("9876r") == 9876
        assert _clean_number("555*") == 555
        assert _clean_number("1,234,567") == 1234567
        assert _clean_number(None) is None
        assert _clean_number("") is None

    def test_count_dtype_is_int64(self):
        """count column must be int64."""
        from src.utils.parse_dsec import parse_xlsx

        path = _make_dsec_wide_xlsx([2023])
        df = parse_xlsx(path)

        assert df["count"].dtype == "int64"

    def test_year_month_is_period(self):
        """year_month column should contain pd.Period objects."""
        from src.utils.parse_dsec import parse_xlsx

        path = _make_dsec_wide_xlsx([2023])
        df = parse_xlsx(path)

        assert isinstance(df["year_month"].iloc[0], pd.Period)
        assert df["year_month"].iloc[0].freqstr == "M"

    def test_raises_on_missing_file(self):
        """FileNotFoundError for a non-existent path."""
        from src.utils.parse_dsec import parse_xlsx

        with pytest.raises(FileNotFoundError):
            parse_xlsx(Path("data/raw/dsec/does_not_exist.xlsx"))

    def test_all_12_months_present_per_year(self):
        """Each year in the output should have exactly 12 month rows."""
        from src.utils.parse_dsec import parse_xlsx

        path = _make_dsec_wide_xlsx([2022, 2023])
        df = parse_xlsx(path)

        for year in [2022, 2023]:
            mask = df["year_month"].apply(lambda p: p.year == year)
            assert mask.sum() == 12, f"Expected 12 rows for {year}, got {mask.sum()}"


# ---------------------------------------------------------------------------
# parse_mgto tests
# ---------------------------------------------------------------------------


class TestParseMgtoManualCsv:
    """Tests for src.utils.parse_mgto.parse_manual_csv."""

    def test_valid_csv_loads_correctly(self):
        """Valid CSV with known node_ids loads without error."""
        from src.utils.parse_mgto import parse_manual_csv

        rows = [
            {"node_id": "ruins_st_pauls", "year": 2022, "annual_visitors": 1_500_000,
             "source": "MGTO 2022", "confidence": "direct"},
            {"node_id": "senado_square", "year": 2022, "annual_visitors": 1_200_000,
             "source": "MGTO 2022", "confidence": "direct"},
        ]
        path = _make_mgto_manual_csv(rows)
        df = parse_manual_csv(path)

        assert len(df) == 2
        assert set(df["node_id"]) == {"ruins_st_pauls", "senado_square"}
        assert df["annual_visitors"].dtype == "int64"

    def test_comments_are_skipped(self):
        """Lines beginning with '#' should not appear in parsed data."""
        from src.utils.parse_mgto import parse_manual_csv

        rows = [
            {"node_id": "ruins_st_pauls", "year": 2021, "annual_visitors": 500_000,
             "source": "MGTO 2021", "confidence": "direct"},
        ]
        path = _make_mgto_manual_csv(rows, extra_comments=True)
        df = parse_manual_csv(path)

        assert len(df) == 1

    def test_unknown_node_id_raises_value_error(self):
        """A node_id not in ATTRACTION_IDS should raise ValueError."""
        from src.utils.parse_mgto import parse_manual_csv

        rows = [
            {"node_id": "unknown_attraction", "year": 2022, "annual_visitors": 100,
             "source": "test", "confidence": "direct"},
        ]
        path = _make_mgto_manual_csv(rows)

        with pytest.raises(ValueError, match="Unknown node_id"):
            parse_manual_csv(path)

    def test_blank_annual_visitors_rows_are_dropped(self):
        """Rows with blank annual_visitors should be silently dropped."""
        from src.utils.parse_mgto import parse_manual_csv

        rows = [
            {"node_id": "ruins_st_pauls", "year": 2022, "annual_visitors": 1_000_000,
             "source": "MGTO 2022", "confidence": "direct"},
            {"node_id": "ruins_st_pauls", "year": 2021, "annual_visitors": "",
             "source": "MGTO 2021", "confidence": "direct"},
        ]
        path = _make_mgto_manual_csv(rows)
        df = parse_manual_csv(path)

        # Only the row with a real value should survive
        assert len(df) == 1
        assert df.iloc[0]["year"] == 2022

    def test_raises_on_missing_file(self):
        """FileNotFoundError if the CSV path does not exist."""
        from src.utils.parse_mgto import parse_manual_csv

        with pytest.raises(FileNotFoundError):
            parse_manual_csv(Path("data/raw/mgto/does_not_exist.csv"))


class TestParseMgtoSyntheticFallback:
    """Tests for src.utils.parse_mgto.build_synthetic_from_attractions_py."""

    def test_covers_all_attraction_nodes(self):
        """10 attraction nodes must be present in the output."""
        from src.utils.attractions import ATTRACTION_IDS
        from src.utils.parse_mgto import build_synthetic_from_attractions_py

        df = build_synthetic_from_attractions_py(years=[2022])
        assert set(df["node_id"]) == set(ATTRACTION_IDS)
        assert len(df["node_id"].unique()) == len(ATTRACTION_IDS)

    def test_proportions_are_positive(self):
        """All annual_visitors values must be strictly positive integers."""
        from src.utils.parse_mgto import build_synthetic_from_attractions_py

        df = build_synthetic_from_attractions_py(years=[2022])
        assert (df["annual_visitors"] > 0).all()

    def test_confidence_is_estimate(self):
        """Synthetic fallback must tag all rows as confidence='estimate'."""
        from src.utils.parse_mgto import build_synthetic_from_attractions_py

        df = build_synthetic_from_attractions_py(years=[2022, 2023])
        assert (df["confidence"] == "estimate").all()

    def test_correct_row_count(self):
        """10 nodes × n_years rows expected."""
        from src.utils.attractions import ATTRACTION_IDS
        from src.utils.parse_mgto import build_synthetic_from_attractions_py

        years = [2019, 2020, 2021]
        df = build_synthetic_from_attractions_py(years=years)
        assert len(df) == len(ATTRACTION_IDS) * len(years)

    def test_no_transit_nodes_included(self):
        """Transit nodes (ferry_outer, border_gate, hotel_belt) must NOT appear."""
        from src.utils.attractions import TRANSIT_IDS
        from src.utils.parse_mgto import build_synthetic_from_attractions_py

        df = build_synthetic_from_attractions_py(years=[2022])
        assert not any(nid in df["node_id"].values for nid in TRANSIT_IDS)


# ---------------------------------------------------------------------------
# data_loader tests
# ---------------------------------------------------------------------------


class TestDataLoader:
    """Tests for src.utils.data_loader (schema validation + missing-file errors)."""

    def test_load_arrivals_raises_on_missing_file(self, tmp_path):
        """FileNotFoundError when arrivals parquet does not exist."""
        from src.utils.data_loader import load_arrivals_monthly

        fake_path = tmp_path / "arrivals_monthly.parquet"
        with pytest.raises(FileNotFoundError, match="DSEC arrivals data not found"):
            load_arrivals_monthly(path=fake_path)

    def test_load_arrivals_raises_on_missing_columns(self, tmp_path):
        """ValueError when parquet is missing required columns."""
        from src.utils.data_loader import load_arrivals_monthly

        # Write a parquet that is missing the 'count' column
        df_bad = pd.DataFrame({"year_month": ["2023-01"], "transit_point": ["total"]})
        bad_path = tmp_path / "arrivals_monthly.parquet"
        df_bad.to_parquet(bad_path, index=False)

        with pytest.raises(ValueError, match="missing required columns"):
            load_arrivals_monthly(path=bad_path)

    def test_load_arrivals_succeeds_with_valid_parquet(self, tmp_path):
        """load_arrivals_monthly returns DataFrame when parquet is valid."""
        from src.utils.data_loader import load_arrivals_monthly

        # Build a minimal valid parquet
        periods = pd.period_range("2022-01", periods=12, freq="M")
        df_ok = pd.DataFrame({
            "year_month": periods,
            "transit_point": "total",
            "count": range(1_000_000, 1_000_012),
        })
        ok_path = tmp_path / "arrivals_monthly.parquet"
        df_ok.to_parquet(ok_path, index=False)

        result = load_arrivals_monthly(path=ok_path)
        assert len(result) == 12
        assert result["count"].dtype == "int64"

    def test_load_attractions_raises_on_missing_file(self, tmp_path):
        """FileNotFoundError when attractions parquet does not exist."""
        from src.utils.data_loader import load_attraction_counts

        fake_path = tmp_path / "attractions.parquet"
        with pytest.raises(FileNotFoundError, match="MGTO attraction counts not found"):
            load_attraction_counts(path=fake_path)

    def test_load_attractions_raises_on_missing_columns(self, tmp_path):
        """ValueError when attractions parquet is missing required columns."""
        from src.utils.data_loader import load_attraction_counts

        df_bad = pd.DataFrame({"node_id": ["ruins_st_pauls"], "year": [2022]})
        bad_path = tmp_path / "attractions.parquet"
        df_bad.to_parquet(bad_path, index=False)

        with pytest.raises(ValueError, match="missing required columns"):
            load_attraction_counts(path=bad_path)

    def test_load_attractions_succeeds_with_valid_parquet(self, tmp_path):
        """load_attraction_counts returns DataFrame when parquet is valid."""
        from src.utils.data_loader import load_attraction_counts

        df_ok = pd.DataFrame({
            "node_id": ["ruins_st_pauls", "senado_square"],
            "year": [2022, 2022],
            "annual_visitors": [1_500_000, 1_200_000],
            "source": ["MGTO 2022", "MGTO 2022"],
            "confidence": ["direct", "direct"],
        })
        ok_path = tmp_path / "attractions.parquet"
        df_ok.to_parquet(ok_path, index=False)

        result = load_attraction_counts(path=ok_path)
        assert len(result) == 2
        assert result["annual_visitors"].dtype == "int64"
