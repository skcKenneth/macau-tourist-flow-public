"""Data loading utilities for DSEC and MGTO tourism statistics.

See docs/04_data_sources.md for:
- Acquisition instructions (how to download official PDFs/XLSXs)
- Data schemas (column names, dtypes)
- Licensing and citation requirements

Data must be placed in data/raw/ and processed via ``src/ingest_data.py``
before these loaders can be used.  The loaders read from data/processed/
(cleaned parquet files) and validate the schema on load.

Typical workflow::

    # 1. Download DSEC Excel to data/raw/dsec/arrivals_YYYYMMDD.xlsx
    # 2. Run: python -m src.ingest_data --source dsec --input <path>
    # 3. Run: python -m src.ingest_data --source mgto --fallback
    # 4. In code:
    from src.utils.data_loader import load_arrivals_monthly, load_attraction_counts
    arrivals = load_arrivals_monthly()
    attractions = load_attraction_counts()
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths (relative to repo root)
# ---------------------------------------------------------------------------

DEFAULT_ARRIVALS_PATH = Path("data/processed/arrivals_monthly.parquet")
DEFAULT_ATTRACTIONS_PATH = Path("data/processed/attractions.parquet")

# ---------------------------------------------------------------------------
# Required columns for each schema
# ---------------------------------------------------------------------------

_ARRIVALS_REQUIRED_COLS = {"year_month", "transit_point", "count"}
_ATTRACTIONS_REQUIRED_COLS = {"node_id", "year", "annual_visitors", "confidence"}


# ---------------------------------------------------------------------------
# Schema validators (private)
# ---------------------------------------------------------------------------


def _validate_arrivals_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if the arrivals DataFrame is missing required columns."""
    missing = _ARRIVALS_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"arrivals_monthly.parquet is missing required columns: {missing}\n"
            f"Expected at minimum: {_ARRIVALS_REQUIRED_COLS}\n"
            f"Got: {set(df.columns)}\n"
            "Re-run: python -m src.ingest_data --source dsec --input <path>"
        )
    if df["count"].dtype.kind not in ("i", "u"):
        logger.warning(
            "arrivals 'count' column has dtype %s; expected integer. "
            "Casting to int64.", df["count"].dtype
        )


def _validate_attractions_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if the attractions DataFrame is missing required columns."""
    missing = _ATTRACTIONS_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"attractions.parquet is missing required columns: {missing}\n"
            f"Expected at minimum: {_ATTRACTIONS_REQUIRED_COLS}\n"
            f"Got: {set(df.columns)}\n"
            "Re-run: python -m src.ingest_data --source mgto [--fallback]"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_arrivals_monthly(
    path: Path | str = DEFAULT_ARRIVALS_PATH,
) -> pd.DataFrame:
    """Load processed monthly tourist arrivals data.

    Data source: DSEC (Statistics and Census Service, Macau SAR).
    Acquisition: see docs/04_data_sources.md §A.
    Ingest:  ``python -m src.ingest_data --source dsec --input <xlsx_path>``
    Citation: "Source: Statistics and Census Service (DSEC), Macau SAR"

    Schema:
        year_month    : pd.Period (freq="M") — first month of the period
        transit_point : str — "total" | "outer_harbour" | "border_gate" |
                        "airport" | "cotai_ferry" | "inner_harbour" | ...
        count         : int64 — number of visitor arrivals

    Args:
        path: Path to the processed parquet file.  Defaults to
            ``data/processed/arrivals_monthly.parquet``.

    Returns:
        DataFrame with at least the schema columns above.

    Raises:
        FileNotFoundError: If the parquet has not yet been generated.
            Run the ingest script first (see docs/04_data_sources.md §A).
        ValueError: If the parquet is missing required columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"DSEC arrivals data not found at {path}.\n"
            "Download the DSEC Excel file and run:\n"
            "  python -m src.ingest_data --source dsec --input data/raw/dsec/<file>.xlsx\n"
            "See docs/04_data_sources.md §A for full instructions."
        )

    df = pd.read_parquet(path)
    _validate_arrivals_schema(df)

    # Restore Period dtype if parquet serialised it as object
    if not isinstance(df["year_month"].dtype, pd.PeriodDtype):
        try:
            df["year_month"] = df["year_month"].dt.to_period("M")
        except AttributeError:
            df["year_month"] = df["year_month"].apply(
                lambda x: pd.Period(x, freq="M") if pd.notna(x) else x
            )

    df["count"] = df["count"].astype("int64")

    logger.info(
        "Loaded arrivals: %d rows, %d unique months, transit_points=%s.",
        len(df),
        df["year_month"].nunique(),
        sorted(df["transit_point"].unique()),
    )
    return df


def load_attraction_counts(
    path: Path | str = DEFAULT_ATTRACTIONS_PATH,
) -> pd.DataFrame:
    """Load processed per-attraction visitor counts.

    Data source: MGTO (Macao Government Tourism Office) Annual Reports / Yearbook,
    or synthetic proxy from ``src/utils/attractions.py`` (confidence="estimate").
    Acquisition: see docs/04_data_sources.md §B.
    Ingest:  ``python -m src.ingest_data --source mgto [--fallback]``
    Citation: "Source: Macao Government Tourism Office (MGTO) Yearbook [YYYY]"

    Schema:
        node_id         : str — matches AttractionNode.node_id
        year            : int
        annual_visitors : int64
        source          : str — bibliographic reference
        confidence      : str — "direct" | "estimated" | "estimate"

    Args:
        path: Path to the processed parquet file.  Defaults to
            ``data/processed/attractions.parquet``.

    Returns:
        DataFrame with at least the schema columns above.

    Raises:
        FileNotFoundError: If the parquet has not yet been generated.
            Run the ingest script first (see docs/04_data_sources.md §B).
        ValueError: If the parquet is missing required columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"MGTO attraction counts not found at {path}.\n"
            "Run:\n"
            "  python -m src.ingest_data --source mgto --fallback\n"
            "(or fill data/raw/mgto/attractions_manual.csv and omit --fallback)\n"
            "See docs/04_data_sources.md §B for full instructions."
        )

    df = pd.read_parquet(path)
    _validate_attractions_schema(df)

    df["year"] = df["year"].astype(int)
    df["annual_visitors"] = df["annual_visitors"].astype("int64")

    # Warn if all rows are synthetic estimates
    if (df["confidence"] == "estimate").all():
        logger.warning(
            "All attraction counts are synthetic estimates (confidence='estimate'). "
            "EXP-05 results should be caveated accordingly. "
            "See docs/04_data_sources.md §B for how to obtain official MGTO data."
        )

    logger.info(
        "Loaded attraction counts: %d rows, %d nodes, years=%s, confidence=%s.",
        len(df),
        df["node_id"].nunique(),
        sorted(df["year"].unique()),
        df["confidence"].value_counts().to_dict(),
    )
    return df
