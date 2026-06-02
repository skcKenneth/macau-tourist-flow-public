"""Parse MGTO (Macao Government Tourism Office) per-attraction visitor counts.

Two data paths:

1. **Manual CSV** (``data/raw/mgto/attractions_manual.csv``): the user fills in
   annual visitor counts extracted from MGTO Annual Reports or the Macao Yearbook
   Tourism chapter.  Use ``parse_manual_csv()`` to load this file.

2. **Synthetic fallback** (``build_synthetic_from_attractions_py()``): uses the
   ``annual_visitors_est`` already encoded in ``src/utils/attractions.py`` as a
   proxy.  All attraction nodes get constant proportions across years (no
   temporal variation modelled). ``confidence`` is set to ``"estimate"``.

Both paths return a DataFrame with the same schema:
    node_id         : str — matches AttractionNode.node_id
    year            : int
    annual_visitors : int64
    source          : str — bibliographic note
    confidence      : str — "direct" | "estimated" | "estimate"

Acquisition instructions: see ``docs/04_data_sources.md §B``.

Citation: "Source: Macao Government Tourism Office (MGTO) Yearbook [YYYY]"
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils.attractions import ATTRACTION_IDS, NODE_BY_ID

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MGTO_CSV_COLUMNS: list[str] = [
    "node_id",
    "year",
    "annual_visitors",
    "source",
    "confidence",
]

_VALID_CONFIDENCE = frozenset({"direct", "estimated", "estimate"})

# Default path to the user-filled CSV template
DEFAULT_MANUAL_CSV = Path("data/raw/mgto/attractions_manual.csv")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_manual_csv(path: Path | str = DEFAULT_MANUAL_CSV) -> pd.DataFrame:
    """Load the user-filled MGTO attractions CSV.

    Validates:
    - Required columns present.
    - ``node_id`` values are recognised attraction nodes (from ``attractions.py``).
    - ``annual_visitors`` is non-negative where present.
    - Rows with blank ``annual_visitors`` are dropped with a warning.

    Args:
        path: Path to ``attractions_manual.csv``.

    Returns:
        Clean DataFrame with schema: node_id (str), year (int),
        annual_visitors (int64), source (str), confidence (str).

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If required columns are missing or invalid node_ids found.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"MGTO manual CSV not found: {path}\n"
            "Run the ingest script with --fallback to use the synthetic proxy instead,\n"
            "or fill in the template and re-run."
        )

    # Skip comment lines beginning with '#'
    df = pd.read_csv(path, comment="#", dtype=str)

    # Validate required columns
    missing_cols = set(MGTO_CSV_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"attractions_manual.csv is missing required columns: {missing_cols}\n"
            f"Expected columns: {MGTO_CSV_COLUMNS}"
        )

    # Validate node_ids
    unknown = set(df["node_id"].dropna()) - set(ATTRACTION_IDS)
    if unknown:
        raise ValueError(
            f"Unknown node_id values in attractions_manual.csv: {unknown}\n"
            f"Valid ids: {ATTRACTION_IDS}"
        )

    # Drop rows where annual_visitors is blank/missing
    before = len(df)
    df = df[df["annual_visitors"].notna() & (df["annual_visitors"].str.strip() != "")]
    dropped = before - len(df)
    if dropped:
        logger.info(
            "Dropped %d rows with missing annual_visitors from %s.", dropped, path.name
        )

    if df.empty:
        raise ValueError(
            f"No complete rows found in {path}.\n"
            "Fill in the annual_visitors column and re-run, or use --fallback."
        )

    # Cast types
    df["year"] = df["year"].astype(int)
    df["annual_visitors"] = df["annual_visitors"].str.replace(",", "", regex=False).astype("int64")
    df["source"] = df["source"].fillna("MGTO Annual Report (year unknown)")
    df["confidence"] = df["confidence"].fillna("direct")

    # Warn on unrecognised confidence values (don't error — user might use their own tags)
    bad_confidence = set(df["confidence"]) - _VALID_CONFIDENCE
    if bad_confidence:
        logger.warning(
            "Unrecognised confidence value(s) in %s: %s. "
            "Expected one of %s.",
            path.name, bad_confidence, _VALID_CONFIDENCE,
        )

    # Validate non-negative counts
    negative_mask = df["annual_visitors"] < 0
    if negative_mask.any():
        raise ValueError(
            f"Negative annual_visitors values found in {path}:\n"
            f"{df[negative_mask][['node_id', 'year', 'annual_visitors']]}"
        )

    logger.info(
        "Loaded %d attraction-year rows from %s (confidence: %s).",
        len(df),
        path.name,
        df["confidence"].value_counts().to_dict(),
    )

    return df[MGTO_CSV_COLUMNS].reset_index(drop=True)


def build_synthetic_from_attractions_py(
    years: list[int] | None = None,
) -> pd.DataFrame:
    """Build a synthetic per-attraction DataFrame from ``attractions.py`` estimates.

    Uses ``annual_visitors_est`` already encoded in ``AttractionNode`` as a
    proportional proxy.  All years receive the *same* visitor proportions
    (no temporal variation modelled).  This is sufficient to calibrate alpha
    parameters for EXP-05 when official MGTO per-attraction data is unavailable.

    Args:
        years: List of years to generate rows for.  Defaults to 2019–2024.

    Returns:
        DataFrame with schema: node_id (str), year (int), annual_visitors (int64),
        source (str), confidence ("estimate").
    """
    if years is None:
        years = list(range(2019, 2025))

    records = []
    for node_id in ATTRACTION_IDS:
        node = NODE_BY_ID[node_id]
        for year in years:
            records.append(
                {
                    "node_id": node_id,
                    "year": year,
                    "annual_visitors": node.annual_visitors_est,
                    "source": "src/utils/attractions.py (annual_visitors_est)",
                    "confidence": "estimate",
                }
            )

    df = pd.DataFrame(records)
    df["annual_visitors"] = df["annual_visitors"].astype("int64")
    logger.info(
        "Built synthetic MGTO fallback: %d nodes × %d years = %d rows.",
        len(ATTRACTION_IDS),
        len(years),
        len(df),
    )
    return df[MGTO_CSV_COLUMNS].reset_index(drop=True)


def load_or_build(
    csv_path: Path | str = DEFAULT_MANUAL_CSV,
    fallback: bool = True,
    years: list[int] | None = None,
) -> pd.DataFrame:
    """Load MGTO data from manual CSV, falling back to synthetic if unavailable.

    Tries ``parse_manual_csv(csv_path)`` first.  If the file is missing or
    contains no complete rows and ``fallback=True``, returns
    ``build_synthetic_from_attractions_py(years)`` instead.

    Args:
        csv_path: Path to the user-filled attractions CSV.
        fallback: If True, use synthetic data when manual CSV unavailable.
        years: Years to generate in fallback mode.

    Returns:
        DataFrame with MGTO_CSV_COLUMNS schema.

    Raises:
        FileNotFoundError: If ``fallback=False`` and the CSV does not exist.
        ValueError: If ``fallback=False`` and the CSV has no valid rows.
    """
    csv_path = Path(csv_path)
    try:
        df = parse_manual_csv(csv_path)
        logger.info("Using manual MGTO CSV: %d rows loaded.", len(df))
        return df
    except (FileNotFoundError, ValueError) as exc:
        if not fallback:
            raise
        logger.info(
            "Manual MGTO CSV unavailable (%s). Using synthetic fallback.", exc
        )
        return build_synthetic_from_attractions_py(years=years)
