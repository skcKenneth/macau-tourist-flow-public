"""Parse DSEC (Statistics and Census Service) visitor arrivals data.

Supports two input formats:

1. **Monthly fast-release Excel** (``E_MV_FR_YYYY_MNN.xlsx`` or
   ``CPE_MV_FR_YYYY_MNN.xlsx``) — the format downloaded from the DSEC portal.
   Each file covers *one* month and contains many sheets, one per transit point.
   Each sheet also carries the *prior-year same month* in a comparison column,
   so a single file yields two months of data.
   → Use ``parse_monthly_fast_release()`` or ``parse_all_monthly_fast_release()``.

2. **Wide annual Excel** — a single file with one row per year and 12 month
   columns (the legacy "main table" format sometimes published separately).
   → Use ``parse_xlsx()``.

3. **Quarterly PDF fast-release** (backup, requires ``pdfplumber``).
   → Use ``parse_quarterly_pdf()``.

All parsers return a long-format DataFrame with columns:
    year_month   : pd.Period (freq="M")
    transit_point: str — "total" | "outer_harbour" | "taipa_ferry" |
                   "inner_harbour" | "border_gate" | "by_air" | "by_sea" |
                   "by_land"
    count        : int64 (number of visitor arrivals)

Acquisition instructions: see ``docs/04_data_sources.md §A``.

Citation: "Source: Statistics and Census Service (DSEC), Macau SAR"
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MONTH_NAMES: list[str] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

_YEAR_PATTERN = re.compile(r"^\s*(\d{4})\s*(/\d+)?\s*$")
_STRIP_MARKERS = re.compile(r"[pPrRe*,\s]+")

# Filename pattern: E_MV_FR_2025_M01.xlsx  or  CPE_MV_FR_2026_M04.xlsx
_FILENAME_PATTERN = re.compile(r"(?:CPE_|E_)?MV_FR_(\d{4})_M(\d{2})", re.IGNORECASE)

# Sheet number → transit_point slug for monthly fast-release files
_SHEET_NUM_TO_TRANSIT: dict[str, str] = {
    "1":  "total",          # All transit points combined
    "2":  "by_sea",         # All sea routes
    "2a": "outer_harbour",  # Outer Harbour Ferry Terminal
    "2b": "taipa_ferry",    # Taipa (Cotai) Ferry Terminal
    "2c": "inner_harbour",  # Inner Harbour Ferry Terminal
    "3":  "by_land",        # All land crossings
    "3a": "border_gate",    # Border Gate (關閘)
    "4":  "by_air",         # Airport
}

# Rows (0-based) in each monthly fast-release sheet
_ROW_DATE = 5    # date cells: col 3 = current month, col 5 = prior year same month
_ROW_TOTAL = 9   # 總數 / Total row: col 3 = current count, col 5 = prior year count
_COL_CURRENT = 3
_COL_PRIOR   = 5

# Legacy wide-format: sheet name → transit_point slug
_SHEET_TRANSIT_MAP: dict[str, str] = {
    "outer harbour": "outer_harbour",
    "outer harbour ferry terminal": "outer_harbour",
    "inner harbour": "inner_harbour",
    "inner harbour ferry terminal": "inner_harbour",
    "border gate": "border_gate",
    "border gate crossing": "border_gate",
    "airport": "by_air",
    "cotai ferry terminal": "taipa_ferry",
    "taipa ferry terminal": "taipa_ferry",
    "heliport": "heliport",
    "total": "total",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_number(raw: object) -> int | None:
    """Strip DSEC provisional markers and return int, or None if unparseable."""
    if pd.isna(raw):
        return None
    text = _STRIP_MARKERS.sub("", str(raw))
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _parse_date_cell(cell: object) -> pd.Period | None:
    """Convert a DSEC date cell to a monthly Period.

    DSEC stores dates as Excel datetime objects (e.g. 2025-01-01 00:00:00)
    or as strings like "2025-01-01".
    """
    if pd.isna(cell):
        return None
    if isinstance(cell, (pd.Timestamp,)):
        return pd.Period(year=cell.year, month=cell.month, freq="M")
    # Try parsing a string like "2025-01-01"
    try:
        ts = pd.Timestamp(str(cell))
        return pd.Period(year=ts.year, month=ts.month, freq="M")
    except Exception:
        return None


def _filename_to_period(path: Path) -> pd.Period | None:
    """Extract year/month from a DSEC monthly filename as a fallback."""
    m = _FILENAME_PATTERN.search(path.stem)
    if m:
        return pd.Period(year=int(m.group(1)), month=int(m.group(2)), freq="M")
    return None


# ---------------------------------------------------------------------------
# Monthly fast-release parser (primary format)
# ---------------------------------------------------------------------------


def _extract_period_count_pairs(
    df: pd.DataFrame,
    period_from_filename: pd.Period | None,
    include_prior_year: bool,
) -> list[dict]:
    """Scan row _ROW_DATE for date cells; match to row _ROW_TOTAL counts.

    DSEC monthly files have inconsistent column ordering:
    - M01: [current, NaN, prior_year, NaN, change]
    - M02: [current, prior_year, change, current_YTD, NaN, prior_YTD, ...]
    - M03+: [prior_year, current, change, prior_YTD, NaN, current_YTD, ...]

    Strategy: scan all columns in row _ROW_DATE for datetime objects.  Use the
    filename period to identify which column contains the current month and which
    contains the prior-year same month.  Ignore cumulative YTD columns (they
    repeat dates but with different counts).  Return only the monthly columns.
    """
    if period_from_filename is None:
        return []

    prior_period = pd.Period(
        year=period_from_filename.year - 1,
        month=period_from_filename.month,
        freq="M",
    )

    # Collect (col_index, period) for every datetime cell in the date row
    col_period_pairs: list[tuple[int, pd.Period]] = []
    for col_idx in range(len(df.columns)):
        p = _parse_date_cell(df.iloc[_ROW_DATE, col_idx])
        if p is not None:
            col_period_pairs.append((col_idx, p))

    # Separate the first occurrence of current and prior-year periods
    # (second occurrences are the cumulative YTD section — skip them)
    seen: set[pd.Period] = set()
    monthly_cols: list[tuple[int, pd.Period]] = []
    for col_idx, p in col_period_pairs:
        if p in (period_from_filename, prior_period) and p not in seen:
            monthly_cols.append((col_idx, p))
            seen.add(p)

    records = []
    for col_idx, period in monthly_cols:
        is_prior = period == prior_period
        if is_prior and not include_prior_year:
            continue
        count = _clean_number(df.iloc[_ROW_TOTAL, col_idx])
        if count is not None:
            records.append({"year_month": period, "count": count})

    return records


def parse_monthly_fast_release(
    path: Path | str,
    include_prior_year: bool = True,
) -> pd.DataFrame:
    """Parse one DSEC monthly fast-release Excel file.

    File naming convention: ``E_MV_FR_YYYY_MNN.xlsx`` (English) or
    ``CPE_MV_FR_YYYY_MNN.xlsx`` (Chinese/Portuguese/English combined).

    Each file contains sheets numbered '1', '2', '2a', '2b', '2c', '3',
    '3a', '4', … Each sheet covers one transit point.  Row 9 (0-based) is
    the 總數 / Total row.  The column layout varies by month — the parser
    identifies current vs prior-year columns from their date cells rather
    than relying on fixed column indices.

    Args:
        path: Path to the ``.xlsx`` file.
        include_prior_year: If True (default), also extract the prior-year
            same-month count.  This gives two rows per sheet at no extra
            download cost.

    Returns:
        Long-format DataFrame: year_month (Period[M]), transit_point (str),
        count (int64).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DSEC monthly file not found: {path}")

    period_from_filename = _filename_to_period(path)
    if period_from_filename is None:
        logger.warning(
            "Could not determine period from filename %s. "
            "File will be skipped if date cells are also unparseable.",
            path.name,
        )

    records: list[dict] = []

    xf = pd.ExcelFile(path)
    available_sheets = set(xf.sheet_names)

    for sheet_num, transit_point in _SHEET_NUM_TO_TRANSIT.items():
        sn = None
        for candidate in [sheet_num, f"'{sheet_num}'"]:
            if candidate in available_sheets:
                sn = candidate
                break
        if sn is None:
            logger.debug("Sheet %r not found in %s — skipping.", sheet_num, path.name)
            continue

        try:
            df = pd.read_excel(path, sheet_name=sn, header=None, dtype=object)
        except Exception as exc:
            logger.warning("Could not read sheet %r in %s: %s", sn, path.name, exc)
            continue

        if len(df) <= _ROW_TOTAL:
            logger.warning("Sheet %r in %s has only %d rows — skipping.",
                           sn, path.name, len(df))
            continue

        pairs = _extract_period_count_pairs(df, period_from_filename, include_prior_year)
        for pair in pairs:
            records.append({
                "year_month": pair["year_month"],
                "transit_point": transit_point,
                "count": pair["count"],
            })

    df_out = pd.DataFrame(records)
    if df_out.empty:
        logger.warning("No data extracted from %s.", path.name)
        return df_out

    df_out["count"] = df_out["count"].astype("int64")
    df_out = df_out.drop_duplicates(subset=["year_month", "transit_point"])
    logger.info(
        "%s → %d rows (%d transit_points, periods: %s)",
        path.name,
        len(df_out),
        df_out["transit_point"].nunique(),
        sorted(str(p) for p in df_out["year_month"].unique()),
    )
    return df_out


def parse_all_monthly_fast_release(
    directory: Path | str,
    glob_pattern: str = "*.xlsx",
    include_prior_year: bool = True,
) -> pd.DataFrame:
    """Parse all DSEC monthly fast-release Excel files in a directory.

    Scans ``directory`` for files matching ``glob_pattern``, parses each
    with ``parse_monthly_fast_release()``, and concatenates the results.
    Duplicate (year_month, transit_point) rows are dropped — later files
    take precedence (files are processed in alphabetical order so newer
    releases overwrite older provisional figures).

    Args:
        directory: Directory containing the downloaded ``.xlsx`` files.
        glob_pattern: Glob pattern to filter filenames. Default ``"*.xlsx"``.
        include_prior_year: Pass-through to ``parse_monthly_fast_release()``.

    Returns:
        Combined long-format DataFrame sorted by transit_point, year_month.

    Raises:
        FileNotFoundError: If ``directory`` does not exist.
        ValueError: If no Excel files match the pattern.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"DSEC directory not found: {directory}")

    paths = sorted(directory.glob(glob_pattern))
    # Exclude sub-directories and non-DSEC files
    paths = [p for p in paths if p.is_file() and _FILENAME_PATTERN.search(p.stem)]

    if not paths:
        raise ValueError(
            f"No DSEC monthly fast-release files found in {directory} "
            f"matching pattern '{glob_pattern}'.\n"
            "Expected filenames like E_MV_FR_2025_M01.xlsx or CPE_MV_FR_2026_M04.xlsx."
        )

    logger.info("Found %d DSEC monthly files to parse in %s.", len(paths), directory)

    all_frames: list[pd.DataFrame] = []
    for p in paths:
        try:
            frame = parse_monthly_fast_release(p, include_prior_year=include_prior_year)
            if not frame.empty:
                all_frames.append(frame)
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", p.name, exc)

    if not all_frames:
        raise ValueError("All DSEC files failed to parse. Check file format.")

    combined = pd.concat(all_frames, ignore_index=True)
    # Keep last occurrence on duplicates (alphabetical order → newer = later)
    combined = combined.drop_duplicates(subset=["year_month", "transit_point"], keep="last")
    combined = combined.sort_values(["transit_point", "year_month"]).reset_index(drop=True)

    n_months = combined["year_month"].nunique()
    date_range = f"{combined['year_month'].min()} → {combined['year_month'].max()}"
    logger.info(
        "Combined DSEC: %d rows, %d unique months (%s), transit_points=%s",
        len(combined),
        n_months,
        date_range,
        sorted(combined["transit_point"].unique()),
    )
    return combined


# ---------------------------------------------------------------------------
# Legacy wide-format parser (annual main-table Excel)
# ---------------------------------------------------------------------------


def _detect_year_col(df_raw: pd.DataFrame) -> int:
    for row_idx in range(min(20, len(df_raw))):
        cell = str(df_raw.iloc[row_idx, 0])
        if _YEAR_PATTERN.match(cell):
            return row_idx
    raise ValueError(
        "Could not auto-detect DSEC year column in the first 20 rows. "
        "This file may be a monthly fast-release — use parse_monthly_fast_release() instead."
    )


def _slugify_sheet(sheet_name: str) -> str:
    key = sheet_name.lower().strip()
    return _SHEET_TRANSIT_MAP.get(key, key.replace(" ", "_"))


def _parse_wide_sheet(
    df_raw: pd.DataFrame,
    transit_point: str,
    first_data_row: int,
) -> pd.DataFrame:
    records: list[dict] = []
    for row_idx in range(first_data_row, len(df_raw)):
        row = df_raw.iloc[row_idx]
        year_match = _YEAR_PATTERN.match(str(row.iloc[0]))
        if not year_match:
            break
        year = int(year_match.group(1))
        month_vals: list[int | None] = []
        for col_offset in range(1, min(14, len(row))):
            val = _clean_number(row.iloc[col_offset])
            if val is not None:
                month_vals.append(val)
            if len(month_vals) == 12:
                break
        for month_idx, count in enumerate(month_vals):
            if count is None:
                continue
            period = pd.Period(year=year, month=month_idx + 1, freq="M")
            records.append({"year_month": period, "transit_point": transit_point, "count": count})
    if not records:
        return pd.DataFrame(columns=["year_month", "transit_point", "count"])
    df = pd.DataFrame(records)
    df["count"] = df["count"].astype("int64")
    return df


def parse_xlsx(
    path: Path | str,
    sheet_name: str | int | None = None,
    header_row: int | None = None,
) -> pd.DataFrame:
    """Parse a DSEC wide-format annual Excel (one row per year, 12 month columns).

    Note: The DSEC monthly fast-release files (``E_MV_FR_YYYY_MNN.xlsx``) use a
    *different* format — use ``parse_monthly_fast_release()`` for those.

    Args:
        path: Path to the ``.xlsx`` file.
        sheet_name: Sheet to load. ``None`` = load all sheets.
        header_row: First year-data row (0-based). ``None`` = auto-detect.

    Returns:
        Long-format DataFrame: year_month (Period[M]), transit_point (str), count (int64).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DSEC Excel file not found: {path}")

    # Auto-detect: if this looks like a monthly fast-release file, delegate
    if _FILENAME_PATTERN.search(path.stem):
        logger.info(
            "%s looks like a monthly fast-release file. "
            "Delegating to parse_monthly_fast_release().", path.name
        )
        return parse_monthly_fast_release(path)

    xf = pd.ExcelFile(path)
    sheet_names: list[str] = xf.sheet_names
    sheets_to_load = (
        [sheet_name] if sheet_name is not None
        else [s for s in sheet_names if not s.lower().startswith("note")]
    )

    all_frames: list[pd.DataFrame] = []
    for sn in sheets_to_load:
        raw = pd.read_excel(path, sheet_name=sn, header=None, dtype=object)
        if raw.empty:
            continue
        try:
            first_row = header_row if header_row is not None else _detect_year_col(raw)
        except ValueError as exc:
            logger.warning("Sheet %r: %s — skipping.", sn, exc)
            continue
        transit = _slugify_sheet(sn) if len(sheet_names) > 1 else "total"
        frame = _parse_wide_sheet(raw, transit_point=transit, first_data_row=first_row)
        if not frame.empty:
            all_frames.append(frame)

    if not all_frames:
        raise ValueError(f"No visitor arrival data could be parsed from {path}.")

    result = pd.concat(all_frames, ignore_index=True)
    result = result.drop_duplicates(subset=["year_month", "transit_point"])
    result = result.sort_values(["transit_point", "year_month"]).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Quarterly PDF parser (backup)
# ---------------------------------------------------------------------------


def parse_quarterly_pdf(path: Path | str) -> pd.DataFrame:
    """Parse a DSEC quarterly fast-release PDF using ``pdfplumber`` (optional).

    Requires: ``pip install pdfplumber``

    Args:
        path: Path to the downloaded ``.pdf`` file.

    Returns:
        Long-format DataFrame: year_month (Period[M]), transit_point ("total"),
        count (int64). Only the 3 months of the quarter are returned.

    Raises:
        ImportError: If ``pdfplumber`` is not installed.
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If no numeric data table can be found.
    """
    try:
        import pdfplumber  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "pdfplumber is required for PDF parsing. Install with: pip install pdfplumber"
        ) from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DSEC quarterly PDF not found: {path}")

    records: list[dict] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    if not row or row[0] is None:
                        continue
                    cell0 = str(row[0]).strip()
                    month_match = re.match(
                        r"([A-Za-z]{3})\s+(\d{4})|(\d{4})[/-](\d{1,2})", cell0
                    )
                    if not month_match:
                        continue
                    if month_match.group(1):
                        month_num = _MONTH_NAMES.index(month_match.group(1)) + 1
                        year = int(month_match.group(2))
                    else:
                        year = int(month_match.group(3))
                        month_num = int(month_match.group(4))
                    count = _clean_number(row[1] if len(row) > 1 else None)
                    if count is None:
                        continue
                    period = pd.Period(year=year, month=month_num, freq="M")
                    records.append({"year_month": period, "transit_point": "total", "count": count})

    if not records:
        raise ValueError(f"No monthly visitor data found in {path}.")

    df = pd.DataFrame(records)
    df["count"] = df["count"].astype("int64")
    df = df.drop_duplicates(subset=["year_month", "transit_point"])
    df = df.sort_values("year_month").reset_index(drop=True)
    return df
