"""Data ingestion script for DSEC and MGTO tourism statistics.

Parses raw downloaded files into cleaned parquet files under ``data/processed/``,
then logs an entry in ``data/raw/MANIFEST.txt``.

Usage::

    # After downloading DSEC Excel to data/raw/dsec/:
    python -m src.ingest_data --source dsec --input data/raw/dsec/arrivals_20260527.xlsx

    # MGTO: use synthetic fallback (no download required):
    python -m src.ingest_data --source mgto --fallback

    # MGTO: use user-filled manual CSV:
    python -m src.ingest_data --source mgto

    # Both sources in one command:
    python -m src.ingest_data --source all --input data/raw/dsec/arrivals_20260527.xlsx --fallback

Outputs written to ``data/processed/``:
    arrivals_monthly.parquet   — DSEC monthly arrivals
    attractions.parquet        — MGTO per-attraction counts (or synthetic proxy)
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import logging
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest_data")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROCESSED_DIR = Path("data/processed")
MANIFEST_PATH = Path("data/raw/MANIFEST.txt")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    """Return first 12 hex chars of SHA-256 hash of a file (or 'directory' for dirs)."""
    if path.is_dir():
        # Hash the sorted list of filenames in the directory as a fingerprint
        h = hashlib.sha256()
        for p in sorted(path.iterdir()):
            h.update(p.name.encode())
        return h.hexdigest()[:12]
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _append_manifest(
    source: str,
    raw_path: Path | None,
    processed_path: Path,
    n_rows: int,
    notes: str = "",
) -> None:
    """Append an ingestion record to MANIFEST.txt."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sha = _sha256(raw_path) if raw_path and raw_path.exists() else "synthetic"
    line = (
        f"{timestamp} | source={source} | raw={raw_path or 'N/A'} "
        f"| sha256_prefix={sha} | processed={processed_path} "
        f"| rows={n_rows} | {notes}\n"
    )
    with open(MANIFEST_PATH, "a", encoding="utf-8") as fh:
        fh.write(line)
    logger.info("MANIFEST updated: %s", MANIFEST_PATH)


def _print_arrivals_summary(df: pd.DataFrame) -> None:
    """Print a human-readable summary of the arrivals DataFrame."""
    print("\n── DSEC Arrivals Summary ───────────────────────────────────────")
    print(f"  Rows          : {len(df):,}")
    print(f"  Transit points: {sorted(df['transit_point'].unique())}")
    if not df.empty:
        periods = sorted(df["year_month"].unique())
        print(f"  Date range    : {periods[0]}  →  {periods[-1]}")

        # Check for missing months
        all_months = pd.period_range(periods[0], periods[-1], freq="M")
        for tp in df["transit_point"].unique():
            sub = df[df["transit_point"] == tp]["year_month"]
            missing = [str(m) for m in all_months if m not in sub.values]
            if missing:
                print(f"  WARN — missing months for {tp}: {missing[:6]}{'…' if len(missing) > 6 else ''}")

    print("────────────────────────────────────────────────────────────────\n")


def _print_attractions_summary(df: pd.DataFrame) -> None:
    """Print a human-readable summary of the attractions DataFrame."""
    print("\n── MGTO Attractions Summary ────────────────────────────────────")
    print(f"  Rows       : {len(df):,}")
    print(f"  Nodes      : {sorted(df['node_id'].unique())}")
    print(f"  Years      : {sorted(df['year'].unique())}")
    print(f"  Confidence : {df['confidence'].value_counts().to_dict()}")
    if "estimate" in df["confidence"].values:
        print("  NOTE: confidence='estimate' → synthetic proxy; cite accordingly.")
    print("────────────────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# Source-specific ingestion functions
# ---------------------------------------------------------------------------


def ingest_dsec(input_path: Path) -> None:
    """Parse DSEC data: single Excel file OR directory of monthly fast-release files."""
    from src.utils.parse_dsec import parse_all_monthly_fast_release, parse_xlsx

    if input_path.is_dir():
        logger.info("Parsing DSEC directory (monthly fast-release): %s", input_path)
        df = parse_all_monthly_fast_release(input_path)
    else:
        logger.info("Parsing DSEC Excel: %s", input_path)
        df = parse_xlsx(input_path)

    out_path = PROCESSED_DIR / "arrivals_monthly.parquet"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    logger.info("Saved: %s (%d rows)", out_path, len(df))
    _print_arrivals_summary(df)
    _append_manifest(
        source="dsec",
        raw_path=input_path,
        processed_path=out_path,
        n_rows=len(df),
        notes=f"transit_points={list(df['transit_point'].unique())}",
    )


def ingest_mgto(fallback: bool = True) -> None:
    """Parse MGTO CSV (or build synthetic fallback), save parquet, update MANIFEST."""
    from src.utils.parse_mgto import load_or_build

    manual_csv = Path("data/raw/mgto/attractions_manual.csv")
    logger.info(
        "Loading MGTO data (manual_csv=%s, fallback=%s).", manual_csv, fallback
    )
    df = load_or_build(csv_path=manual_csv, fallback=fallback)

    out_path = PROCESSED_DIR / "attractions.parquet"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    logger.info("Saved: %s (%d rows)", out_path, len(df))
    _print_attractions_summary(df)

    raw_path = manual_csv if manual_csv.exists() else None
    _append_manifest(
        source="mgto",
        raw_path=raw_path,
        processed_path=out_path,
        n_rows=len(df),
        notes=f"confidence={df['confidence'].unique().tolist()}",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest DSEC/MGTO tourism data into data/processed/ parquet files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        choices=["dsec", "mgto", "all"],
        required=True,
        help=(
            "Which data source to ingest. "
            "'dsec' requires --input. "
            "'mgto' uses data/raw/mgto/attractions_manual.csv (or --fallback). "
            "'all' runs both."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to the raw DSEC Excel file (required when --source dsec or all).",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        default=False,
        help=(
            "For MGTO: use synthetic proxy from attractions.py if manual CSV is "
            "unavailable or empty.  Always safe to pass."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    source: str = args.source
    input_path: Path | None = args.input
    fallback: bool = args.fallback

    success = True

    if source in ("dsec", "all"):
        if input_path is None:
            parser.error("--input is required when --source is 'dsec' or 'all'.")
        if not input_path.exists():
            logger.error("Input file not found: %s", input_path)
            return 1
        try:
            ingest_dsec(input_path)
        except Exception as exc:
            logger.error("DSEC ingestion failed: %s", exc)
            success = False

    if source in ("mgto", "all"):
        try:
            ingest_mgto(fallback=fallback)
        except Exception as exc:
            logger.error("MGTO ingestion failed: %s", exc)
            success = False

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
