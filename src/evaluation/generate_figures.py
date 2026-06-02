"""Entry point: regenerate all figures from experiment outputs.

Every figure must be regeneratable from a single script run. This script curates
the canonical figure(s) from the latest run of each experiment (EXP-01..EXP-12)
into ``figures/`` under stable names (both ``.png`` @300 DPI and ``.pdf``, exactly
as each experiment already saved them), and writes a master results table
(``results_table.{md,csv}``).

Copying the already-validated figures is deterministic and faithful; re-running the
script reproduces ``figures/`` from ``experiments/``. (Re-plotting from the saved
CSVs is a possible later enhancement.)

Usage::

    python -m src.evaluation.generate_figures
    python -m src.evaluation.generate_figures --exp_dir experiments/ --out_dir report/figures/

Figure inventory (report order):
    Fig01 — Macau heritage graph (EXP-01)
    Fig02 — Calibration: predicted vs observed attraction shares (EXP-05)
    Fig03 — Monthly calibration MAE, train vs held-out (EXP-05)
    Fig04 — Sensitivity tornado at the bottleneck (EXP-06)
    Fig05 — Entrance-metering Pareto frontier (EXP-07)
    Fig06 — Routing recommendations: top edges (EXP-08)
    Fig07 — Routing edge ablation (EXP-08)
    Fig08 — Combined metering+routing Pareto (EXP-09)
    Fig09 — Routing benefit vs compliance phi (EXP-09)
    Fig10 — Baselines comparison: held-out MAE (EXP-10)
    Fig11 — Validity scope: peak reduction vs assumed profile (EXP-11)
    Fig12 — Validity scope: held-out MAE invariance (EXP-11)
    Fig13 — Validity scope: alpha across profiles (EXP-11)
    Fig14 — Contraction factor, damped lambda=0.5 (EXP-12)
    Fig15 — Contraction factor, undamped lambda=1.0 (EXP-12)
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# (experiment pattern, source figure stem, report figure name)
FIGURE_MAP: list[tuple[str, str, str]] = [
    ("EXP-01", "graph", "Fig01_macau_graph"),
    ("EXP-05", "predicted_vs_observed", "Fig02_calibration_fit"),
    ("EXP-05", "monthly_mae", "Fig03_calibration_monthly_mae"),
    ("EXP-06", "tornado_peak_density", "Fig04_sensitivity_tornado"),
    ("EXP-07", "pareto_202508", "Fig05_metering_pareto"),
    ("EXP-08", "top_edges_202508", "Fig06_routing_top_edges"),
    ("EXP-08", "ablation_202508", "Fig07_routing_ablation"),
    ("EXP-09", "pareto_combined", "Fig08_combined_pareto"),
    ("EXP-09", "compliance_phi", "Fig09_compliance"),
    ("EXP-10", "model_comparison", "Fig10_baselines"),
    ("EXP-11", "peak_reduction_vs_profile", "Fig11_validity_peak_reduction"),
    ("EXP-11", "val_mae_across_profiles", "Fig12_validity_mae"),
    ("EXP-11", "alpha_across_profiles", "Fig13_validity_alpha"),
    ("EXP-12", "contraction_lambda_0p5", "Fig14_contraction_damped"),
    ("EXP-12", "contraction_lambda_1p0", "Fig15_contraction_undamped"),
]

# Headline results for the master table (validated values; see docs/05 + each
# experiment's summary.txt for provenance).
HEADLINE_RESULTS: list[tuple[str, str, str]] = [
    ("EXP-01 Graph", "Reference edges within 10% of Google Maps", "20/20"),
    ("EXP-03 Solver", "Cases matching analytical equilibrium", "5/5"),
    ("EXP-04 Calibration", "Synthetic alpha recovery (MRE)", "<10% (4.1-9.6%)"),
    ("EXP-05 Real calibration", "Held-out spatial MAE", "0.018"),
    ("EXP-06 Sensitivity", "Dominant driver of peak density", "demand volume (S=0.35)"),
    ("EXP-07 Metering", "Peak reduction at Ruins (single terminal)", "5.5-6.6%"),
    ("EXP-08 Routing", "Peak reduction (full compliance, upper bound)", "~71%"),
    ("EXP-09 Combined", "System-peak reduction / visits preserved", "72.5% / 94%"),
    ("EXP-09 Compliance", "Deployable band (phi=0.1 -> 1.0)", "6.7% -> 70.9%"),
    ("EXP-10 Baselines", "Best baseline vs MFG (held-out MAE)", "gravity 0.0003 vs MFG 0.018"),
    ("EXP-11 Validity", "Calibration MAE across 4 profiles", "0.0182-0.0184 (invariant)"),
    ("EXP-11 Validity", "Routing peak reduction across profiles", "70.6-71.0%"),
    ("EXP-12 Convergence", "Contraction factor at lambda=0.5 (fitted beta)", "~0.5 (~20 iters)"),
    ("EXP-12 Gradient", "One-step vs true gradient direction", "cosine 1.000 (magnitude-biased in beta)"),
]


def _latest_exp_dir(exp_root: Path, pattern: str) -> Path | None:
    """Return the most recent ``experiments/*<pattern>*`` directory, or None."""
    dirs = sorted(d for d in exp_root.glob(f"*{pattern}*") if d.is_dir())
    return dirs[-1] if dirs else None


def _build_results_table(exp_dir: Path, out_dir: Path) -> None:
    """Write the Table-3 master results (markdown + CSV) into ``out_dir``."""
    # CSV
    with open(out_dir / "results_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "metric", "value"])
        w.writerows(HEADLINE_RESULTS)

    lines = [
        "# Results summary",
        "",
        "*Auto-generated by `src/evaluation/generate_figures.py`. Provenance: `docs/05_experiment_plan.md` and each experiment's `summary.txt`.*",
        "",
        "| Experiment | Metric | Value |",
        "|---|---|---|",
    ]
    lines += [f"| {e} | {m} | {v} |" for (e, m, v) in HEADLINE_RESULTS]

    # Append the EXP-10 model-comparison sub-table parsed from its CSV (machine-readable).
    d10 = _latest_exp_dir(exp_dir, "EXP-10")
    csv10 = (d10 / "model_comparison.csv") if d10 else None
    if csv10 and csv10.exists():
        lines += ["", "## EXP-10 — model comparison (held-out spatial MAE, lower is better)", "",
                  "| Model | Held-out MAE |", "|---|---|"]
        with open(csv10, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                lines.append(f"| {row['model']} | {float(row['held_out_mae']):.4f} |")

    (out_dir / "results_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote master results table: %s", out_dir / "results_table.md")


def generate_all_figures(exp_dir: Path, out_dir: Path, dpi: int = 300) -> dict[str, list[str]]:
    """Regenerate all report figures from experiment outputs.

    Curates each figure in ``FIGURE_MAP`` from the latest matching experiment run
    into ``out_dir`` (PNG + PDF), and writes the master results table. A missing
    source figure is a warning, not an error, so a partial experiment set still
    produces what it can.

    Args:
        exp_dir: Root experiment directory (``experiments/``).
        out_dir: Output directory for figures (``report/figures/``).
        dpi: Retained for API/CLI compatibility; sources are already 300 DPI.

    Returns:
        Dict with ``copied`` and ``missing`` lists of report figure names.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    missing: list[str] = []

    for pattern, stem, report_name in FIGURE_MAP:
        d = _latest_exp_dir(exp_dir, pattern)
        if d is None:
            missing.append(report_name)
            logger.warning("No experiment dir for %s (figure %s)", pattern, report_name)
            continue
        any_ext = False
        for ext in ("png", "pdf"):
            src = d / f"{stem}.{ext}"
            if src.exists():
                shutil.copyfile(src, out_dir / f"{report_name}.{ext}")
                any_ext = True
        if any_ext:
            copied.append(report_name)
            logger.info("%-32s <- %s/%s", report_name, d.name, stem)
        else:
            missing.append(report_name)
            logger.warning("Source figure %s.{png,pdf} not found in %s", stem, d.name)

    _build_results_table(exp_dir, out_dir)
    logger.info("Figures regenerated: %d copied, %d missing -> %s", len(copied), len(missing), out_dir)
    return {"copied": copied, "missing": missing}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate all figures from experiment outputs.")
    parser.add_argument("--exp_dir", type=Path, default=Path("experiments"))
    parser.add_argument("--out_dir", type=Path, default=Path("figures"))
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    generate_all_figures(args.exp_dir, args.out_dir, args.dpi)
    return 0


if __name__ == "__main__":
    sys.exit(main())
