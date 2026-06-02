"""main.py — run the whole Macau tourist-flow MFG pipeline end to end.

This single entry point runs the experiments in the correct order and **auto-wires**
the EXP-05 calibration output into the downstream interventions/baselines/validity
experiments, so you no longer have to edit each config's ``fitted_params_path`` by
hand.

Examples
--------
    python main.py                     # run the full pipeline in order
    python main.py --list              # show the pipeline steps and exit
    python main.py --only exp08 exp09  # run just these (uses the latest EXP-05 fit)
    python main.py --from exp07        # run from this step to the end
    python main.py --skip exp02 exp03  # run everything except these
    python main.py --keep-going        # don't stop at the first failing step

Data note: EXP-05 and the experiments after it need the processed datasets
(``data/processed/*.parquet``). If they are missing, acquire the source data and run
``python -m src.ingest_data`` first (see ``data/README.md``).
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

REPO = Path(__file__).resolve().parent
CONFIGS = REPO / "configs"
EXPERIMENTS = REPO / "experiments"

# step id -> (human label, module providing run(cfg), config filename | None)
STEPS: list[tuple[str, str, str | None, str | None]] = [
    ("exp01", "Graph sanity",            "src.run_exp01",             "exp01_graph_sanity.yaml"),
    ("exp02", "Random-walk baseline",    "src.run_exp02",             "exp02_baseline.yaml"),
    ("exp03", "Solver validation",       "src.run_exp03",             "exp03_solver_validation.yaml"),
    ("exp04", "Calibration recovery",    "src.run_exp04",             "exp04_calibration_recovery.yaml"),
    ("exp05", "Real-data calibration",   "src.run_exp05",             "exp05_real_calibration.yaml"),
    ("exp06", "Sensitivity analysis",    "src.run_exp06",             "exp06_sensitivity.yaml"),
    ("exp07", "Entrance metering",       "src.run_exp07",             "exp07_entrance_metering.yaml"),
    ("exp08", "Routing recommendations", "src.run_exp08",             "exp08_routing.yaml"),
    ("exp09", "Combined + compliance",   "src.run_exp09",             "exp09_combined.yaml"),
    ("exp10", "Baselines + ablation",    "src.run_exp10",             "exp10_baselines.yaml"),
    ("exp11", "Validity scope",          "src.run_exp11",             "exp11_validity_scope.yaml"),
    ("exp12", "Convergence + gradient",  "src.run_exp12_convergence", "exp12_convergence.yaml"),
    ("figures", "Generate figures + results table", "src.evaluation.generate_figures", None),
]
STEP_IDS = [s[0] for s in STEPS]
STEP_INFO = {s[0]: s for s in STEPS}

# Steps whose config carries an EXP-05 fitted_params path to auto-wire.
NEEDS_FITTED = {"exp07", "exp08", "exp09", "exp10", "exp11"}
# Steps that read the processed real datasets.
DATA_DEPENDENT = {"exp05", "exp07", "exp08", "exp09", "exp10", "exp11"}


def _latest_exp05_fitted() -> Path:
    """Path to the most recent EXP-05 fitted_params.yaml, or a clear error."""
    for d in sorted(EXPERIMENTS.glob("*EXP-05*"), reverse=True):
        fp = d / "fitted_params.yaml"
        if fp.exists():
            return fp
    raise FileNotFoundError(
        "No EXP-05 fitted_params.yaml found under experiments/. "
        "Run the calibration first:  python main.py --only exp05"
    )


def _check_processed_data(selected: list[str]) -> None:
    """Fail early with a helpful message if real datasets are missing."""
    if not any(s in DATA_DEPENDENT for s in selected):
        return
    required = [
        REPO / "data" / "processed" / "arrivals_monthly.parquet",
        REPO / "data" / "processed" / "attractions.parquet",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        names = ", ".join(p.relative_to(REPO).as_posix() for p in missing)
        raise SystemExit(
            f"Missing processed data ({names}).\n"
            "Acquire the source data and build it first:\n"
            "    python -m src.ingest_data --source all\n"
            "See data/README.md for acquisition details."
        )


def _run_step(step_id: str) -> bool:
    _, label, module_name, cfg_name = STEP_INFO[step_id]
    module = importlib.import_module(module_name)

    if step_id == "figures":
        rc = module.main([])  # uses the script's default exp_dir/out_dir
        return rc == 0

    with open(CONFIGS / cfg_name, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if step_id in NEEDS_FITTED:
        fp = _latest_exp05_fitted()
        cfg.setdefault("data", {})["fitted_params_path"] = str(fp)
        logger.info("  auto-wired EXP-05 fit: %s", fp.relative_to(REPO).as_posix())

    return bool(module.run(cfg))


def _select_steps(args: argparse.Namespace) -> list[str]:
    if args.only:
        unknown = [s for s in args.only if s not in STEP_IDS]
        if unknown:
            raise SystemExit(f"Unknown step(s): {unknown}. Valid: {STEP_IDS}")
        return [s for s in STEP_IDS if s in set(args.only)]  # keep canonical order
    steps = STEP_IDS
    if args.from_step:
        if args.from_step not in STEP_IDS:
            raise SystemExit(f"Unknown --from step: {args.from_step}. Valid: {STEP_IDS}")
        steps = STEP_IDS[STEP_IDS.index(args.from_step):]
    if args.skip:
        steps = [s for s in steps if s not in set(args.skip)]
    return steps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", metavar="STEP", help="run only these steps")
    parser.add_argument("--from", dest="from_step", metavar="STEP", help="run from this step to the end")
    parser.add_argument("--skip", nargs="+", metavar="STEP", help="skip these steps")
    parser.add_argument("--keep-going", action="store_true", help="continue after a failing step")
    parser.add_argument("--list", action="store_true", help="list the pipeline steps and exit")
    args = parser.parse_args(argv)

    if args.list:
        print("Pipeline steps (in order):")
        for sid, label, _mod, cfg in STEPS:
            tag = " [needs EXP-05 fit]" if sid in NEEDS_FITTED else ""
            print(f"  {sid:8s} {label}{tag}")
        return 0

    selected = _select_steps(args)
    _check_processed_data(selected)

    logger.info("Running %d step(s): %s", len(selected), ", ".join(selected))
    results: list[tuple[str, str, float]] = []
    t_all = time.perf_counter()

    for sid in selected:
        label = STEP_INFO[sid][1]
        logger.info("=" * 70)
        logger.info(">>> %s — %s", sid, label)
        t0 = time.perf_counter()
        try:
            ok = _run_step(sid)
            status = "PASS" if ok else "FAIL"
        except Exception as exc:  # noqa: BLE001 — report and optionally continue
            logger.exception("step %s raised: %s", sid, exc)
            status = "ERROR"
            ok = False
        dt = time.perf_counter() - t0
        results.append((sid, status, dt))
        logger.info("<<< %s — %s (%.1f s)", sid, status, dt)
        if not ok and not args.keep_going:
            logger.error("Stopping at first failure (use --keep-going to continue).")
            break

    logger.info("=" * 70)
    logger.info("Pipeline summary (%.1f s total):", time.perf_counter() - t_all)
    for sid, status, dt in results:
        logger.info("  %-8s %-6s %6.1f s", sid, status, dt)
    n_ok = sum(1 for _, s, _ in results if s == "PASS")
    logger.info("%d/%d steps passed.", n_ok, len(results))
    return 0 if n_ok == len(results) and len(results) == len(selected) else 1


if __name__ == "__main__":
    sys.exit(main())
