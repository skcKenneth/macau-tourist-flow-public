"""Shared I/O helpers for experiment management.

Provides utilities for:
- Creating timestamped experiment output directories
- Saving matplotlib figures as both PNG (300 DPI) and PDF
- Setting all RNG seeds deterministically
- Capturing the current git commit hash
- Snapshotting experiment configs to YAML
"""

from __future__ import annotations

import logging
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def make_experiment_dir(base: Path | str, name: str) -> Path:
    """Create a timestamped experiment output directory.

    Directory name format: ``YYYYMMDD_<name>``. If a directory with that name
    already exists, a numeric suffix is appended (``_1``, ``_2``, …).

    Args:
        base: Parent directory (usually ``experiments/``).
        name: Short slug describing the experiment, e.g.
            ``"EXP-01_graph_sanity"``.

    Returns:
        Path to the newly created directory.

    Example:
        >>> outdir = make_experiment_dir(Path("experiments"), "EXP-01_graph_sanity")
        >>> # Creates experiments/20260526_EXP-01_graph_sanity/
    """
    base = Path(base)
    date_str = datetime.now().strftime("%Y%m%d")
    stem = f"{date_str}_{name}"
    outdir = base / stem

    if outdir.exists():
        suffix = 1
        while (base / f"{stem}_{suffix}").exists():
            suffix += 1
        outdir = base / f"{stem}_{suffix}"
        logger.info("Directory %s already exists; using %s", base / stem, outdir)

    outdir.mkdir(parents=True, exist_ok=False)
    logger.info("Created experiment directory: %s", outdir)
    return outdir


def save_figure(
    fig: Any,
    outdir: Path | str,
    stem: str,
    dpi: int = 300,
) -> tuple[Path, Path]:
    """Save a matplotlib figure as both PNG and PDF.

    Args:
        fig: A ``matplotlib.figure.Figure`` instance.
        outdir: Output directory (must already exist).
        stem: Filename stem without extension (e.g. ``"graph"``).
        dpi: DPI for the raster PNG. Defaults to 300.

    Returns:
        Tuple of (png_path, pdf_path).
    """
    outdir = Path(outdir)
    png_path = outdir / f"{stem}.png"
    pdf_path = outdir / f"{stem}.pdf"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    logger.info("Saved figure: %s (PNG + PDF)", png_path.name)
    return png_path, pdf_path


def set_all_seeds(seed: int) -> None:
    """Set seeds for Python random, NumPy, and PyTorch (CPU + CUDA).

    Call at the very start of every experiment script to ensure
    full reproducibility.

    Args:
        seed: Integer seed value. Logged to INFO for traceability.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Lazy import: torch may not be available in lightweight test environments
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        logger.warning("torch not available; only random and numpy seeds set.")

    logger.info("All RNG seeds set to %d", seed)


def get_git_hash(short: bool = True) -> str:
    """Return the current git HEAD commit hash.

    Args:
        short: If True (default), return the 8-character short hash.

    Returns:
        Git commit hash string, or ``"unknown"`` if not in a git repository
        or git is not installed.
    """
    try:
        cmd = ["git", "rev-parse", "--short" if short else "", "HEAD"]
        cmd = [c for c in cmd if c]  # remove empty string if not short
        result = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        return result.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Could not retrieve git hash (not in a repo or git not found).")
        return "unknown"


def save_config(cfg: dict[str, Any], outdir: Path | str) -> Path:
    """Write an experiment config dict to ``<outdir>/config.yaml``.

    Args:
        cfg: Configuration dictionary (must be YAML-serialisable).
        outdir: Output directory (must already exist).

    Returns:
        Path to the written config file.
    """
    outdir = Path(outdir)
    cfg_path = outdir / "config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    logger.info("Config snapshot saved to %s", cfg_path)
    return cfg_path
