"""Unit tests for src/utils/io.py."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest


class TestMakeExperimentDir:
    """Tests for make_experiment_dir."""

    def test_creates_directory(self, tmp_path: Path) -> None:
        from src.utils.io import make_experiment_dir

        outdir = make_experiment_dir(tmp_path, "TEST_exp")
        assert outdir.exists()
        assert outdir.is_dir()

    def test_name_contains_date_prefix(self, tmp_path: Path) -> None:
        from datetime import datetime

        from src.utils.io import make_experiment_dir

        outdir = make_experiment_dir(tmp_path, "TEST_exp")
        date_str = datetime.now().strftime("%Y%m%d")
        assert outdir.name.startswith(date_str)

    def test_name_contains_slug(self, tmp_path: Path) -> None:
        from src.utils.io import make_experiment_dir

        outdir = make_experiment_dir(tmp_path, "MY_SLUG")
        assert "MY_SLUG" in outdir.name

    def test_no_collision_on_repeated_call(self, tmp_path: Path) -> None:
        from src.utils.io import make_experiment_dir

        outdir1 = make_experiment_dir(tmp_path, "TEST_exp")
        outdir2 = make_experiment_dir(tmp_path, "TEST_exp")
        assert outdir1 != outdir2
        assert outdir1.exists()
        assert outdir2.exists()


class TestSaveFigure:
    """Tests for save_figure."""

    def test_creates_png_and_pdf(self, tmp_path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend for tests
        import matplotlib.pyplot as plt

        from src.utils.io import save_figure

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])

        png_path, pdf_path = save_figure(fig, tmp_path, "test_figure")
        plt.close(fig)

        assert png_path.exists(), "PNG file was not created"
        assert pdf_path.exists(), "PDF file was not created"
        assert png_path.suffix == ".png"
        assert pdf_path.suffix == ".pdf"

    def test_png_is_nonzero_bytes(self, tmp_path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from src.utils.io import save_figure

        fig, ax = plt.subplots()
        ax.plot([0, 1])
        png_path, _ = save_figure(fig, tmp_path, "nonzero_test")
        plt.close(fig)

        assert png_path.stat().st_size > 0


class TestSetAllSeeds:
    """Tests for set_all_seeds."""

    def test_python_random_is_seeded(self) -> None:
        from src.utils.io import set_all_seeds

        set_all_seeds(123)
        val1 = random.random()
        set_all_seeds(123)
        val2 = random.random()
        assert val1 == val2

    def test_numpy_is_seeded(self) -> None:
        from src.utils.io import set_all_seeds

        set_all_seeds(456)
        arr1 = np.random.rand(5)
        set_all_seeds(456)
        arr2 = np.random.rand(5)
        np.testing.assert_array_equal(arr1, arr2)

    def test_different_seeds_differ(self) -> None:
        from src.utils.io import set_all_seeds

        set_all_seeds(1)
        val1 = random.random()
        set_all_seeds(2)
        val2 = random.random()
        assert val1 != val2


class TestGetGitHash:
    """Tests for get_git_hash."""

    def test_returns_string(self) -> None:
        from src.utils.io import get_git_hash

        result = get_git_hash()
        assert isinstance(result, str)

    def test_returns_nonempty(self) -> None:
        from src.utils.io import get_git_hash

        result = get_git_hash()
        assert len(result) > 0

    def test_short_hash_shorter_than_full(self) -> None:
        from src.utils.io import get_git_hash

        short = get_git_hash(short=True)
        full = get_git_hash(short=False)
        # If both are "unknown" (not in git repo), lengths are equal — that's OK
        if short != "unknown":
            assert len(short) <= len(full)


class TestSaveConfig:
    """Tests for save_config."""

    def test_creates_config_yaml(self, tmp_path: Path) -> None:
        import yaml

        from src.utils.io import save_config

        cfg = {"seed": 42, "model": {"type": "mfg"}, "nested": [1, 2, 3]}
        cfg_path = save_config(cfg, tmp_path)

        assert cfg_path.exists()
        assert cfg_path.name == "config.yaml"

        with open(cfg_path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        assert loaded == cfg

    def test_unicode_preserved(self, tmp_path: Path) -> None:
        import yaml

        from src.utils.io import save_config

        cfg = {"name_zh": "大三巴牌坊", "value": 1}
        cfg_path = save_config(cfg, tmp_path)

        with open(cfg_path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        assert loaded["name_zh"] == "大三巴牌坊"
