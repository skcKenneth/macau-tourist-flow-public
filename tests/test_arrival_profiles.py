"""Tests for the within-day arrival profiles (EXP-11 / Goal A).

Covers normalization and qualitative shape of each profile, the registry
dispatcher, and a backward-compatibility regression guaranteeing that the
refactored ``_build_real_arrival_tensor`` (default ``profile="gaussian"``)
reproduces the project's original hard-coded Gaussian shape exactly.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.utils import arrival_profiles as ap


T_STEPS = 168          # 14 h at 5-min steps (the project default)
DT = 14.0 / 168.0


def _mass(w: torch.Tensor) -> float:
    return float((w.sum() * DT).item())


class TestProfileInvariants:
    @pytest.mark.parametrize("name", sorted(ap.PROFILES))
    def test_shape_and_normalization(self, name):
        w = ap.weights(name, T_STEPS, DT)
        assert w.shape == (T_STEPS,)
        assert torch.all(w >= 0), f"{name} has negative weights"
        assert abs(_mass(w) - 1.0) < 1e-5, f"{name} mass={_mass(w):.6f} != 1"

    @pytest.mark.parametrize("name", sorted(ap.PROFILES))
    def test_finite(self, name):
        w = ap.weights(name, T_STEPS, DT)
        assert torch.isfinite(w).all()

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError):
            ap.weights("does_not_exist", T_STEPS, DT)


class TestProfileShapes:
    def test_gaussian_single_peak_at_expected_time(self):
        w = ap.weights("gaussian", T_STEPS, DT, peak_time_hours=3.0, sigma_hours=1.5)
        peak_idx = int(torch.argmax(w).item())
        peak_hour = peak_idx * DT
        assert abs(peak_hour - 3.0) < 0.2

    def test_near_uniform_is_flat(self):
        w = ap.weights("near_uniform", T_STEPS, DT)
        assert float((w.max() - w.min()).item()) < 1e-6

    def test_double_peak_has_two_local_maxima(self):
        w = ap.weights("double_peak", T_STEPS, DT,
                       peak1_hours=2.5, peak2_hours=8.0, sigma_hours=1.0)
        # count interior local maxima
        maxima = [
            i for i in range(1, T_STEPS - 1)
            if w[i] > w[i - 1] and w[i] > w[i + 1]
        ]
        assert len(maxima) == 2, f"expected 2 peaks, found {len(maxima)}"

    def test_plateau_is_broader_than_gaussian(self):
        """The plateau should spread its mass more widely than a sharp Gaussian."""
        g = ap.weights("gaussian", T_STEPS, DT, peak_time_hours=7.0, sigma_hours=1.5)
        p = ap.weights("broad_midday_plateau", T_STEPS, DT)
        # Variance of the implied distribution (w*dt is a probability mass fn).
        t = torch.linspace(0.0, (T_STEPS - 1) * DT, T_STEPS)

        def var(w):
            pmf = w * DT
            mean = float((pmf * t).sum().item())
            return float((pmf * (t - mean) ** 2).sum().item())

        assert var(p) > var(g)


class TestBackwardCompat:
    def test_build_real_arrival_tensor_default_matches_legacy_gaussian(self):
        """Refactored builder (profile='gaussian') == the original hard-coded shape."""
        from src.run_exp05 import _build_real_arrival_tensor

        node_order = ["ruins_st_pauls", "senado_square", "ferry_outer"]
        daily = {"ferry_outer": 1234.0}
        peak, sigma = 2.0, 1.5

        got = _build_real_arrival_tensor(
            node_order=node_order, T_steps=T_STEPS, dt=DT,
            daily_source_counts=daily, peak_time_hours=peak, sigma_hours=sigma,
        )

        # Legacy formula, inline, as it was before the refactor.
        t_vec = torch.linspace(0.0, (T_STEPS - 1) * DT, T_STEPS)
        g_raw = torch.exp(-0.5 * ((t_vec - peak) / sigma) ** 2)
        g_norm = g_raw / (g_raw.sum() * DT + 1e-12)
        expected = torch.zeros(T_STEPS, len(node_order))
        expected[:, node_order.index("ferry_outer")] = 1234.0 * g_norm

        assert torch.allclose(got, expected, atol=1e-6)

    def test_profile_changes_shape_but_conserves_total(self):
        """A non-default profile changes timing but preserves the daily total."""
        from src.run_exp05 import _build_real_arrival_tensor

        node_order = ["ruins_st_pauls", "ferry_outer"]
        daily = {"ferry_outer": 1000.0}
        col = node_order.index("ferry_outer")

        g_gauss = _build_real_arrival_tensor(
            node_order, T_STEPS, DT, daily, 2.0, 1.5, profile="gaussian",
        )
        g_unif = _build_real_arrival_tensor(
            node_order, T_STEPS, DT, daily, 2.0, 1.5, profile="near_uniform",
        )

        # Same conserved daily total (sum * dt), different temporal shape.
        assert abs(float((g_gauss[:, col].sum() * DT).item())
                   - float((g_unif[:, col].sum() * DT).item())) < 1e-3
        assert not torch.allclose(g_gauss[:, col], g_unif[:, col])
