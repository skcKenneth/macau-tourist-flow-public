"""Tests for the competing baseline models (EXP-10 / Goal C)."""

from __future__ import annotations

import torch

from src.models.baselines import (
    GravityModel,
    MultinomialLogitModel,
    evaluate_static_model,
    fit_static_model,
)


N_SOURCES = 3
N_ATT = 5


def _dist():
    torch.manual_seed(0)
    return torch.rand(N_SOURCES, N_ATT) * 1000.0 + 100.0  # 100–1100 m


def _items(model_target):
    """Build train items whose obs come from a fixed target model (recoverable)."""
    items = []
    for _ in range(4):
        w = torch.rand(N_SOURCES)
        with torch.no_grad():
            obs = model_target.predict(w)
        items.append({"source_weights": w, "obs": obs})
    return items


class TestPredictShapes:
    def test_gravity_predict_normalized(self):
        m = GravityModel(N_ATT, _dist())
        p = m.predict(torch.rand(N_SOURCES))
        assert p.shape == (N_ATT,)
        assert torch.all(p >= 0)
        assert abs(float(p.sum().detach()) - 1.0) < 1e-5

    def test_mnl_predict_normalized(self):
        m = MultinomialLogitModel(N_ATT, _dist())
        p = m.predict(torch.rand(N_SOURCES))
        assert p.shape == (N_ATT,)
        assert torch.all(p >= 0)
        assert abs(float(p.sum().detach()) - 1.0) < 1e-5

    def test_as_dict_keys(self):
        assert set(GravityModel(N_ATT, _dist()).as_dict()) == {"mass", "theta"}
        assert set(MultinomialLogitModel(N_ATT, _dist()).as_dict()) == {"alpha", "gamma"}


class TestFitting:
    def test_gravity_fit_reduces_loss(self):
        D = _dist()
        target = GravityModel(N_ATT, D)
        with torch.no_grad():
            target.log_mass.copy_(torch.randn(N_ATT))
            target.log_theta.copy_(torch.tensor(0.3))
        items = _items(target)
        model = GravityModel(N_ATT, D)
        hist = fit_static_model(model, items, n_epochs=200, lr=5e-2)
        assert hist[-1] < hist[0]
        assert evaluate_static_model(model, items) < 0.05

    def test_mnl_fit_reduces_loss(self):
        D = _dist()
        target = MultinomialLogitModel(N_ATT, D)
        with torch.no_grad():
            target.alpha.copy_(torch.randn(N_ATT))
            target.log_gamma.copy_(torch.tensor(0.2))
        items = _items(target)
        model = MultinomialLogitModel(N_ATT, D)
        hist = fit_static_model(model, items, n_epochs=200, lr=5e-2)
        assert hist[-1] < hist[0]

    def test_mnl_distance_penalizes_far_attractions(self):
        """With equal alpha, a single source favours the nearest attraction."""
        D = torch.tensor([[100.0, 500.0, 1000.0, 1500.0, 2000.0]])
        m = MultinomialLogitModel(5, D, gamma_init=2.0)
        with torch.no_grad():
            m.alpha.zero_()
        p = m.predict(torch.tensor([1.0]))
        assert int(torch.argmax(p)) == 0  # nearest gets the most mass
