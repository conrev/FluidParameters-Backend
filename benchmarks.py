"""
Standard synthetic test functions for the preferential-BO evaluation (textbook PBO setup).

All are MINIMISATION problems from `botorch.test_functions.synthetic`; the latent utility used by
the oracle is g = -f (higher = more preferred). We keep the 2-D members of the usual PBO/BO suite
(González 2017; Astudillo et al. qEUBO 2023), since the current system optimises the acquisition by
exhaustive enumeration over a discrete grid and therefore stays low-dimensional.
"""

from __future__ import annotations

import torch
from botorch.test_functions import synthetic as S


class Forrester:
    """
    Forrester et al. (2008) 1-D test function: f(x) = (6x-2)^2 sin(12x-4) on [0, 1].
    Global minimum f* ≈ -6.0207 at x ≈ 0.7572. Not in BoTorch, so defined here with the same
    minimal interface (`.bounds`, `.dim`, `.optimal_value`, callable) that make_benchmark needs.
    """

    dim = 1
    _f_star = -6.020740

    def __init__(self, negate: bool = False) -> None:
        self.negate = negate
        self.bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    @property
    def optimal_value(self) -> float:
        return -self._f_star if self.negate else self._f_star

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        x = X[..., 0]
        y = (6.0 * x - 2.0) ** 2 * torch.sin(12.0 * x - 4.0)
        return -y if self.negate else y


# name -> (class, kwargs).  Dimensionality varies (see the `dim` kwarg).
BENCHMARKS: dict[str, tuple] = {
    "Forrester": (Forrester, {}),          # 1-D, single global min
    "Branin": (S.Branin, {}),              # 2-D, 3 tied global minima
    "SixHumpCamel": (S.SixHumpCamel, {}),  # 2-D, 2 tied global minima + local minima
    "Ackley": (S.Ackley, {"dim": 2}),      # 2-D, global min inside a funnel of local minima
    "Ackley4D": (S.Ackley, {"dim": 4}),    # 4-D, the dimensionality-scaling example
    "Levy": (S.Levy, {"dim": 2}),          # 2-D, many local minima
    "Rosenbrock": (S.Rosenbrock, {"dim": 2}),  # 2-D, narrow curved valley
}

# Curated multi-dimensional example: 1-D / 2-D / 4-D.
MULTI_D_SUITE = ["Forrester", "Branin", "Ackley4D"]


def make_benchmark(name: str):
    """Return (function, bounds(2,d), dim, optimal_value) for a named benchmark."""
    cls, kwargs = BENCHMARKS[name]
    f = cls(**kwargs)
    return f, f.bounds.double(), int(f.dim), float(f.optimal_value)
