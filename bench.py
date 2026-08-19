#!/usr/bin/env python
"""
bench.py — cost/latency harness for the preferential-BO backend.

Times the hot operations of the PRODUCTION backend (optim/PBO.py) directly, so the numbers reflect
what the live server actually pays:

  * UPDATE GP        — fit_preference_model(datapoints, comparisons)        [grows with #comparisons]
  * GET NEW CANDIDATE — select_next_duel(model, all_X, incumbent)  (EUBO)   [grows with grid size N]
  * RECOMMEND        — model.posterior(all_X).mean  (what _make_result does) [SUPER-linear in N]

Parameter-space size is a knob: N = per_dim ** dim candidates.

Usage:
  python bench.py                 # default sweep: grid-size scaling, then #comparisons scaling
  python bench.py prod            # the real PARAM_SPACE (production grid) at a few comparison counts
  python bench.py 2 60 40         # single config: dim=2, per_dim=60, 40 comparisons
  python bench.py 4 11 40         # a 4-D grid (11^4 = 14,641 candidates)
  python bench.py 10x8x5x5x3 40   # UNEQUAL sizes: 5 params, 6,000 candidates, 40 comparisons

Reads only from optim.PBO (production); does not modify anything.
"""

from __future__ import annotations

import sys
import time
import warnings

import torch

warnings.filterwarnings("ignore")

from optim.PBO import (
    build_candidate_tensor,
    fit_preference_model,
    select_next_duel,
    PARAM_SPACE,
)

REPEATS = 3  # repeats per measurement; we report the minimum (least noisy)


def make_param_space(dim: int, per_dim: int) -> dict:
    """A synthetic discrete parameter space with `per_dim` values per dim (N = per_dim ** dim)."""
    vals = [k / (per_dim - 1) for k in range(per_dim)]
    return {f"x{i}": vals for i in range(dim)}


def make_param_space_sizes(sizes: list[int]) -> dict:
    """A discrete parameter space with UNEQUAL per-dim sizes (N = product of sizes).
    e.g. sizes=[10, 8, 5, 5, 3] -> 5 params, 6,000 candidates."""
    return {
        f"x{i}": ([k / (n - 1) for k in range(n)] if n > 1 else [0.0])
        for i, n in enumerate(sizes)
    }


def synth_inputs(all_X: torch.Tensor, n_comp: int, n_seen: int, seed: int = 0):
    """Distinct seen points + consistent random comparisons + an incumbent (global grid index).

    A smooth synthetic latent makes the comparisons non-degenerate so the Laplace fit is realistic.
    n_seen datapoints drives the O(n_seen^3) GP cost; n_comp drives the likelihood terms.
    """
    gen = torch.Generator().manual_seed(seed)
    n_total, dim = all_X.shape
    n_seen = min(n_seen, n_total)
    seen = torch.randperm(n_total, generator=gen)[:n_seen]
    dp = all_X[seen]
    center = torch.rand(dim, generator=gen, dtype=torch.double)
    u = -((dp - center) ** 2).sum(-1)
    rows = []
    for _ in range(n_comp):
        i, j = torch.randperm(n_seen, generator=gen)[:2].tolist()
        rows.append([i, j] if u[i] > u[j] else [j, i])
    incumbent = int(seen[int(u.argmax())])
    return dp, torch.tensor(rows, dtype=torch.long), incumbent


def _time(fn, repeats: int = REPEATS) -> float:
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def bench(param_space: dict, n_comp: int, n_seen: int | None = None, repeats: int = REPEATS) -> dict:
    """Measure UPDATE-GP, GET-CANDIDATE and RECOMMEND times for one (grid, #comparisons) config."""
    all_X, _, _, _ = build_candidate_tensor(param_space)
    n_seen = n_seen or (n_comp + 1)  # EUBO/king-of-the-hill reuses the incumbent -> ~M+1 distinct
    dp, comps, incumbent = synth_inputs(all_X, n_comp, n_seen)

    model = fit_preference_model(dp, comps)  # warm one fit for the select/recommend timings
    fit_s = _time(lambda: fit_preference_model(dp, comps), repeats)
    sel_s = _time(lambda: select_next_duel(model, all_X, incumbent), repeats)
    with torch.no_grad():  # recommend is expensive at large N -> time it once
        rec_s = _time(lambda: model.posterior(all_X).mean, 1)
    return {
        "dim": all_X.shape[1],
        "N": len(all_X),
        "M": n_comp,
        "fit": fit_s,
        "select": sel_s,
        "recommend": rec_s,
    }


def _fmt(s: float) -> str:
    return f"{s * 1e3:8.1f} ms" if s < 1 else f"{s:8.2f} s "


def print_table(rows: list[dict]) -> None:
    print(
        f"\n{'dim':>3} {'sizes':>12} {'N cand':>10} {'#comp':>6} "
        f"{'update GP':>12} {'get cand':>12} {'recommend':>12}"
    )
    print("-" * 77)
    for r in rows:
        print(
            f"{r['dim']:>3} {str(r.get('per_dim', '-')):>12} {r['N']:>10,} {r['M']:>6} "
            f"{_fmt(r['fit']):>12} {_fmt(r['select']):>12} {_fmt(r['recommend']):>12}"
        )
    print(
        "\nupdate GP  grows with #comparisons (Laplace ~ O(datapoints^3));\n"
        "get cand   grows with N (exhaustive EUBO over the grid);\n"
        "recommend  is super-linear in N (posterior mean over the full grid, as in _make_result)."
    )


def main() -> None:
    args = sys.argv[1:]
    rows: list[dict] = []

    bench(make_param_space(2, 8), 4, repeats=1)  # warm-up: absorb first-call JIT/import overhead

    if args and ("x" in args[0] or "," in args[0]):  # unequal per-dim sizes, e.g. "10x8x5x5x3"
        sep = "x" if "x" in args[0] else ","
        sizes = [int(s) for s in args[0].split(sep)]
        m = int(args[1]) if len(args) > 1 else 40
        r = bench(make_param_space_sizes(sizes), m)
        r["per_dim"] = args[0]
        rows.append(r)
    elif args and args[0] == "prod":
        for m in (10, 25, 50):
            r = bench(PARAM_SPACE, m)
            r["per_dim"] = "prod"
            print(f"  measured prod grid, M={m} ...", flush=True)
            rows.append(r)
    elif len(args) == 3:
        dim, per_dim, m = int(args[0]), int(args[1]), int(args[2])
        r = bench(make_param_space(dim, per_dim), m)
        r["per_dim"] = per_dim
        rows.append(r)
    else:
        print("default sweep (this can take a minute at the larger grids)...", flush=True)
        for per_dim in (20, 40, 60, 80):  # grid-size scaling (2-D), fixed #comparisons
            r = bench(make_param_space(2, per_dim), 40)
            r["per_dim"] = per_dim
            print(f"  2-D per_dim={per_dim} (N={r['N']:,}) done", flush=True)
            rows.append(r)
        for m in (10, 40, 80):  # #comparisons scaling, fixed grid
            r = bench(make_param_space(2, 60), m)
            r["per_dim"] = 60
            print(f"  2-D per_dim=60, M={m} done", flush=True)
            rows.append(r)

    print_table(rows)


if __name__ == "__main__":
    main()
