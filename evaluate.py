"""
evaluate.py — external evaluation harness for the preferential-BO system.

Standalone: it imports the existing `PreferentialBOSession` and reads its state; it does NOT modify
any core component. Run from the repo root:

    python evaluate.py suite                       # textbook PBO benchmark (EUBO vs Random, log-regret)
    python evaluate.py suite 20 40                 # suite: 20 seeds, 40 comparisons (publication-ish)

    python evaluate.py                             # single-oracle mode (defaults)
    python evaluate.py 6 14 bimodal best_live 0.35 # oracle, metric, noise-fraction overrides
    #                  seeds iters oracle  metric   noise

`suite` mode is the standard preferential-BO benchmark: EUBO vs. Random acquisition on the common
synthetic test functions (benchmarks.py), reporting inference regret of the posterior-mean
recommendation vs. number of comparisons on a LOG axis (González 2017; qEUBO 2023). Both share a
random initialisation, so the comparison isolates the acquisition function.

Outputs `eval/convergence.png` (mean regret ± SEM vs #comparisons) and `eval/results.csv`.

------------------------------------------------------------------------------------------------
Methodology (for the paper)
------------------------------------------------------------------------------------------------
* Synthetic oracle. We fix a ground-truth latent utility g over the normalised parameter cube,
  with a known maximiser x* = argmax_x g(x). Default `unimodal`: an ideal-point utility
  g(x) = -||x - x*||^2 (single peak). Optional `bimodal`: log-sum-exp of two bumps (two competing
  optima — exercises the "similarly good basins" case; distance-to-x* is then ill-defined, so we
  report utility regret only).

* Noisy preferences (probit / Thurstone). When the session shows options A and B, the oracle picks
  the one with the higher *noisy* utility: choose A iff g(A)+eps_A >= g(B)+eps_B, with
  eps ~ N(0, sigma^2). This is exactly the choice model the PairwiseGP assumes. sigma is set as a
  fraction (`NOISE_FRAC`) of the utility's spread so it is scale-free across oracle instances.

* Metric (Y axis) — normalised regret in [0, 1], 0 = optimal:
      r_t = ( g(x*) - g(x̂_t) ) / ( g(x*) - min_x g(x) ).
  Three choices via METRIC (or argv[4]):
    - "best_observed" (default): x̂_t = the best (by true g) of ALL points shown in duels up to t.
      Monotone; the classic BO "best-so-far" curve. Measures how quickly good reconstructions are
      surfaced — FORGIVING in low dimension, where random coverage stumbles on good points too.
    - "best_live": x̂_t = the best RECOMMENDATION (posterior-mean argmax) seen up to t. Monotone,
      but measures model *identification* — active EUBO's advantage shows here (and grows with
      noise), because it must actually locate the optimum, not merely have shown it once.
    - "live": the current recommendation only. Same as best_live but NOT monotone (the recommended
      point moves as the noisily-fit posterior changes) — expected for preferential BO.
  Normalisation makes curves comparable across oracle instances (utility is identified only up to an
  affine transform, so raw units are not meaningful — see the guide §3.5).

* X axis — number of comparisons (duels) elicited, 1 .. n_init + n_iterations.

* Baselines / conditions compared:
    - EUBO (active duel selection) vs. random duels  — the core "does the acquisition help?" test.
    - Sobol vs. random warm-up                        — does the space-filling init help early on?
  Each condition is replicated over N_SEEDS independent oracle instances (different x*) and the
  mean ± standard error is plotted.

* Performance note. `PairwiseGP.posterior(X).mean` is super-linear in |X| (~13 s over a 14k-point
  grid) because it builds the joint test covariance. The posterior mean is a per-point marginal, so
  we compute it in CHUNKS (`posterior_mean_chunked`): identical result, but each chunk is small and
  fast, letting the per-step recommendation read-out cover the FULL grid even in higher dimensions —
  so it never skips the optimum cell. The oracle utility is closed-form, so regret is exact anywhere.

Extensions: distance-to-x*; a "basin hit-rate" (does the top reported scenario contain x*?); sweep
NOISE_FRAC for robustness to human inconsistency; sweep n_init.
"""

from __future__ import annotations

import sys
import csv
from pathlib import Path

import torch
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from optim.PBO import PreferentialBOSession, build_candidate_tensor, PARAM_SPACE
from benchmarks import BENCHMARKS, MULTI_D_SUITE, make_benchmark

# BoTorch primitives for the CONTINUOUS eval engine (used directly, NOT via PreferentialBOSession —
# so this path shares only the model class with production and touches no production state).
from botorch.acquisition import PosteriorMean
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.fit import fit_gpytorch_mll
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.optim import optimize_acqf

# ─────────────────────────────── configuration ───────────────────────────────
N_SEEDS = 8  # replications per condition (independent oracle instances)
N_INIT = 6  # warm-up comparisons
N_ITER = 16  # post-warm-up comparisons
NOISE_FRAC = (
    0.05  # preference noise sigma as a fraction of the utility std (0 = noiseless)
)
ORACLE = "unimodal"  # "unimodal" | "bimodal"
GRID_PER_DIM = 15  # fixed grid resolution per dim for ALL functions (n^dim candidates)
SUITE_FLOOR = (
    1e-3  # log-axis floor (clamp + y-limit bottom) shared across all suite panels
)
SUITE_YLIM = (8e-4, 1)  # common y-range so every panel is directly comparable
# "best_observed": best point ever SHOWN (monotone; forgiving — measures coverage).
# "best_live":     best RECOMMENDATION so far (monotone; measures model identification).
# "live":          current recommendation (non-monotone; strict model identification).
METRIC = "best_observed"

CONDITIONS = [
    {"label": "EUBO + Sobol warm-up", "method": "eubo", "warmup": "sobol"},
    {"label": "Random search", "method": "random", "warmup": "random"},
    {"label": "EUBO + random warm-up", "method": "eubo", "warmup": "random"},
]

OUT_DIR = Path("eval")
OUT_PNG = OUT_DIR / "convergence.png"
OUT_CSV = OUT_DIR / "results.csv"
SUITE_PNG = OUT_DIR / "pbo_benchmarks.png"
SUITE_CSV = OUT_DIR / "pbo_benchmarks.csv"


# ─────────────────────────────── synthetic oracle ────────────────────────────
def build_oracle(param_space: dict, oracle_seed: int, kind: str) -> dict:
    """Fix a ground-truth utility g (a closed-form function of NORMALISED coords) and its optimum."""
    all_X, _, x_min, x_range = build_candidate_tensor(param_space)
    gen = torch.Generator().manual_seed(oracle_seed)
    n = len(all_X)

    if kind == "unimodal":
        xstar = all_X[torch.randint(0, n, (1,), generator=gen).item()]

        def g(xn: torch.Tensor) -> torch.Tensor:  # xn: (..., D) normalised
            return -((xn - xstar) ** 2).sum(-1)
    elif kind == "bimodal":
        a = all_X[torch.randint(0, n, (1,), generator=gen).item()]
        b = all_X[torch.randint(0, n, (1,), generator=gen).item()]

        def g(xn: torch.Tensor) -> torch.Tensor:
            return torch.logaddexp(
                -8 * ((xn - a) ** 2).sum(-1), -8 * ((xn - b) ** 2).sum(-1)
            )
    else:
        raise ValueError(f"unknown oracle kind: {kind!r}")

    g_full = g(all_X)
    return {
        "param_space": param_space,
        "g": g,
        "x_min": x_min,
        "x_range": x_range,
        "g_ref": float(
            g_full.max()
        ),  # x* is a grid point here, so best-grid == true optimum
        "g_min": float(g_full.min()),
        "g_std": float(g_full.std()),
        "dim": all_X.shape[1],
    }


def grid_res_for_dim(dim: int) -> int:
    """
    Fixed grid resolution per dimension (GRID_PER_DIM) for ALL functions, so grid DENSITY is held
    constant across the suite — this isolates the effect of dimension rather than confounding it
    with resolution. Total points = GRID_PER_DIM ** dim (e.g. 4-D = 30^4 = 810,000).
    """
    return GRID_PER_DIM


def build_benchmark_oracle(name: str, per_dim: int | None = None) -> dict:
    """
    Wrap a BoTorch synthetic test function `f` as a preference oracle over the CONTINUOUS domain:
    utility g = -f, with normalised [0,1]^d coords mapped onto the function's native domain. Regret
    is measured against the TRUE optimum (`f.optimal_value`).

    Grid-free: the utility's range/spread (for regret normalisation and noise scaling) is estimated
    by **Sobol sampling the continuous unit cube**, NOT by enumerating a grid — so the oracle is
    cheap in any dimension and the continuous engine touches no grid. `param_space` is still emitted
    as a cheap grid *definition* used ONLY by the discrete engine's `PreferentialBOSession`; the
    continuous engine ignores it.
    """
    f, bounds, dim, opt_val = make_benchmark(name)
    lb, ub = bounds[0], bounds[1]
    if per_dim is None:
        per_dim = grid_res_for_dim(dim)
    param_space = {  # discrete grid DEFINITION (cheap); enumerated only inside the discrete session
        f"x{i}": torch.linspace(0.0, 1.0, per_dim).tolist() for i in range(dim)
    }

    def g(xn: torch.Tensor) -> torch.Tensor:
        return -f(lb + xn * (ub - lb))  # normalised coords -> domain -> utility

    # Estimate g_min / g_std over the CONTINUOUS domain via Sobol (no grid enumeration).
    sob = (
        torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=0)
        .draw(8192)
        .double()
    )
    g_sample = g(sob)
    return {
        "param_space": param_space,
        "g": g,
        "x_min": torch.zeros(dim, dtype=torch.double),
        "x_range": torch.ones(dim, dtype=torch.double),
        "g_ref": -opt_val,  # true maximum utility (exact)
        "g_min": float(g_sample.min()),
        "g_std": float(g_sample.std()),
        "dim": dim,
    }


def _normalise_cfg(oracle: dict, cfg: dict) -> torch.Tensor:
    raw = torch.tensor(
        list(cfg.values()), dtype=torch.double
    )  # values are in key order
    return (raw - oracle["x_min"]) / oracle["x_range"]


def oracle_choice(oracle: dict, cfg_a: dict, cfg_b: dict, sigma: float, gen) -> str:
    """Probit/Thurstone noisy preference: prefer the higher noisy utility."""
    u_a = (
        float(oracle["g"](_normalise_cfg(oracle, cfg_a)))
        + float(torch.randn((), generator=gen)) * sigma
    )
    u_b = (
        float(oracle["g"](_normalise_cfg(oracle, cfg_b)))
        + float(torch.randn((), generator=gen)) * sigma
    )
    return "A" if u_a >= u_b else "B"


def normalised_regret(oracle: dict, xn: torch.Tensor) -> float:
    span = oracle["g_ref"] - oracle["g_min"] + 1e-12
    return max(0.0, (oracle["g_ref"] - float(oracle["g"](xn))) / span)


# ─────────────────────── read the system's recommendation ────────────────────
def posterior_mean_chunked(
    model, X: torch.Tensor, chunk_size: int = 2048
) -> torch.Tensor:
    """
    Posterior mean over X, computed in chunks. The mean is a per-point MARGINAL, so chunking is
    exact (identical result, only faster) — it avoids `PairwiseGP.posterior(X).mean` building the
    joint test covariance over all of X at once, which is super-linear in |X| (~13 s at 14k points).
    This lets the read-out use the FULL grid even in higher dimensions, so it never skips the
    optimum cell (the coarse-subsample artifact that pinned Ackley-4D `live` regret at its floor).
    """
    means = []
    with torch.no_grad():
        for start in range(0, len(X), chunk_size):
            means.append(
                model.posterior(X[start : start + chunk_size]).mean.squeeze(-1)
            )
    return torch.cat(means)


def recommendation_point(
    session: PreferentialBOSession, grid: torch.Tensor
) -> torch.Tensor:
    """
    Normalised coords of the argmax of the posterior mean over the FULL grid. This is EXACTLY the
    system's `optimalParameter` (what `_make_result` reports) — no approximation: the chunked mean
    equals `model.posterior(grid).mean` to machine precision, just computed without its ~13 s cost.
    """
    if session.model is not None:
        mean = posterior_mean_chunked(session.model, grid)
        return grid[int(mean.argmax())]
    if session.prev_winner is not None:
        return session.all_X[session.prev_winner]
    return session.all_X[0]


# ─────────────────────────────── one replication ─────────────────────────────
def run_trace(
    oracle: dict, method: str, warmup: str, rep: int, n_iter: int, metric: str
) -> list[float]:
    """Drive one full session against `oracle`; return the per-comparison regret trace."""
    sigma = NOISE_FRAC * oracle["g_std"]
    noise_gen = torch.Generator().manual_seed(2000 + rep)
    session = PreferentialBOSession(
        oracle["param_space"],
        n_init=N_INIT,
        n_iterations=n_iter,
        method=method,
        warmup=warmup,
        seed=rep,
    )
    uses_live = metric in ("live", "best_live")
    # Live metrics read the recommendation over the FULL grid (chunked mean -> cheap & faithful, so
    # it never misses the optimum cell). best_observed reads no posterior at all.
    readout = session.all_X if uses_live else None

    msg = session.start()
    regrets: list[float] = []
    best_seen = float("inf")  # best-observed (monotone)
    best_live = float("inf")  # best live recommendation so far (monotone)
    while True:
        r_a = normalised_regret(oracle, _normalise_cfg(oracle, msg["optionA"]))
        r_b = normalised_regret(oracle, _normalise_cfg(oracle, msg["optionB"]))
        best_seen = min(best_seen, r_a, r_b)

        choice = oracle_choice(oracle, msg["optionA"], msg["optionB"], sigma, noise_gen)
        msg = session.submit_preference(msg["duelId"], choice)

        live = (
            normalised_regret(oracle, recommendation_point(session, readout))
            if uses_live
            else None
        )
        if metric == "live":
            regrets.append(live)
        elif metric == "best_live":
            best_live = min(best_live, live)
            regrets.append(best_live)
        else:  # "best_observed"
            regrets.append(best_seen)
        if msg["type"] == "result":
            return regrets


def run_one(
    param_space: dict, condition: dict, rep: int, n_iter: int, kind: str
) -> list[float]:
    """Legacy single-oracle path (unimodal/bimodal), using the global METRIC."""
    oracle = build_oracle(param_space, oracle_seed=1000 + rep, kind=kind)
    return run_trace(
        oracle, condition["method"], condition["warmup"], rep, n_iter, METRIC
    )


# ───────────────────── CONTINUOUS engine (eval-only, no grid) ─────────────────
# A self-contained BoTorch preferential-BO loop that mirrors the discrete `run_trace` but replaces
# the grid with continuous `optimize_acqf`. It shares the same MODEL (PairwiseGP) and the same
# oracle/metric as the discrete path, so PBO-vs-Random stays a fair comparison — the only variable
# is discrete-enumeration vs continuous-optimisation. It never imports or mutates production state.
def _prefers(
    oracle: dict, xa: torch.Tensor, xb: torch.Tensor, sigma: float, gen
) -> bool:
    """Probit/Thurstone noisy preference between two NORMALISED points; True if A is preferred."""
    ua = float(oracle["g"](xa)) + float(torch.randn((), generator=gen)) * sigma
    ub = float(oracle["g"](xb)) + float(torch.randn((), generator=gen)) * sigma
    return ua >= ub


def _fit_pairwise(X: torch.Tensor, comps: list) -> PairwiseGP:
    """Fit a PairwiseGP on continuous points (same model production uses, assembled directly)."""
    model = PairwiseGP(X, torch.tensor(comps, dtype=torch.long), jitter=1e-4)
    try:
        fit_gpytorch_mll(PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model))
    except Exception as exc:  # noqa: BLE001 — degenerate early data; keep the near-prior model
        print(f"  (continuous fit fallback: {exc})", flush=True)
    model.eval()
    return model


def _argmax_acqf(acqf, bounds: torch.Tensor, restarts: int, raw: int) -> torch.Tensor:
    """Continuous maximiser of an acquisition over `bounds`; returns a (d,) point."""
    cand, _ = optimize_acqf(
        acqf, bounds=bounds, q=1, num_restarts=restarts, raw_samples=raw
    )
    return cand.squeeze(0).detach()


def run_trace_continuous(
    oracle: dict,
    method: str,
    rep: int,
    n_iter: int,
    metric: str,
    restarts: int = 6,
    raw: int = 128,
) -> list[float]:
    """
    Continuous preferential-BO trace: random warm-up, then EUBO (via `optimize_acqf`) vs. Random,
    with the recommendation = continuous argmax of the posterior mean (`optimize_acqf(PosteriorMean)`)
    — the off-grid analogue of `optimalParameter`, so there is NO grid-resolution floor. Structure
    mirrors `run_trace` (same N_INIT / incumbent-vs-challenger / noise / metric semantics).
    """
    d = oracle["dim"]
    bounds = torch.stack(
        [torch.zeros(d), torch.ones(d)]
    ).double()  # normalised unit cube
    sigma = NOISE_FRAC * oracle["g_std"]
    g_noise = torch.Generator().manual_seed(2000 + rep)
    g_warm = torch.Generator().manual_seed(1000 + rep)
    g_duel = torch.Generator().manual_seed(3000 + rep)
    uses_live = metric in ("live", "best_live")

    warm = torch.rand(
        (2 * N_INIT, d), generator=g_warm, dtype=torch.double
    )  # random init pairs
    X = torch.empty((0, d), dtype=torch.double)
    comps: list = []
    incumbent = None
    model = None
    regrets: list[float] = []
    best_seen = float("inf")
    best_live = float("inf")

    for t in range(N_INIT + n_iter):
        if t < N_INIT:  # warm-up: a fresh random pair
            a, b = warm[2 * t], warm[2 * t + 1]
        elif method == "eubo":  # challenger (EUBO) vs. current incumbent
            acqf = AnalyticExpectedUtilityOfBestOption(
                pref_model=model, previous_winner=incumbent.unsqueeze(0)
            )
            a, b = _argmax_acqf(acqf, bounds, restarts, raw), incumbent
        else:  # random: two fresh random points (mirrors select_next_duel_random)
            pts = torch.rand((2, d), generator=g_duel, dtype=torch.double)
            a, b = pts[0], pts[1]

        a_wins = _prefers(oracle, a, b, sigma, g_noise)
        incumbent = a if a_wins else b
        i, j = len(X), len(X) + 1
        X = torch.cat([X, a.unsqueeze(0), b.unsqueeze(0)], dim=0)
        comps.append([i, j] if a_wins else [j, i])
        best_seen = min(
            best_seen, normalised_regret(oracle, a), normalised_regret(oracle, b)
        )

        model = _fit_pairwise(X, comps)
        if uses_live:
            rec = _argmax_acqf(PosteriorMean(model), bounds, restarts, raw)
            live = normalised_regret(oracle, rec)
            best_live = min(best_live, live)
            regrets.append(live if metric == "live" else best_live)
        else:  # best_observed — no posterior read-out needed
            regrets.append(best_seen)
    return regrets


# ──────────────────────────── aggregate + plot ───────────────────────────────
def mean_sem(rows: list[list[float]]) -> tuple[list[float], list[float]]:
    t = torch.tensor(rows)  # (n_seeds, n_steps)
    return t.mean(0).tolist(), (t.std(0) / (t.shape[0] ** 0.5)).tolist()


METRIC_YLABEL = {
    "live": "inference regret (norm., log)",
    "best_live": "best-recommendation regret (norm., log)",
    "best_observed": "best-observed regret (norm., log)",
}


def run_suite(
    n_seeds: int,
    n_iter: int,
    metric: str = "best_observed",
    functions: list[str] | None = None,
    continuous: bool = False,
) -> None:
    """
    PBO benchmark: PBO (EUBO) vs. Random on standard synthetic test functions, reporting regret vs.
    number of comparisons on a LOG axis. `metric` selects "live" (posterior-mean recommendation —
    González 2017 / qEUBO 2023), "best_live" (best recommendation so far), or "best_observed" (best
    point shown so far — needs no posterior read-out, so it scales to higher dimension cheaply).
    Both conditions share a random initialisation, so only the acquisition differs.

    `continuous=True` swaps the DISCRETE grid engine (`run_trace`) for the CONTINUOUS
    `optimize_acqf` engine (`run_trace_continuous`) — no grid floor, so it scales to higher
    dimension. Writes to separate output files so the two engines don't overwrite each other.
    """
    names = functions or MULTI_D_SUITE
    conds = [("Our Method", "eubo"), ("Random", "random")]
    lab0, lab1 = conds[0][0], conds[1][0]
    engine = "continuous" if continuous else "discrete"
    out_pdf = OUT_DIR / f"pbo_benchmarks_{engine}.pdf"  # vector output for the paper
    out_csv = OUT_DIR / f"pbo_benchmarks_{engine}.csv"
    x = list(range(1, N_INIT + n_iter + 1))
    OUT_DIR.mkdir(exist_ok=True)

    ncols = min(3, len(names))
    nrows = (len(names) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.6 * ncols, 3.4 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    summary: dict[tuple[str, str], tuple[float, float]] = {}

    for k, name in enumerate(names):
        ax = axes[k // ncols][k % ncols]
        oracle = build_benchmark_oracle(name)
        print(
            f"\n== {name} ({oracle['dim']}D, grid {len(oracle['param_space']['x0'])}/dim) ==",
            flush=True,
        )
        for label, method in conds:
            if continuous:
                traces = [
                    run_trace_continuous(oracle, method, rep, n_iter, metric)
                    for rep in range(n_seeds)
                ]
            else:
                traces = [
                    run_trace(oracle, method, "random", rep, n_iter, metric)
                    for rep in range(n_seeds)
                ]
            mean, sem = mean_sem(traces)
            summary[(name, label)] = (mean[-1], sem[-1])
            m = torch.tensor(mean).clamp_min(SUITE_FLOOR)
            s = torch.tensor(sem)
            ax.plot(x, m.tolist(), marker="o", ms=2.5, label=label)
            ax.fill_between(
                x, (m - s).clamp_min(SUITE_FLOOR).tolist(), (m + s).tolist(), alpha=0.15
            )
            print(
                f"  {label:10s} final regret = {mean[-1]:.4f} ± {sem[-1]:.4f}",
                flush=True,
            )
        ax.set_yscale("log")
        ax.set_ylim(*SUITE_YLIM)
        ax.axvline(N_INIT + 0.5, ls="--", c="grey", lw=1)
        ax.set_title(f"{name} ({oracle['dim']}D)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="upper right")  # legend on EVERY panel
        if k // ncols == nrows - 1:
            ax.set_xlabel("comparisons")
        if k % ncols == 0:
            ax.set_ylabel(METRIC_YLABEL.get(metric, "regret (norm., log)"))
    for k in range(len(names), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle(
        f" Mean regret for various benchmark functions — {engine} "
        f"({n_seeds} runs per benchmark, noise={NOISE_FRAC})"
    )
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")  # vector PDF (fonts embedded, editable in a paper)
    print(f"\nsaved plot -> {out_pdf}", flush=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(
            ["function", f"{lab0} mean", f"{lab0} sem", f"{lab1} mean", f"{lab1} sem"]
        )
        for name in names:
            a = summary[(name, lab0)]
            b = summary[(name, lab1)]
            w.writerow(
                [name, f"{a[0]:.5f}", f"{a[1]:.5f}", f"{b[0]:.5f}", f"{b[1]:.5f}"]
            )
    print(f"saved table -> {out_csv}", flush=True)

    print("\nfinal inference regret (lower is better):", flush=True)
    for name in names:
        a = summary[(name, lab0)]
        b = summary[(name, lab1)]
        print(
            f"  {name:14s} {lab0} {a[0]:.4f} ± {a[1]:.4f}   {lab1} {b[0]:.4f} ± {b[1]:.4f}",
            flush=True,
        )


def main() -> None:
    global METRIC, NOISE_FRAC
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else N_SEEDS
    n_iter = int(sys.argv[2]) if len(sys.argv) > 2 else N_ITER
    kind = sys.argv[3] if len(sys.argv) > 3 else ORACLE
    if len(sys.argv) > 4:
        METRIC = sys.argv[4]
    if len(sys.argv) > 5:
        NOISE_FRAC = float(sys.argv[5])
    total = N_INIT + n_iter
    x = list(range(1, total + 1))

    OUT_DIR.mkdir(exist_ok=True)
    results: dict[str, tuple[list[float], list[float]]] = {}

    for cond in CONDITIONS:
        print(f"\n== {cond['label']} ==", flush=True)
        traces = []
        for rep in range(n_seeds):
            traces.append(run_one(PARAM_SPACE, cond, rep, n_iter, kind))
            print(f"  seed {rep}: final regret = {traces[-1][-1]:.4f}", flush=True)
        results[cond["label"]] = mean_sem(traces)

    # ---- plot ----
    plt.figure(figsize=(8, 5))
    for cond in CONDITIONS:
        mean, sem = results[cond["label"]]
        mean_t, sem_t = torch.tensor(mean), torch.tensor(sem)
        plt.plot(x, mean, marker="o", ms=3, label=cond["label"])
        plt.fill_between(
            x, (mean_t - sem_t).tolist(), (mean_t + sem_t).tolist(), alpha=0.15
        )
    plt.axvline(
        N_INIT + 0.5, ls="--", c="grey", lw=1, label=f"warm-up ends (n_init={N_INIT})"
    )
    ylabel = {
        "best_observed": "normalised regret of best point shown so far",
        "best_live": "normalised regret of best recommendation so far",
        "live": "normalised simple regret (live recommendation)",
    }.get(METRIC, "normalised regret")
    plt.xlabel("number of comparisons")
    plt.ylabel(ylabel)
    plt.title(
        f"Preferential-BO convergence — {METRIC} ({kind}, noise={NOISE_FRAC}, {n_seeds} seeds)"
    )
    plt.ylim(bottom=0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=130)
    print(f"\nsaved plot -> {OUT_PNG}", flush=True)

    # ---- csv ----
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        header = ["comparison"]
        for cond in CONDITIONS:
            header += [f"{cond['label']} mean", f"{cond['label']} sem"]
        w.writerow(header)
        for i, xi in enumerate(x):
            row = [xi]
            for cond in CONDITIONS:
                mean, sem = results[cond["label"]]
                row += [f"{mean[i]:.5f}", f"{sem[i]:.5f}"]
            w.writerow(row)
    print(f"saved table -> {OUT_CSV}", flush=True)

    # ---- summary ----
    print("\nfinal normalised regret (lower is better):", flush=True)
    for cond in CONDITIONS:
        mean, sem = results[cond["label"]]
        print(f"  {cond['label']:24s}  {mean[-1]:.4f} ± {sem[-1]:.4f}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "suite":
        seeds = int(sys.argv[2]) if len(sys.argv) > 2 else N_SEEDS
        iters = int(sys.argv[3]) if len(sys.argv) > 3 else N_ITER
        suite_metric = sys.argv[4] if len(sys.argv) > 4 else "best_observed"
        is_continuous = "continuous" in sys.argv[5:]
        run_suite(seeds, iters, metric=suite_metric, continuous=is_continuous)
    else:
        main()
