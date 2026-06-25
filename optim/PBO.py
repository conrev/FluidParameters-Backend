"""
Preferential Bayesian Optimization with Discrete Parameter Sets
===============================================================
Uses BoTorch's PairwiseGP (Laplace approximation) as the preference model
and the EUBO (Expected Utility of the Best Option) acquisition function to
choose the most informative next duel.

Domain: Hydrodynamic model boundary condition calibration.
Each candidate is a (Q_in, Q_out) pair of boundary discharges.
The oracle (or human expert) judges which of two configurations produces
a better match to observed flow behaviour at measurement points.

WORKFLOW
--------
1. Define a discrete grid of Q boundary condition configurations.
2. Present random pairs (warm-up) to collect initial preference data.
3. Fit a PairwiseGP on the pairwise comparison data.
4. Select the next duel via EUBO: for each candidate, compute
   E[max(f(challenger), f(current_best))] and pick the highest scorer.
5. Collect preference, refit, repeat until budget is exhausted.
6. Recommend the candidate with highest posterior mean utility.

INSTALLATION
------------
    pip install botorch gpytorch torch matplotlib

USAGE
-----
    # Simulated EUBO run:
    python preferential_bo_discrete.py

    # Compare EUBO vs random over many seeds:
    python preferential_bo_discrete.py --compare

    # Real human-in-the-loop run:
    python preferential_bo_discrete.py --human

References
----------
- Chu & Ghahramani (2005): Preference learning with Gaussian processes
- Lin et al. (2022): BoTorch preferential BO tutorial
- BoTorch PairwiseGP: botorch.models.pairwise_gp
"""

from __future__ import annotations

import argparse
import math
import warnings
from itertools import product as cartesian_product
from typing import Optional

import torch

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.double)
torch.manual_seed(0)

# ── BoTorch imports ──────────────────────────────────────────────────────────
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.fit import fit_gpytorch_mll
from botorch.models.pairwise_gp import (
    PairwiseGP,
    PairwiseLaplaceMarginalLogLikelihood,
)


PARAM_SPACE: dict[str, list] = {
    "Q_in_m3s":  list(map(lambda x: x/10.0, range(12, 248, 2))),  # m3/s upstream inflow
    "Q_out_m3s": list(map(lambda x: x/10.0, range(12, 248, 2))),  # m3/s downstream outflow
}

def build_candidate_tensor(
    param_space: dict[str, list],
) -> tuple[torch.Tensor, list[dict], torch.Tensor, torch.Tensor]:
    """
    Enumerate every configuration in the Cartesian product of `param_space`
    and return a min-max normalised (N, D) tensor.

    Returns
    -------
    X_norm   : (N, D) tensor with values in [0, 1]
    configs  : list of N dicts, one per row in X_norm
    x_min    : (D,)  raw per-dimension minima
    x_range  : (D,)  raw per-dimension ranges  (for inverse-normalisation)
    """
    keys    = list(param_space.keys())
    values  = list(param_space.values())
    combos  = list(cartesian_product(*values))
    configs = [dict(zip(keys, c)) for c in combos]

    raw     = torch.tensor(combos, dtype=torch.double)         # (N, D)
    x_min   = raw.min(0).values
    x_range = raw.max(0).values - x_min
    x_range[x_range == 0] = 1.0                                # guard / const dims
    X_norm  = (raw - x_min) / x_range

    return X_norm, configs, x_min, x_range


# ════════════════════════════════════════════════════════════════════════════
# 3 ▷  Oracle  (replace / extend for human-in-the-loop)
# ════════════════════════════════════════════════════════════════════════════

def _synthetic_utility(cfg: dict) -> float:
    """
    Ground-truth latent utility used for *simulation only*.
    This is never seen by the BO loop — it only drives the synthetic oracle.

    Simulates the NSE (Nash-Sutcliffe Efficiency) of a 1D hydrodynamic model
    run with the given boundary conditions, evaluated against synthetic
    "observed" water levels at three interior gauging stations.

    The oracle encodes two physical constraints that define the optimum:

      1. TARGET INFLOW  — Q_in ~ 500 m3/s
         The upstream gauge record, corrected for rating-curve bias, points
         to 500 m3/s as the best-estimate steady inflow. Deviation in either
         direction degrades the simulated flood peak at gauge 1.

      2. CONTINUITY RATIO  — Q_out / Q_in ~ 0.80
         Roughly 20% of the inflow is attenuated by floodplain storage and
         losses between the two boundaries. A ratio too close to 1.0
         (no attenuation) over-predicts downstream levels; too far below 0.8
         under-predicts them. This is the dominant control on gauge 2 and 3.

    The utility surface is deliberately smooth but asymmetric:
      - The Q_in penalty uses log-scale (a 2x error matters as much upstream
        as downstream in flow space).
      - The continuity penalty is Huber-shaped: quadratic near the target,
        then linear once the ratio deviates by more than 0.15.
    """
    q_in  = cfg["Q_in_m3s"]
    q_out = cfg["Q_out_m3s"]

    # 1. Inflow target: peaks at Q_in = 24.0 m3/s (log-scale distance)
    inflow_score = -abs(math.log(q_in / 24.0))

    # 2. Continuity ratio: peaks at Q_out / Q_in = 0.80
    ratio     = q_out / q_in
    ratio_err = ratio - 0.80
    delta     = 0.15
    if abs(ratio_err) <= delta:
        continuity_score = -(ratio_err ** 2) / (2 * delta)
    else:
        continuity_score = -(abs(ratio_err) - delta / 2)

    return inflow_score + continuity_score


def query_preference(
    idx_a: int,
    idx_b: int,
    configs: list[dict],
    use_human: bool = False,
) -> int:
    """
    Ask which of two configurations is preferred.

    Parameters
    ----------
    idx_a, idx_b : global indices into `configs`
    configs      : full list of config dicts
    use_human    : True  -> prompt the user interactively
                   False -> use the synthetic oracle (for testing)

    Returns
    -------
    Global index of the preferred candidate (idx_a or idx_b).
    """
    if use_human:
        print(f"\n  A : {_fmt(configs[idx_a])}")
        print(f"  B : {_fmt(configs[idx_b])}")
        while True:
            choice = input("  Which do you prefer? (A/B) : ").strip().upper()
            if choice in ("A", "B"):
                break
            print("  Please enter A or B.")
        return idx_a if choice == "A" else idx_b
    else:
        ua = _synthetic_utility(configs[idx_a])
        ub = _synthetic_utility(configs[idx_b])
        return idx_a if ua >= ub else idx_b


# ════════════════════════════════════════════════════════════════════════════
# 4 ▷  Preference model
# ════════════════════════════════════════════════════════════════════════════

def fit_preference_model(
    datapoints:  torch.Tensor,   # (M, D)  - candidates seen so far
    comparisons: torch.Tensor,   # (K, 2)  - [winner_local_idx, loser_local_idx]
) -> PairwiseGP:
    """
    Fit (or re-fit) a PairwiseGP on all collected pairwise comparisons.

    The PairwiseGP uses a Laplace approximation over a Bernoulli likelihood
    to learn a latent utility function f: X -> R such that
        P(x_i succ x_j) = sigma(f(x_i) - f(x_j)).

    Parameters
    ----------
    datapoints  : (M, D) tensor of the M distinct candidates seen so far
    comparisons : (K, 2) LongTensor where comparisons[k] = [winner_idx, loser_idx]
                  and indices are *local* (into `datapoints`).

    Returns
    -------
    Fitted PairwiseGP in eval mode.
    """
    model = PairwiseGP(datapoints, comparisons, jitter=1e-4)
    mll   = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
    try:
        fit_gpytorch_mll(mll)
    except Exception as e1:
        print(f"Warning > Laplace approximation Failed: {e1}")
        # Degenerate comparison data (e.g. from random sampling) can cause
        # the Laplace approximation to fail. Retry with higher jitter.
        model = PairwiseGP(datapoints, comparisons, jitter=1e-2)
        mll   = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
        try:
            fit_gpytorch_mll(mll)
        except Exception as e2:
            # Return partially-fitted model; posterior will be near-prior
            print(f"Failed Model Fitting: {e2}")
    model.eval()
    return model


# ════════════════════════════════════════════════════════════════════════════
# 5a ▷  EUBO duel selection
# ════════════════════════════════════════════════════════════════════════════

def select_next_duel(
    model:             PairwiseGP,
    all_X:             torch.Tensor,          # (N, D) full discrete space
    prev_winner_idx:   Optional[int] = None,  # global index of current best
    batch_size:        int = 256,             # max candidates evaluated at once
) -> tuple[int, int]:
    """
    Select the next duel (challenger, reference) to present using EUBO.

    When `prev_winner_idx` is provided (BO phase):
        EUBO is evaluated as E[max(f(challenger), f(prev_winner))]
        for every candidate; the highest scorer becomes the challenger.

    When `prev_winner_idx` is None (rarely needed):
        EUBO is evaluated for all pairs (up to `batch_size`); the
        highest-scoring pair is selected.

    Returns
    -------
    (challenger_global_idx, reference_global_idx)
    """
    N = len(all_X)

    # ── BO phase: challenger vs. known winner ────────────────────────────────
    if prev_winner_idx is not None:
        prev_x = all_X[[prev_winner_idx]]                    # (1, D)
        acqf   = AnalyticExpectedUtilityOfBestOption(
            pref_model=model,
            previous_winner=prev_x,
        )
        X = all_X.unsqueeze(1)                               # (N, 1, D)

        best_val = torch.tensor(-torch.inf)
        best_idx = -1
        for start in range(0, N, batch_size):
            chunk = X[start : start + batch_size]
            with torch.no_grad():
                vals = acqf(chunk)
            offset = torch.zeros(len(chunk))
            if start <= prev_winner_idx < start + len(chunk):
                offset[prev_winner_idx - start] = torch.inf
            vals = vals - offset
            local_best = vals.argmax()
            if vals[local_best] > best_val:
                best_val = vals[local_best]
                best_idx = start + local_best.item()

        return best_idx, prev_winner_idx

    # ── Cold-start fallback: enumerate all pairs ─────────────────────────────
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    if len(pairs) > batch_size:
        chosen = torch.randperm(len(pairs))[:batch_size].tolist()
        pairs  = [pairs[k] for k in chosen]

    X_pairs = torch.stack(
        [torch.stack([all_X[i], all_X[j]]) for i, j in pairs]
    )  # (P, 2, D)

    acqf = AnalyticExpectedUtilityOfBestOption(pref_model=model)
    with torch.no_grad():
        vals = acqf(X_pairs)
    best_pair = pairs[int(vals.argmax())]
    return best_pair[0], best_pair[1]


# ════════════════════════════════════════════════════════════════════════════
# 5b ▷  Random duel selection (baseline)
# ════════════════════════════════════════════════════════════════════════════

def select_next_duel_random(
    all_candidates: torch.Tensor,   # (N, D) full discrete space
) -> tuple[int, int]:
    """
    Select the next duel by picking two distinct candidates uniformly at
    random. This is the baseline against which EUBO is compared: it uses
    the same PairwiseGP for the final recommendation but makes no attempt
    to target informative pairs during data collection.
    """
    N = len(all_candidates)
    a, b = torch.randperm(N)[:2].tolist()
    return a, b


# ════════════════════════════════════════════════════════════════════════════
# 6 ▷  Main preferential BO loop
# ════════════════════════════════════════════════════════════════════════════

def run_preferential_bo(
    param_space:  dict[str, list] = PARAM_SPACE,
    n_init:       int  = 4,
    n_iterations: int  = 12,
    use_human:    bool = False,
    verbose:      bool = True,
    method:       str  = "eubo",   # "eubo" or "random"
) -> tuple[dict, list[float]]:
    """
    Full preferential BO loop over a discrete parameter space.

    Parameters
    ----------
    param_space   : mapping of param name -> list of discrete values
    n_init        : number of *random* warm-up comparisons  (should be >= 4)
    n_iterations  : number of acquisition-guided comparisons after warm-up
    use_human     : True -> prompt the user for preferences interactively
    verbose       : print progress to stdout
    method        : "eubo"   -- use EUBO acquisition to pick duels (default)
                    "random" -- pick duels uniformly at random (baseline)

    Returns
    -------
    best_cfg      : recommended config dict
    gp_util_trace : utility of the GP's top-ranked recommendation after every
                    comparison (warm-up + BO); the metric used in comparisons
    """
    assert method in ("eubo", "random"), "method must be 'eubo' or 'random'"

    all_X, configs, _, _ = build_candidate_tensor(param_space)
    N, D = all_X.shape

    if verbose:
        tag = "EUBO" if method == "eubo" else "Random baseline"
        print(f"[{tag}]  Search space: {N:,} configs x {D} dimensions")
        print(f"Warm-up: {n_init} random duels  |  BO: {n_iterations} {tag} duels\n")

    seen_globals:  list[int]             = []
    comps_local:   list[tuple[int, int]] = []
    gp_util_trace: list[float]           = []
    prev_winner:   Optional[int]         = None

    def g2l(g: int) -> int:
        if g not in seen_globals:
            seen_globals.append(g)
        return seen_globals.index(g)

    def record_duel(a: int, b: int, label: str = "") -> int:
        w  = query_preference(a, b, configs, use_human)
        lo = b if w == a else a
        comps_local.append((g2l(w), g2l(lo)))
        if verbose:
            side = "A" if w == a else "B"
            print(f"  {label}  A={_fmt(configs[a])}  B={_fmt(configs[b])}  -> {side} wins")
        return w

    def _gp_best_utility(model: PairwiseGP) -> float:
        """Utility of the config the GP currently thinks is best."""
        with torch.no_grad():
            mean = model.posterior(all_X).mean.squeeze(-1)
        return _synthetic_utility(configs[int(mean.argmax())])

    # ── Warm-up: random duels ────────────────────────────────────────────────
    if verbose:
        print("-- Warm-up (random duels) ---------------------------------------")

    n_init = max(4, n_init + n_init % 2)
    perm   = torch.randperm(N).tolist()

    for k in range(0, n_init, 2):
        a, b = perm[k], perm[k + 1]
        w    = record_duel(a, b, label=f"init {k // 2 + 1:2d}")
        prev_winner = w
        if not use_human:
            dp  = all_X[seen_globals]
            ct  = torch.tensor(comps_local, dtype=torch.long)
            mdl = fit_preference_model(dp, ct)
            gp_util_trace.append(_gp_best_utility(mdl))

    # ── Acquisition loop ─────────────────────────────────────────────────────
    acq_label = "EUBO" if method == "eubo" else "rand"
    if verbose:
        print(f"\n-- {acq_label.upper()}-guided duels ------------------------------------------")

    for it in range(1, n_iterations + 1):
        dp    = all_X[seen_globals]
        ct    = torch.tensor(comps_local, dtype=torch.long)
        model = fit_preference_model(dp, ct)

        if method == "eubo":
            challenger, reference = select_next_duel(
                model, all_X, prev_winner_idx=prev_winner
            )
        else:
            challenger, reference = select_next_duel_random(all_X)

        w = record_duel(challenger, reference, label=f"{acq_label} {it:2d}")
        prev_winner = w

        if not use_human:
            gp_util_trace.append(_gp_best_utility(model))

    # ── Final recommendation ─────────────────────────────────────────────────
    dp    = all_X[seen_globals]
    ct    = torch.tensor(comps_local, dtype=torch.long)
    model = fit_preference_model(dp, ct)
    with torch.no_grad():
        mean_u = model.posterior(all_X).mean.squeeze(-1)

    best_global = int(mean_u.argmax())
    best_cfg    = configs[best_global]

    if verbose:
        print("\n-- Recommendation -----------------------------------------------")
        print(f"  Best config  : {best_cfg}")
        if not use_human:
            true_best_idx = max(range(N), key=lambda i: _synthetic_utility(configs[i]))
            print(f"  True utility : {_synthetic_utility(best_cfg):.4f}")
            print(f"  Global optimum : {configs[true_best_idx]}")
            print(f"  Optimal utility: {_synthetic_utility(configs[true_best_idx]):.4f}")
            match = "MATCH" if best_global == true_best_idx else "MISMATCH"
            print(f"  {match}")

    return best_cfg, gp_util_trace

def plot_convergence(
    utility_trace: list[float],
    param_space:   dict[str, list],
    save_path:     str = "convergence.png",
) -> None:
    """Plot the GP recommendation utility of a single run."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed -- skipping convergence plot.")
        return

    _, configs, _, _ = build_candidate_tensor(param_space)
    N = len(configs)
    true_best = max(_synthetic_utility(c) for c in configs)

    running_best = np.maximum.accumulate(utility_trace)
    iters = range(1, len(running_best) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iters, running_best, "o-", color="#2563eb", lw=2,
            label="GP recommendation utility")
    ax.axhline(true_best, ls="--", color="#16a34a", lw=1.5,
               label=f"Global optimum ({true_best:.2f})")
    ax.fill_between(iters, running_best, true_best, alpha=0.08, color="#2563eb")
    ax.set_xlabel("Number of comparisons (warm-up + BO)")
    ax.set_ylabel("Utility (NSE surrogate)")
    ax.set_title(f"Preferential BO: Q boundary condition calibration  |  {N} candidates")
    ax.legend()
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"\nConvergence plot saved -> {save_path}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# 9 ▷  Utilities
# ════════════════════════════════════════════════════════════════════════════

def _fmt(cfg: dict) -> str:
    """Compact single-line representation of a config dict."""
    return "{" + ", ".join(f"{k}={v}" for k, v in cfg.items()) + "}"


def get_posterior_rankings(
    param_space: dict[str, list],
    model: PairwiseGP,
    top_k: int = 5,
) -> list[tuple[int, dict, float]]:
    """
    Return the top-k candidates ranked by posterior mean utility.

    Parameters
    ----------
    param_space : same space used during the BO run
    model       : fitted PairwiseGP in eval mode
    top_k       : number of candidates to return

    Returns
    -------
    List of (global_index, config_dict, posterior_mean) sorted descending.
    """
    all_X, configs, _, _ = build_candidate_tensor(param_space)
    with torch.no_grad():
        mean = model.posterior(all_X).mean.squeeze(-1)
    ranked = sorted(range(len(configs)), key=lambda i: -mean[i].item())
    return [(i, configs[i], mean[i].item()) for i in ranked[:top_k]]


# ════════════════════════════════════════════════════════════════════════════
# 10 ▷  Entry point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preferential BO over discrete Q boundary conditions"
    )
    parser.add_argument("--human",   action="store_true", help="Use real human preferences")
    parser.add_argument("--compare", action="store_true", help="EUBO vs random comparison plot")
    parser.add_argument("--n_init",  type=int, default=4,  help="Warm-up comparisons")
    parser.add_argument("--n_iter",  type=int, default=30, help="BO iterations")
    parser.add_argument("--n_seeds", type=int, default=30, help="Seeds for comparison")
    parser.add_argument("--seed",    type=int, default=0,  help="Seed for single run")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    best_cfg, trace = run_preferential_bo(
        param_space  = PARAM_SPACE,
        n_init       = args.n_init,
        n_iterations = args.n_iter,
        use_human    = args.human,
        verbose      = True,
        method       = "eubo",
    )
    if not args.human and trace:
        plot_convergence(trace, PARAM_SPACE)
