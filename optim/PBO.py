from __future__ import annotations

import warnings
from itertools import product as cartesian_product
from typing import Optional
import asyncio
import uuid
import torch
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.fit import fit_gpytorch_mll
from botorch.models.pairwise_gp import (
    PairwiseGP,
    PairwiseLaplaceMarginalLogLikelihood,
)
from botorch.optim import optimize_acqf_discrete

from optim.reporting import build_result

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.double)
torch.manual_seed(0)

PARAM_SPACE: dict[str, list] = {
    "Qin": list(map(lambda x: x / 10.0, range(12, 248, 2))),  # m3/s upstream inflow
    "Qout": list(map(lambda x: x / 10.0, range(12, 248, 2))),  # m3/s downstream outflow
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
    keys = list(param_space.keys())
    values = list(param_space.values())
    combos = list(cartesian_product(*values))
    configs = [dict(zip(keys, c)) for c in combos]

    raw = torch.tensor(combos, dtype=torch.double)  # (N, D)
    x_min = raw.min(0).values
    x_range = raw.max(0).values - x_min
    x_range[x_range == 0] = 1.0  # guard / const dims
    X_norm = (raw - x_min) / x_range

    return X_norm, configs, x_min, x_range


def fit_preference_model(
    datapoints: torch.Tensor,  # (M, D)  - candidates seen so far
    comparisons: torch.Tensor,  # (K, 2)  - [winner_local_idx, loser_local_idx]
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
    mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
    try:
        fit_gpytorch_mll(mll)
    except Exception as e1:
        print(f"Warning > Laplace approximation Failed: {e1}")
        # Degenerate comparison data (e.g. from random sampling) can cause
        # the Laplace approximation to fail. Retry with higher jitter.
        model = PairwiseGP(datapoints, comparisons, jitter=1e-2)
        mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
        try:
            fit_gpytorch_mll(mll)
        except Exception as e2:
            # Return partially-fitted model; posterior will be near-prior
            print(f"Failed Model Fitting: {e2}")
    model.eval()
    return model


def _global_index_of(point: torch.Tensor, all_X: torch.Tensor) -> int:
    """Map a candidate point (returned by the optimiser) back to its row index in the grid."""
    return int(torch.cdist(point.reshape(1, -1), all_X).argmin())


def select_next_duel(
    model: PairwiseGP,
    all_X: torch.Tensor,  # (N, D) full discrete space
    prev_winner_idx: Optional[int] = None,  # global index of current best (the incumbent)
    max_batch_size: int = 2048,  # candidates evaluated per batch (memory only)
) -> tuple[int, int]:
    """
    Select the next duel (challenger, reference) via EXACT discrete EUBO maximisation.

    The analytic EUBO acquisition is evaluated at every candidate on the discrete grid and the
    exact argmax is returned — delegated to ``botorch.optim.optimize_acqf_discrete`` (no optimiser
    restarts, no local optima). This is the correct, exact approach for a small/low-dimensional
    discrete space; for very large or continuous spaces switch to continuous ``optimize_acqf`` or a
    discrete local-search variant.

    The reference is the current incumbent: ``prev_winner_idx`` when known, otherwise the argmax of
    the posterior mean — so the cold start uses the same exact O(N) path, not a random pair sample.
    The incumbent is excluded from the candidate set so it cannot duel itself.

    Returns
    -------
    (challenger_global_idx, reference_global_idx)
    """
    if prev_winner_idx is None:
        with torch.no_grad():
            mean = model.posterior(all_X).mean.squeeze(-1)
        prev_winner_idx = int(mean.argmax())

    acqf = AnalyticExpectedUtilityOfBestOption(
        pref_model=model,
        previous_winner=all_X[[prev_winner_idx]],  # (1, D)
    )
    keep = torch.ones(len(all_X), dtype=torch.bool)
    keep[prev_winner_idx] = False  # the incumbent can't challenge itself
    choices = all_X[keep]

    candidate, _ = optimize_acqf_discrete(
        acqf, q=1, choices=choices, max_batch_size=max_batch_size, unique=True
    )
    challenger_idx = _global_index_of(candidate, all_X)
    return challenger_idx, prev_winner_idx


def select_next_duel_ei(
    model: PairwiseGP,
    all_X: torch.Tensor,  # (N, D) full discrete space
    prev_winner_idx: Optional[int] = None,  # global index of the incumbent
    max_batch_size: int = 2048,
) -> tuple[int, int]:
    """
    PBO-EI baseline: pick the challenger by exact discrete maximisation of Expected Improvement of
    the latent utility over the incumbent's posterior mean, then duel it against the incumbent.

    Structurally identical to `select_next_duel` (same incumbent, same exhaustive
    `optimize_acqf_discrete`, same exclusion of the incumbent) — only the acquisition differs
    (Expected Improvement instead of EUBO). Reference is the incumbent = `prev_winner_idx` when
    known, else the argmax of the posterior mean.
    """
    if prev_winner_idx is None:
        with torch.no_grad():
            mean = model.posterior(all_X).mean.squeeze(-1)
        prev_winner_idx = int(mean.argmax())

    with torch.no_grad():
        best_f = model.posterior(all_X[[prev_winner_idx]]).mean.squeeze().detach()
    acqf = ExpectedImprovement(model, best_f=best_f)
    keep = torch.ones(len(all_X), dtype=torch.bool)
    keep[prev_winner_idx] = False  # the incumbent can't challenge itself
    choices = all_X[keep]

    candidate, _ = optimize_acqf_discrete(
        acqf, q=1, choices=choices, max_batch_size=max_batch_size, unique=True
    )
    return _global_index_of(candidate, all_X), prev_winner_idx


def select_next_duel_random(
    all_candidates: torch.Tensor,  # (N, D) full discrete space
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


WARMUP_METHODS = ("sobol", "lhs", "random")


def _latin_hypercube(n: int, d: int, seed: Optional[int] = None) -> torch.Tensor:
    """
    Latin Hypercube design of `n` points in [0, 1]^d: each dimension is split into `n` equal
    strata and sampled exactly once, with a random jitter inside each stratum. Better per-axis
    coverage than i.i.d. uniform. Pure-torch (no scipy dependency).
    """
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    pts = torch.empty(n, d, dtype=torch.double)
    for j in range(d):
        perm = torch.randperm(n, generator=gen).double()
        jitter = torch.rand(n, generator=gen, dtype=torch.double)
        pts[:, j] = (perm + jitter) / n
    return pts


def build_warmup_sequence(
    all_X: torch.Tensor,  # (N, D) normalised candidate grid in [0, 1]
    n_points: int,  # number of distinct candidates to seed the warm-up with
    method: str = "sobol",
    seed: Optional[int] = None,
) -> list[int]:
    """
    Pick `n_points` distinct grid indices to seed the warm-up, using a space-filling design
    snapped to the discrete candidate grid. Consumed pairwise as warm-up duels.

    - "sobol":  low-discrepancy Sobol sequence — even, low-clumping coverage (recommended default).
    - "lhs":    Latin Hypercube — stratified once per dimension.
    - "random": uniform random permutation (the original baseline behaviour).

    Space-filling points are generated in [0, 1]^D and greedily snapped to the nearest *unused*
    candidate, so the returned indices are always distinct real grid points.
    """
    assert method in WARMUP_METHODS, f"warmup must be one of {WARMUP_METHODS}, got {method!r}"
    N, D = all_X.shape
    n_points = min(n_points, N)

    if method == "random":
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        return torch.randperm(N, generator=gen)[:n_points].tolist()

    if method == "sobol":
        engine = torch.quasirandom.SobolEngine(dimension=D, scramble=True, seed=seed)
        pts = engine.draw(n_points).to(all_X)
    else:  # "lhs"
        pts = _latin_hypercube(n_points, D, seed=seed).to(all_X)

    # Greedily snap each design point to the nearest candidate not already chosen (distinct set).
    dist = torch.cdist(pts, all_X)  # (n_points, N)
    used = torch.zeros(N, dtype=torch.bool)
    chosen: list[int] = []
    for i in range(n_points):
        row = dist[i].clone()
        row[used] = float("inf")
        idx = int(row.argmin())
        used[idx] = True
        chosen.append(idx)
    return chosen


class PreferentialBOSession:
    """
    Parameters
    ----------
    param_space   : discrete parameter grid (see PARAM_SPACE)
    n_init        : warm-up comparisons (rounded up to even, min 4)
    n_iterations  : EUBO- or random-guided comparisons after warm-up
    method        : "eubo" (default), "ei" (PBO-EI baseline), or "random" — BO-phase duel selection
    warmup        : warm-up design — "sobol" (default), "lhs", or "random" (see WARMUP_METHODS)
    seed          : optional RNG seed for reproducible warm-up order
    """

    def __init__(
        self,
        param_space: dict[str, list],
        n_init: int = 4,
        n_iterations: int = 12,
        method: str = "eubo",
        warmup: str = "sobol",
        seed: Optional[int] = None,
    ) -> None:
        assert method in ("eubo", "random", "ei")
        assert warmup in WARMUP_METHODS, f"warmup must be one of {WARMUP_METHODS}, got {warmup!r}"
        if seed is not None:
            torch.manual_seed(seed)

        self.param_space = param_space
        self.n_warmup = n_init
        self.n_iterations = n_iterations
        self.method = method
        self.warmup = warmup
        self.total_duels = self.n_warmup + n_iterations

        self.all_X, self.configs, _, _ = build_candidate_tensor(param_space)
        self.N = len(self.configs)

        # ── BO state ─────────────────────────────────────────────────────────
        self.seen_globals: list[int] = []
        self.comps_local: list[tuple[int, int]] = []
        self.prev_winner: Optional[int] = None
        self.model: Optional[PairwiseGP] = None

        # ── Session state ─────────────────────────────────────────────────────
        self._phase: str = "warmup"
        self._warmup_step: int = 0
        self._bo_step: int = 0
        self._duels_done: int = 0
        self._started: bool = False
        self._pending: dict[str, tuple[int, int]] = {}
        # Space-filling warm-up: pick 2 candidates per warm-up duel via the chosen design.
        self._warmup_perm: list[int] = build_warmup_sequence(
            self.all_X, 2 * self.n_warmup, method=warmup, seed=seed
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _g2l(self, g: int) -> int:
        if g not in self.seen_globals:
            self.seen_globals.append(g)
        return self.seen_globals.index(g)

    def _record(self, idx_a: int, idx_b: int, choice: str) -> int:
        winner = idx_a if choice == "A" else idx_b
        loser = idx_b if choice == "A" else idx_a
        self.comps_local.append((self._g2l(winner), self._g2l(loser)))
        return winner

    def _refit(self) -> None:
        if len(self.comps_local) < 2:
            return
        dp = self.all_X[self.seen_globals]
        ct = torch.tensor(self.comps_local, dtype=torch.long)
        self.model = fit_preference_model(dp, ct)

    def _next_bo_pair(self) -> tuple[int, int]:
        if self.method == "eubo":
            return select_next_duel(self.model, self.all_X, self.prev_winner)
        if self.method == "ei":
            return select_next_duel_ei(self.model, self.all_X, self.prev_winner)
        return select_next_duel_random(self.all_X)

    def _make_duel(self, idx_a: int, idx_b: int, phase: str) -> dict:
        duel_id = str(uuid.uuid4())
        self._pending[duel_id] = (idx_a, idx_b)
        return {
            "type": "duel",
            "duelId": duel_id,
            "phase": phase,
            "progress": {
                "current": self._duels_done + 1,
                "total": self.total_duels,
            },
            "optionA": self.configs[idx_a],
            "optionB": self.configs[idx_b],
        }

    def _make_result(self) -> dict:
        """
        Report the single best reconstruction PLUS the distinct "similarly good" scenarios.

        The cheap mean-argmax over the full duel grid is the single-best anchor / fallback; the
        scenario clustering, per-parameter credible ranges, convergence readout and labeling are
        delegated to `optim.reporting.build_result` (guide §L1). See that module for the honesty
        constraints (relative tolerances, shares as Monte-Carlo estimates, Laplace overconfidence).
        """
        self._refit()

        best_config: Optional[dict] = None
        if self.model is not None:
            with torch.no_grad():
                mean = self.model.posterior(self.all_X).mean.squeeze(-1)
            best_config = self.configs[int(mean.argmax())]

        return build_result(
            model=self.model,
            param_space=self.param_space,
            total_comparisons=self._duels_done,
            best_config=best_config,
        )

    # ── Public API  (sync) ───────────────────────────────────────────────────

    def start(self) -> dict:
        """
        Initialise the session and return the first duel request.

        Must be called exactly once before any submit_preference() calls.
        Returns a dict with type="duel".
        """
        if self._started:
            raise RuntimeError("Session already started.")
        self._started = True
        k = self._warmup_step * 2
        return self._make_duel(
            self._warmup_perm[k], self._warmup_perm[k + 1], phase="warmup"
        )

    def submit_preference(self, duel_id: str, choice: str) -> dict:
        """
        Record a human preference and advance the BO by one step.

        Parameters
        ----------
        duel_id : str      the duel_id field from the last duel message
        choice  : "A"|"B"  which option the user preferred

        Returns
        -------
        dict  — next duel (type="duel") or final result (type="result")

        Raises
        ------
        ValueError  — unknown duel_id or invalid choice
        RuntimeError — session not started or already finished
        """
        if not self._started:
            raise RuntimeError("Call start() before submit_preference().")
        if duel_id not in self._pending:
            raise ValueError(f"Unknown duel_id: {duel_id!r}")
        if choice not in ("A", "B"):
            raise ValueError(f"choice must be 'A' or 'B', got: {choice!r}")

        idx_a, idx_b = self._pending.pop(duel_id)
        self.prev_winner = self._record(idx_a, idx_b, choice)
        self._duels_done += 1

        # ── Warm-up phase ────────────────────────────────────────────────────
        if self._phase == "warmup":
            self._refit()
            self._warmup_step += 1

            if self._warmup_step < self.n_warmup:
                k = self._warmup_step * 2
                return self._make_duel(
                    self._warmup_perm[k], self._warmup_perm[k + 1], phase="warmup"
                )

            # Warmup complete — switch to BO
            self._phase = "bo"
            if self.n_iterations == 0:
                return self._make_result()
            self._refit()
            return self._make_duel(*self._next_bo_pair(), phase="bo")

        # ── BO phase ─────────────────────────────────────────────────────────
        self._refit()
        self._bo_step += 1

        if self._bo_step >= self.n_iterations:
            return self._make_result()
        return self._make_duel(*self._next_bo_pair(), phase="bo")

    # ── Public API  (async) ──────────────────────────────────────────────────

    async def start_async(self) -> dict:
        """
        Async variant of start().
        Offloads candidate selection to the thread-pool executor so the
        event loop is not blocked.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.start)

    async def submit_preference_async(self, duel_id: str, choice: str) -> dict:
        """
        Async variant of submit_preference().
        Model fitting runs in the thread-pool executor; the event loop
        remains free to handle other connections while the GP is being fit.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.submit_preference, duel_id, choice)
