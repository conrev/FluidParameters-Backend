from __future__ import annotations

import warnings
from itertools import product as cartesian_product
from typing import Optional
import asyncio
import uuid
import torch
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.fit import fit_gpytorch_mll
from botorch.models.pairwise_gp import (
    PairwiseGP,
    PairwiseLaplaceMarginalLogLikelihood,
)

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


def select_next_duel(
    model: PairwiseGP,
    all_X: torch.Tensor,  # (N, D) full discrete space
    prev_winner_idx: Optional[int] = None,  # global index of current best
    batch_size: int = 256,  # max candidates evaluated at once
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
        prev_x = all_X[[prev_winner_idx]]  # (1, D)
        acqf = AnalyticExpectedUtilityOfBestOption(
            pref_model=model,
            previous_winner=prev_x,
        )
        X = all_X.unsqueeze(1)  # (N, 1, D)

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
        pairs = [pairs[k] for k in chosen]

    X_pairs = torch.stack(
        [torch.stack([all_X[i], all_X[j]]) for i, j in pairs]
    )  # (P, 2, D)

    acqf = AnalyticExpectedUtilityOfBestOption(pref_model=model)
    with torch.no_grad():
        vals = acqf(X_pairs)
    best_pair = pairs[int(vals.argmax())]
    return best_pair[0], best_pair[1]


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


class PreferentialBOSession:
    """
    Parameters
    ----------
    param_space   : discrete parameter grid (see PARAM_SPACE)
    n_init        : warm-up comparisons (rounded up to even, min 4)
    n_iterations  : EUBO- or random-guided comparisons after warm-up
    method        : "eubo" (default) or "random" (baseline)
    top_k         : candidates to include in the final rankings
    seed          : optional RNG seed for reproducible warm-up order
    """

    def __init__(
        self,
        param_space: dict[str, list],
        n_init: int = 4,
        n_iterations: int = 12,
        method: str = "eubo",
        seed: Optional[int] = None,
    ) -> None:
        assert method in ("eubo", "random")
        if seed is not None:
            torch.manual_seed(seed)

        self.param_space = param_space
        self.n_warmup = n_init
        self.n_iterations = n_iterations
        self.method = method
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
        self._warmup_perm: list[int] = torch.randperm(self.N).tolist()

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
        self._refit()
        if self.model is not None:
            with torch.no_grad():
                mean = self.model.posterior(self.all_X).mean.squeeze(-1)
            ranked = sorted(range(self.N), key=lambda i: -mean[i].item())
        else:
            ranked = list(range(self.N))
            mean = torch.zeros(self.N)

        # rankings = [
        #     {
        #         "rank": r + 1,
        #         "config": self.configs[ranked[r]],
        #         "posterior_mean": float(mean[ranked[r]]),
        #     }
        #     for r in range(min(self.top_k, self.N))
        # ]
        return {
            "type": "result",
            "optimalParameter": self.configs[ranked[0]],
            "totalComparison": self._duels_done,
            # "rankings": rankings,
        }

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
