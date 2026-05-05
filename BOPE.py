from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.fit import fit_gpytorch_mll
from botorch.models.deterministic import FixedSingleSampleModel
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.pairwise_gp import (
    PairwiseGP,
    PairwiseLaplaceMarginalLogLikelihood,
)
from botorch.models.transforms.input import Normalize
from botorch.optim.optimize import optimize_acqf_discrete
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


def extract_features(sim_output: dict) -> np.ndarray:
    """
    Convert raw simulation output into a fixed-length feature vector Y.
    Args:
        sim_output: dict with keys from shallow water solver output,
                    e.g. {"depth_field": ..., "velocity_field": ..., "timestamps": ...}

    Returns:
        np.ndarray of shape (Y_DIM,) — the feature vector for this run
    """
    #  Replace with actual feature extraction logic later, e.g. take the depth out of simulation result output in hdf5
    features = np.array([
        sim_output.get("max_flood_extent", 0.0),
        sim_output.get("fill_time_zone_A", 0.0),
        sim_output.get("fill_time_zone_B", 0.0),
        sim_output.get("max_depth_centroid", 0.0),
        sim_output.get("flow_duration", 0.0),
        sim_output.get("spatial_symmetry", 0.0),
    ], dtype=np.float32)
    return features

class SimulationLibrary:
    """
    Manages the precomputed simulation library.

    Expected CSV format:
        theta_0, theta_1, ..., theta_9, feat_0, feat_1, ..., feat_k, run_id
    Or load thetas and features separately if you store simulation outputs elsewhere.
    """

    def __init__(self, theta_dim: int = 10, feature_dim: int = 6):
        self.theta_dim = theta_dim
        self.feature_dim = feature_dim
        self.thetas: Optional[torch.Tensor] = None   # (N, theta_dim)
        self.features: Optional[torch.Tensor] = None # (N, feature_dim)
        self.run_ids: list[str] = []

    def load_from_csv(self, path: str | Path):
        """Load library from a CSV with columns: theta_*, feat_*, run_id."""
        df = pd.read_csv(path)
        theta_cols = [c for c in df.columns if c.startswith("theta_")]
        feat_cols  = [c for c in df.columns if c.startswith("feat_")]

        assert len(theta_cols) == self.theta_dim, \
            f"Expected {self.theta_dim} theta columns, got {len(theta_cols)}"
        assert len(feat_cols) == self.feature_dim, \
            f"Expected {self.feature_dim} feature columns, got {len(feat_cols)}"

        self.thetas   = torch.tensor(df[theta_cols].values, dtype=torch.double)
        self.features = torch.tensor(df[feat_cols].values,  dtype=torch.double)
        self.run_ids  = df["run_id"].tolist() if "run_id" in df.columns else \
                        [str(i) for i in range(len(df))]
        print(f"Loaded library: {len(self.run_ids)} runs, "
              f"{self.theta_dim}D theta, {self.feature_dim}D features")

    def load_from_arrays(
        self,
        thetas: np.ndarray,
        features: np.ndarray,
        run_ids: Optional[list[str]] = None,
    ):
        """Load directly from numpy arrays (useful for testing)."""
        self.thetas   = torch.tensor(thetas,   dtype=torch.double)
        self.features = torch.tensor(features, dtype=torch.double)
        self.run_ids  = run_ids or [str(i) for i in range(len(thetas))]

    def make_synthetic(self, n: int = 200, seed: int = 42):
        """
        Generate a synthetic library for testing without real simulations.
        TODO: replace with real simulation data
        """
        rng = np.random.default_rng(seed)
        thetas   = rng.uniform(0, 1, size=(n, self.theta_dim)).astype(np.float32)
        # Synthetic features: smooth functions of theta for testing
        features = np.stack([
            np.sin(thetas[:, 0] * np.pi) * thetas[:, 1],
            thetas[:, 2] ** 2 + thetas[:, 3],
            np.exp(-thetas[:, 4]) * thetas[:, 5],
            thetas[:, 6] * thetas[:, 7],
            np.tanh(thetas[:, 8] - 0.5),
            thetas[:, 9] ** 0.5,
        ], axis=1).astype(np.float32)
        self.load_from_arrays(thetas, features)
        print(f"Generated synthetic library: {n} runs")

    @property
    def n(self) -> int:
        return len(self.run_ids)

    def get_run(self, idx: int) -> dict:
        return {
            "run_id": self.run_ids[idx],
            "theta":  self.thetas[idx].numpy(),
            "features": self.features[idx].numpy(),
        }

class HumanOracle:
    """
    Interface for collecting pairwise preference comparisons from a human.
    for now, a simple CLI interface
    TODO: 
    """

    def __init__(self, unity_endpoint: Optional[str] = None):
        """

        """
        self.unity_endpoint = unity_endpoint
        self.comparison_log: list[dict] = []

    def compare(
        self,
        run_id_a: str,
        run_id_b: str,
        features_a: np.ndarray,
        features_b: np.ndarray,
    ) -> int:
        if self.unity_endpoint is not None:
            return self._compare_via_unity(run_id_a, run_id_b)
        else:
            return self._compare_via_cli(run_id_a, run_id_b, features_a, features_b)

    def _compare_via_unity(self, run_id_a: str, run_id_b: str) -> int:
        """TODO
           contact unity to get comparison results via websocketz       
        """
    def _compare_via_cli(
        self,
        run_id_a: str,
        run_id_b: str,
        features_a: np.ndarray,
        features_b: np.ndarray,
    ) -> int:
        """Fallback: prompt human in terminal."""
        print("\n" + "=" * 60)
        print(f"  Run A  (id={run_id_a}): {np.round(features_a, 3)}")
        print(f"  Run B  (id={run_id_b}): {np.round(features_b, 3)}")
        print("=" * 60)
        while True:
            choice = input("Which simulation looks more correct? [A/B]: ").strip().upper()
            if choice in ("A", "B"):
                result = 0 if choice == "A" else 1
                self.comparison_log.append({
                    "run_a": run_id_a, "run_b": run_id_b, "preferred": choice
                })
                return result
            print("  Please enter A or B.")

    def save_log(self, path: str | Path):
        """Save all comparisons to JSON for reproducibility."""
        with open(path, "w") as f:
            json.dump(self.comparison_log, f, indent=2)
        print(f"Saved {len(self.comparison_log)} comparisons to {path}")

class BOPESolver:
    """
    BOPE over a precomputed simulation library.

    The solver needs:
      - outcome_model: SingleTaskGP fitting θ → Y (fitted once from library)
      - pref_model:    PairwiseGP fitting Y → utility (updated each round)
      - train_Y:       outcome vectors seen in comparisons so far
      - train_comps:   pairwise comparison results so far
    """

    def __init__(
        self,
        library: SimulationLibrary,
        oracle: HumanOracle,
    ):
        self.library = library
        self.oracle  = oracle

        # These are set during initialize()
        self.outcome_model: Optional[SingleTaskGP] = None
        self.pref_model:    Optional[PairwiseGP]   = None
        self.train_Y:       Optional[torch.Tensor] = None  # (n_comps*2, Y_dim)
        self.train_comps:   Optional[torch.Tensor] = None  # (n_comps, 2)

        self.round_log: list[dict] = []

    def fit_outcome_model(self) -> SingleTaskGP:
        """
        Fit the outcome model f: θ → Y over the full precomputed library.

        Since all runs are precomputed, we fit this once on the entire library
        rather than incrementally. This is the key advantage of your setup —
        the outcome model is fully informed from the start.
        """
        X = self.library.thetas    # (N, theta_dim)
        Y = self.library.features  # (N, feature_dim)

        model = SingleTaskGP(
            train_X=X,
            train_Y=Y,
            input_transform=Normalize(d=X.shape[-1]),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        print(f"Fitted outcome model on {X.shape[0]} library runs.")
        return model

    def initialize(self, n_init_comps: int = 3, seed: int = 0):
        """
        Initialize BOPE:
          1. Fit the outcome model on the full library (free, since precomputed)
          2. Collect n_init_comps random pairwise comparisons to warm-start the
             preference model
        """
        print("=== Initializing BOPE ===")
        self.outcome_model = self.fit_outcome_model()

        # Warm-start: sample random pairs from library and ask human
        print(f"Collecting {n_init_comps} initial comparisons...")
        rng = np.random.default_rng(seed)
        idxs = rng.choice(self.library.n, size=n_init_comps * 2, replace=False)

        Y_list, comps_list = [], []
        for i in range(n_init_comps):
            idx_a, idx_b = idxs[2 * i], idxs[2 * i + 1]
            Y_list, comps_list = self._collect_comparison(
                idx_a, idx_b, Y_list, comps_list
            )

        self.train_Y     = torch.cat(Y_list, dim=0)    # (n_init_comps*2, Y_dim)
        self.train_comps = torch.stack(comps_list, dim=0)  # (n_init_comps, 2)
        print(f"Initialization complete. {n_init_comps} comparisons collected.")

    def preference_exploration_round(self, n_comps: int = 5):
        """
        Preference Exploration (PE) stage:
        Use EUBO-ζ acquisition to select the most informative pair of outcomes
        to show the human, then collect their preference.
        """
        print(f"\n--- Preference Exploration: {n_comps} comparisons ---")

        for i in range(n_comps):
            # 1. Fit preference model on current comparisons
            pref_model = self._fit_pref_model()

            # 2. Use EUBO-ζ to select the most informative pair from library
            idx_a, idx_b = self._select_pair_eubo(pref_model)

            # 3. Collect human comparison
            Y_list    = [self.train_Y]
            comps_list = list(self.train_comps)
            Y_list, comps_list = self._collect_comparison(
                idx_a, idx_b, Y_list, comps_list
            )

            # 4. Update training data
            self.train_Y     = torch.cat(Y_list, dim=0)
            self.train_comps = torch.stack(comps_list, dim=0)

            print(f"  [{i+1}/{n_comps}] Compared run {self.library.run_ids[idx_a]} "
                  f"vs {self.library.run_ids[idx_b]}")

        # Cache the final preference model for this round
        self.pref_model = self._fit_pref_model()

    def recommend(self) -> tuple[np.ndarray, int, str]:
        """
        Experimentation stage:
        Given the current preference model, find the library run with the
        highest expected utility using qNEIUU over the discrete candidate set.

        Returns:
            (theta, library_index, run_id) of the recommended run
        """
        if self.pref_model is None:
            self.pref_model = self._fit_pref_model()

        with torch.no_grad():
            Y_all = self.library.features  # (N, Y_dim)
            utility_mean = self.pref_model.posterior(Y_all).mean.squeeze(-1)

        best_idx = int(utility_mean.argmax().item())
        best_theta  = self.library.thetas[best_idx].numpy()
        best_run_id = self.library.run_ids[best_idx]

        print(f"\n*** Recommendation: run_id={best_run_id}, idx={best_idx} ***")
        print(f"    theta = {np.round(best_theta, 4)}")
        print(f"    estimated utility = {utility_mean[best_idx]:.4f}")

        self.round_log.append({
            "recommended_run_id": best_run_id,
            "recommended_idx": best_idx,
            "estimated_utility": float(utility_mean[best_idx]),
            "n_comparisons": len(self.train_comps),
        })

        return best_theta, best_idx, best_run_id

    def top_k_recommendations(self, k: int = 5) -> pd.DataFrame:
        """Return top-k runs by estimated utility."""
        if self.pref_model is None:
            self.pref_model = self._fit_pref_model()

        with torch.no_grad():
            Y_all = self.library.features
            utility_mean = self.pref_model.posterior(Y_all).mean.squeeze(-1).numpy()
            utility_std  = self.pref_model.posterior(Y_all).variance.squeeze(-1).sqrt().numpy()

        top_idxs = np.argsort(utility_mean)[::-1][:k]
        rows = []
        for rank, idx in enumerate(top_idxs):
            rows.append({
                "rank": rank + 1,
                "run_id": self.library.run_ids[idx],
                "library_idx": idx,
                "utility_mean": round(float(utility_mean[idx]), 4),
                "utility_std":  round(float(utility_std[idx]),  4),
                **{f"theta_{j}": round(float(self.library.thetas[idx, j]), 4)
                   for j in range(self.library.theta_dim)},
            })
        return pd.DataFrame(rows)

    # helpers

    def _fit_pref_model(self) -> PairwiseGP:
        """Fit the preference model g on current comparison data."""
        model = PairwiseGP(
            self.train_Y,
            self.train_comps,
            input_transform=Normalize(d=self.train_Y.shape[-1]),
        )
        mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

    def _select_pair_eubo(self, pref_model: PairwiseGP) -> tuple[int, int]:
        """
        Use EUBO-ζ acquisition to select the most informative pair.

        EUBO-ζ works by:
          1. Drawing a sample ζ from the outcome model posterior (a FixedSingleSampleModel)
          2. Finding the pair of library candidates that maximizes expected
             utility-of-best-option under current preference uncertainty
        """
        # Sample one realization of the outcome model to condition on
        outcome_sampler = FixedSingleSampleModel(model=self.outcome_model)

        acqf = AnalyticExpectedUtilityOfBestOption(
            pref_model=pref_model,
            model=outcome_sampler,
        )

        # Candidate set: all (N, 1, theta_dim) library points
        # optimize_acqf_discrete picks the best pair from this set
        candidates = self.library.thetas.unsqueeze(1)  # (N, 1, theta_dim)

        # Optimize over discrete candidate set — returns 2 candidates (a pair)
        best_pair, _ = optimize_acqf_discrete(
            acq_function=acqf,
            q=2,
            choices=candidates,
        )

        # Find which library indices these correspond to
        idx_a = self._find_library_idx(best_pair[0])
        idx_b = self._find_library_idx(best_pair[1])
        return idx_a, idx_b

    def _find_library_idx(self, theta: torch.Tensor) -> int:
        """Find the library index of a given theta vector by nearest-neighbor."""
        dists = torch.norm(self.library.thetas - theta.unsqueeze(0), dim=-1)
        return int(dists.argmin().item())

    def _collect_comparison(
        self,
        idx_a: int,
        idx_b: int,
        Y_list: list,
        comps_list: list,
    ) -> tuple[list, list]:
        """
        Show runs idx_a and idx_b to the human oracle, collect preference,
        and append to the running Y and comps lists.

        Comparison encoding (BoTorch convention):
          comps[i] = [winner_idx, loser_idx] in the Y tensor
        """
        run_a = self.library.get_run(idx_a)
        run_b = self.library.get_run(idx_b)

        preferred = self.oracle.compare(
            run_a["run_id"], run_b["run_id"],
            run_a["features"], run_b["features"],
        )

        # Append the two outcome vectors as new rows in train_Y
        y_a = self.library.features[idx_a].unsqueeze(0)  # (1, Y_dim)
        y_b = self.library.features[idx_b].unsqueeze(0)  # (1, Y_dim)

        # Track the absolute indices in the growing Y tensor
        current_len = sum(y.shape[0] for y in Y_list)
        abs_idx_a   = current_len
        abs_idx_b   = current_len + 1

        if preferred == 0:  # A preferred
            comp = torch.tensor([[abs_idx_a, abs_idx_b]])
        else:               # B preferred
            comp = torch.tensor([[abs_idx_b, abs_idx_a]])

        Y_list.append(y_a)
        Y_list.append(y_b)
        comps_list.append(comp.squeeze(0))

        return Y_list, comps_list

    def save_state(self, path: str | Path):
        state = {
            "train_Y":     self.train_Y.numpy().tolist(),
            "train_comps": self.train_comps.numpy().tolist(),
            "round_log":   self.round_log,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        print(f"Saved solver state to {path}")

    def load_state(self, path: str | Path):
        """Resume from a saved session."""
        with open(path) as f:
            state = json.load(f)
        self.train_Y     = torch.tensor(state["train_Y"],     dtype=torch.double)
        self.train_comps = torch.tensor(state["train_comps"], dtype=torch.long)
        self.round_log   = state["round_log"]
        # Re-fit models
        self.outcome_model = self.fit_outcome_model()
        self.pref_model    = self._fit_pref_model()
        print(f"Resumed with {len(self.train_comps)} prior comparisons.")


if __name__ == "__main__":
    print("Running BOPE with synthetic library and CLI oracle...\n")

    library = SimulationLibrary(theta_dim=10, feature_dim=6)
    library.make_synthetic(n=200, seed=42)

    oracle = HumanOracle(unity_endpoint=None)

    solver = BOPESolver(library=library, oracle=oracle)
    solver.initialize(n_init_comps=3, seed=0)

    N_ROUNDS   = 2
    COMPS_PER_ROUND = 3

    for round_idx in range(N_ROUNDS):
        print(f"\n{'='*60}")
        print(f"  BOPE Round {round_idx + 1} / {N_ROUNDS}")
        print(f"{'='*60}")

        solver.preference_exploration_round(n_comps=COMPS_PER_ROUND)

        theta, idx, run_id = solver.recommend()

        print("\nTop-5 candidates:")
        print(solver.top_k_recommendations(k=5).to_string(index=False))

    solver.save_state("bope_state.json")
    oracle.save_log("comparisons_log.json")