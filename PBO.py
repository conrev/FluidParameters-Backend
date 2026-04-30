import torch
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.optim import optimize_acqf
from botorch.utils.sampling import manual_seed

torch.manual_seed(42)

# ── Config ─────────────────────────────────────────────────────────────────
PARAM_DIM   = 2        # e.g. [wind_speed, temperature], both in [0, 1]
BOUNDS      = torch.stack([torch.zeros(PARAM_DIM), torch.ones(PARAM_DIM)])
N_COLD_START = 3       # random pairs before fitting the GP
N_ITERATIONS = 10      # total comparison rounds

# Real parameter ranges (for display only — BoTorch works in [0,1] internally)
PARAM_NAMES = ["wind_speed (0–100)", "temperature (−20–50)"]


# ── Storage ────────────────────────────────────────────────────────────────
# datapoints:   shape (N, D)  — all unique designs seen so far
# comparisons:  shape (M, 2)  — each row is (winner_idx, loser_idx)
datapoints  = torch.empty((0, PARAM_DIM), dtype=torch.double)
comparisons = torch.empty((0, 2), dtype=torch.long)


# ── Helpers ────────────────────────────────────────────────────────────────
def add_datapoint(x: torch.Tensor) -> int:
    """Register a design vector, return its index."""
    global datapoints
    datapoints = torch.cat([datapoints, x.unsqueeze(0)])
    return len(datapoints) - 1


def add_comparison(winner_idx: int, loser_idx: int):
    global comparisons
    pair = torch.tensor([[winner_idx, loser_idx]], dtype=torch.long)
    comparisons = torch.cat([comparisons, pair])


def denormalize(x: torch.Tensor) -> list[str]:
    """Pretty-print normalized params as real-world values."""
    ranges = [(0, 100), (-20, 50)]
    return [
        f"{PARAM_NAMES[i]}: {ranges[i][0] + x[i].item() * (ranges[i][1] - ranges[i][0]):.2f}"
        for i in range(PARAM_DIM)
    ]


def random_candidate() -> torch.Tensor:
    return torch.rand(PARAM_DIM, dtype=torch.double)


# ── Model ──────────────────────────────────────────────────────────────────
def fit_model() -> PairwiseGP:
    """Fit a PairwiseGP on all comparisons collected so far."""
    model = PairwiseGP(
        datapoints,        # all observed design points
        comparisons,       # pairwise preference data
    )
    mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()
    return model


# ── Acquisition ────────────────────────────────────────────────────────────
def suggest_next_pair(model: PairwiseGP) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Use EUBO to find the most informative pair to compare next.
    EUBO = Expected Utility of the Best Option.
    It returns TWO candidates (q=2) — one pair to show the human.
    """
    eubo = AnalyticExpectedUtilityOfBestOption(pref_model=model)

    # optimize_acqf returns shape (q, D) — here (2, D)
    candidates, acq_value = optimize_acqf(
        eubo,
        bounds=BOUNDS,
        q=2,              # always 2 for a pairwise comparison
        num_restarts=8,
        raw_samples=64,
    )
    print(f"\n  [EUBO acq value: {acq_value.item():.4f}]")
    return candidates[0], candidates[1]   # design A, design B


# ── Human-in-the-loop ──────────────────────────────────────────────────────
def ask_human(a: torch.Tensor, b: torch.Tensor) -> int:
    """
    Show two designs and ask which is preferred.
    Returns 0 if A is better, 1 if B is better.
    """
    print("\n  Design A:")
    for line in denormalize(a): print(f"    {line}")
    print("  Design B:")
    for line in denormalize(b): print(f"    {line}")

    while True:
        choice = input("\n  Which design is better? [A/B]: ").strip().upper()
        if choice in ("A", "B"):
            return 0 if choice == "A" else 1
        print("  Please enter A or B.")


# ── Best design so far ─────────────────────────────────────────────────────
def show_best(model: PairwiseGP):
    """Find the design with the highest predicted utility."""
    with torch.no_grad():
        posterior = model.posterior(datapoints)
        utility   = posterior.mean.squeeze()   # shape (N,)
    best_idx  = utility.argmax().item()
    best_x    = datapoints[best_idx]
    print(f"\n  Best design so far (idx {best_idx}, utility={utility[best_idx]:.3f}):")
    for line in denormalize(best_x): print(f"    {line}")


# ══════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════
def run():
    print("=" * 60)
    print("  Preferential Bayesian Optimization — commandline demo")
    print("=" * 60)

    for iteration in range(1, N_ITERATIONS + 1):
        print(f"\n{'─'*60}")
        print(f"  Round {iteration}/{N_ITERATIONS}  |  comparisons so far: {len(comparisons)}")

        # ── 1. Generate candidate pair ─────────────────────────────────
        if len(comparisons) < N_COLD_START:
            # Cold start: random candidates, no model yet
            print("  [Cold start — generating random pair]")
            a = random_candidate()
            b = random_candidate()
        else:
            # Warm: fit GP and use EUBO to propose the best pair
            print("  [Fitting PairwiseGP + optimizing EUBO...]")
            model = fit_model()
            show_best(model)
            a, b  = suggest_next_pair(model)

        # ── 2. Register designs (dedup by adding to datapoints) ────────
        idx_a = add_datapoint(a)
        idx_b = add_datapoint(b)

        # ── 3. Human comparison ────────────────────────────────────────
        choice = ask_human(a, b)   # 0=A wins, 1=B wins
        winner_idx = idx_a if choice == 0 else idx_b
        loser_idx  = idx_b if choice == 0 else idx_a
        add_comparison(winner_idx, loser_idx)
        print(f"  → {'A' if choice == 0 else 'B'} preferred. Total comparisons: {len(comparisons)}")

    # ── Final result ───────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  Optimization complete!")
    model = fit_model()
    show_best(model)
    print(f"\n  All {len(datapoints)} designs evaluated:")
    with torch.no_grad():
        utilities = model.posterior(datapoints).mean.squeeze()
    ranked = utilities.argsort(descending=True)
    for rank, idx in enumerate(ranked[:5]):
        print(f"    #{rank+1}  utility={utilities[idx]:.3f}  {denormalize(datapoints[idx])}")


if __name__ == "__main__":
    run()