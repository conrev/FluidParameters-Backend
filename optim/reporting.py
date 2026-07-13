"""
Posterior reporting & uncertainty quantification for the preference GP (guide §L1).

The deliverable is not a single winner but the *set* of reconstructions experts found
comparably convincing: the genuinely-distinct competing scenarios, each with a **share**
("how often this setting is the best fit across plausible utility surfaces"), a representative
parameter setting, and a per-parameter credible range.

Design notes (see PBO_IMPLEMENTATION_GUIDE.md and the plan; both validated empirically):

* `draw_matheron_paths` is NOT implemented for `PairwiseGP`, so whole-surface samples come from
  `model.posterior(grid).rsample(...)` (the guide's L1.b/L1.d fallback branch).
* The full duel grid (~14k points) cannot be jointly sampled (it needs a ~14k x 14k Cholesky), so
  reporting runs on a *separate coarse grid* (`build_report_grid`). The fine grid stays for duels.
* `prob_near_best >= threshold` can be an EMPTY region, so it is NOT the primary output. Shares are
  read off the **argmax distribution** partitioned into clusters -> they always sum to 1.
* Single-linkage clustering chains distinct basins into one blob, so clustering is **mode-seeking**
  (DBSCAN-core): link only frequently-winning "core" cells, then assign every sample to the nearest
  core cluster.

All internal math is in normalised [0, 1]^D space; conversion to raw physical units happens once,
at the end, in `build_result`.
"""

from __future__ import annotations

from typing import Optional

import torch
from botorch.models.pairwise_gp import PairwiseGP

# ── Tunables (documented; safe defaults validated on synthetic two-basin surfaces) ──────────────
N_PER_DIM = 41          # reporting-grid resolution per dimension
MAX_POINTS = 4096       # cap on total grid points (keeps the joint Cholesky feasible in higher D)
N_SAMPLES = 512         # posterior draws of the whole utility surface (Monte-Carlo resolution)
SEED = 0                # sampler seed -> reproducible shares
TOL_FRAC = 0.10         # relative near-best tolerance (§3.5: never a raw utility cutoff)
CORE_FRAC = 0.15        # a grid cell is a clustering "core" if it wins >= CORE_FRAC * max wins
MERGE_RADIUS = 0.12     # single-linkage radius over cores, in normalised space
SHARE_FLOOR = 0.05      # drop clusters below this share (anti-overclaim); mass -> droppedShare
QUANTILES = (0.10, 0.90)  # per-parameter credible-range quantiles of the argmax cloud

# Labeling discipline (§3.5–§3.7) — attached to every emitted result.
NOTES = [
    "Scenarios are reconstructions experts found comparably convincing — not physically "
    "determined ranges.",
    "Ranges are relative credible bands from the preference posterior; treat them as a floor "
    "(the Laplace approximation is overconfident).",
    f"Shares are Monte-Carlo estimates over {N_SAMPLES} posterior samples of the utility surface.",
]


def _subsample(values: torch.Tensor, n: int) -> torch.Tensor:
    """Return <= n evenly-spaced entries of a sorted 1-D tensor, always including both endpoints."""
    m = values.numel()
    if m <= n:
        return values
    idx = torch.linspace(0, m - 1, n).round().long().unique()
    return values[idx]


def build_report_grid(
    axis_values: list[torch.Tensor],
    n_per_dim: int = N_PER_DIM,
    max_points: int = MAX_POINTS,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Build the Cartesian reporting grid by SUBSAMPLING the real discrete parameter values.

    Crucially, every grid point is an actual candidate in the (discrete) parameter space — so the
    argmax modes and quantile endpoints we later report are guaranteed to be real reconstructions,
    not interpolated off-grid points. `axis_values` are the per-dimension allowed values in the
    model's normalised [0, 1] space (a subset of the true grid keeps the joint Cholesky tractable).

    `n_per_dim` is auto-reduced so the total point count stays <= `max_points` in higher dimensions.

    Returns
    -------
    grid   : (prod n_i, d) tensor of real candidate points in normalised space
    axes   : list of d 1-D tensors (the subsampled per-dimension values actually used)
    """
    d = len(axis_values)
    while n_per_dim > 2 and n_per_dim**d > max_points:
        n_per_dim -= 1
    axes = [_subsample(v, n_per_dim) for v in axis_values]
    if d == 1:
        grid = axes[0].unsqueeze(-1)
    else:
        grid = torch.cartesian_prod(*axes)
    return grid, axes


def sample_surface(
    model: PairwiseGP,
    grid: torch.Tensor,
    n_samples: int = N_SAMPLES,
    seed: int = SEED,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Draw `n_samples` joint samples of the latent utility over `grid`, plus the marginal band.

    Uses `posterior(grid).rsample` because pathwise (Matheron) sampling is unsupported for
    `PairwiseGP`. The global RNG is seeded (and its state saved/restored) so reported shares are
    reproducible without disturbing the rest of the program's randomness.

    Returns
    -------
    F     : (n_samples, |grid|) sampled utility surfaces
    mean  : (|grid|,) posterior mean utility (L1.a)
    std   : (|grid|,) posterior std (relative interpretation only, §3.5/§3.6)
    """
    with torch.no_grad():
        posterior = model.posterior(grid)
        mean = posterior.mean.squeeze(-1)
        std = posterior.variance.clamp_min(0.0).sqrt().squeeze(-1)
        rng_state = torch.random.get_rng_state()
        try:
            torch.manual_seed(seed)
            F = posterior.rsample(torch.Size([n_samples])).squeeze(-1)
        finally:
            torch.random.set_rng_state(rng_state)
    return F, mean, std


def _single_linkage(points: torch.Tensor, radius: float) -> list[int]:
    """Union-find single-linkage labels: points within `radius` share a component id."""
    n = len(points)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    if n > 1:
        dmat = torch.cdist(points, points)
        for i, j in (dmat <= radius).nonzero().tolist():
            if i < j:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb
    return [find(i) for i in range(n)]


def cluster_scenarios(
    F: torch.Tensor,
    grid: torch.Tensor,
    core_frac: float = CORE_FRAC,
    merge_radius: float = MERGE_RADIUS,
    share_floor: float = SHARE_FLOOR,
    tol_frac: float = TOL_FRAC,
    quantiles: tuple[float, float] = QUANTILES,
) -> tuple[list[dict], float]:
    """
    Partition the per-sample argmax distribution into distinct "similarly good" scenarios.

    Mode-seeking (DBSCAN-core) clustering — validated to recover competing basins where plain
    single-linkage chains them into one blob:

      1. argmax per sample -> the L1.b argmax cloud.
      2. "core" cells = grid cells that win the argmax frequently (>= core_frac * max wins);
         this discards the scattered tail responsible for chaining.
      3. single-linkage union-find over cores only -> count-weighted centroids (one per basin).
      4. consolidate centroids within merge_radius, so sparse cores that failed to bridge don't
         surface as near-duplicate scenarios.
      5. assign EVERY sample to its nearest centroid -> shares that sum to 1.
      6. drop sub-`share_floor` clusters (mass accumulated into `dropped_share`).

    Each surviving scenario's representative is the MODE of its argmax cloud, so the reported value
    sits inside its credible range. Each scenario is a dict in normalised space:
      {center (d,), lo (d,), hi (d,), share float, prob_near_best float}

    Returns
    -------
    scenarios     : list[dict] sorted by descending share
    dropped_share : float — mass in clusters below the floor
    """
    n_samples = F.shape[0]
    argmax_idx = F.argmax(dim=-1)                      # (n_samples,)
    pts = grid[argmax_idx]                             # (n_samples, d)

    # Per-sample near-best band (relative tolerance, §3.5) — reused for prob_near_best (L1.d).
    per_best = F.max(dim=-1, keepdim=True).values      # (n_samples, 1)
    per_span = per_best - F.min(dim=-1, keepdim=True).values
    near_thresh = (per_best - tol_frac * per_span).squeeze(-1)   # (n_samples,)

    uniq, counts = torch.unique(argmax_idx, return_counts=True)
    if uniq.numel() == 0:
        return [], 0.0

    # ── cores: frequently-winning cells only ────────────────────────────────────────────────
    core_min = max(2, int(core_frac * int(counts.max())))
    core_mask = counts >= core_min
    if not bool(core_mask.any()):
        core_mask = counts == counts.max()             # degenerate: keep the single top cell
    core_cells = grid[uniq[core_mask]]                 # (C, d)
    core_counts = counts[core_mask].double()

    # ── single-linkage union-find over cores (concentrated -> no chaining across empty space) ─
    labels = _single_linkage(core_cells, merge_radius)
    grouped = []
    for r in sorted(set(labels)):
        members = [i for i in range(len(core_cells)) if labels[i] == r]
        w = core_counts[members]
        grouped.append(((core_cells[members] * w.unsqueeze(-1)).sum(0) / w.sum(), float(w.sum())))

    # Consolidate centroids within merge_radius (fixes over-segmented sparse cores that failed to
    # bridge in the single-linkage pass and would otherwise surface as near-duplicate scenarios).
    cpts = torch.stack([c for c, _ in grouped])
    cw = torch.tensor([w for _, w in grouped], dtype=torch.double)
    clab = _single_linkage(cpts, merge_radius)
    merged = []
    for r in sorted(set(clab)):
        members = [i for i in range(len(cpts)) if clab[i] == r]
        w = cw[members]
        merged.append((cpts[members] * w.unsqueeze(-1)).sum(0) / w.sum())
    centroids = torch.stack(merged)                    # (K, d)

    # ── assign every sample to its nearest centroid -> shares sum to 1 ───────────────────────
    assign = torch.cdist(pts, centroids).argmin(dim=-1)   # (n_samples,)

    scenarios: list[dict] = []
    dropped_share = 0.0
    lo_q, hi_q = quantiles
    for k in range(len(centroids)):
        m = assign == k
        n_k = int(m.sum())
        share = n_k / n_samples
        if share < share_floor:
            dropped_share += share
            continue
        cluster_pts = pts[m]                            # (n_k, d)
        cluster_argmax = argmax_idx[m]                  # grid indices, (n_k,)
        # Representative = the MODE of this cluster's argmax cloud (the most-often-best cell);
        # it lives in the dense region, so the reported value sits inside its credible range.
        u_cells, u_counts = torch.unique(cluster_argmax, return_counts=True)
        rep_grid_idx = u_cells[u_counts.argmax()]
        center = grid[rep_grid_idx]                     # (d,)
        # `interpolation="nearest"` keeps the endpoints on real grid points (no off-grid interp).
        lo = cluster_pts.quantile(lo_q, dim=0, interpolation="nearest")
        hi = cluster_pts.quantile(hi_q, dim=0, interpolation="nearest")
        # prob_near_best (L1.d): across samples, how often is THIS setting within tol of the best.
        prob_near_best = float((F[:, rep_grid_idx] >= near_thresh).double().mean())
        scenarios.append(
            {
                "center": center,
                "lo": lo,
                "hi": hi,
                "share": share,
                "prob_near_best": prob_near_best,
            }
        )

    scenarios.sort(key=lambda s: -s["share"])
    return scenarios, dropped_share


def _to_raw(x_norm: torch.Tensor, x_min: torch.Tensor, x_range: torch.Tensor) -> torch.Tensor:
    """Inverse of the min-max normalisation in build_candidate_tensor: raw = norm * range + min."""
    return x_norm * x_range + x_min


def _round(v: float, ndigits: int = 3) -> float:
    return round(float(v), ndigits)


def _convergence(scenarios: list[dict], dropped_share: float) -> dict:
    """
    Convergence readout (L1.b): is the single best pinned down, or is the posterior still open?

    status maps to the mock's action buttons and distinguishes genuine ties from mere ignorance:
      - "converged": one basin dominates (topShare >= 0.8) -> the single best is robust.
      - "refining":  the posterior is still diffuse (many basins, or lots of sub-floor mass) ->
                     more feedback should sharpen it (epistemic uncertainty, §3.3). "Keep refining".
      - "competing": a few genuine basins with comparable shares -> the mock's headline case, truly
                     distinct reconstructions the experts rated comparably. "Compare side by side".
    """
    k = len(scenarios)
    if k == 0:
        return {
            "distinctScenarios": 0, "topShare": 0.0, "entropy": 0.0, "status": "insufficient_data",
        }
    shares = torch.tensor([s["share"] for s in scenarios], dtype=torch.double)
    top_share = float(shares.max())
    if k == 1:
        entropy = 0.0
    else:
        p = shares / shares.sum()
        entropy = float(-(p * p.clamp_min(1e-12).log()).sum() / torch.log(torch.tensor(float(k))))
    if top_share >= 0.8:
        status = "converged"
    elif k >= 4 or dropped_share >= 0.25:
        status = "refining"
    else:
        status = "competing"
    return {
        "distinctScenarios": k,
        "topShare": _round(top_share, 3),
        "entropy": _round(entropy, 3),
        "status": status,
    }


def _minimal_result(
    best_config: Optional[dict],
    total_comparisons: int,
    status: str = "insufficient_data",
) -> dict:
    """Fallback payload used when the model is unavailable or reporting fails — never raises."""
    return {
        "type": "result",
        "optimalParameter": best_config,
        "totalComparison": total_comparisons,
        "best": {"params": best_config, "scenarioId": None},
        "scenarios": [],
        "convergence": {
            "distinctScenarios": 0,
            "topShare": 0.0,
            "entropy": 0.0,
            "status": status,
        },
        "meta": {
            "posteriorSamples": 0,
            "reportGrid": {},
            "toleranceFraction": TOL_FRAC,
            "shareFloor": SHARE_FLOOR,
            "droppedShare": 0.0,
            "laplaceOverconfident": True,
        },
        "notes": NOTES,
    }


def build_result(
    model: Optional[PairwiseGP],
    param_space: dict[str, list],
    total_comparisons: int,
    best_config: Optional[dict] = None,
    n_per_dim: int = N_PER_DIM,
    max_points: int = MAX_POINTS,
    n_samples: int = N_SAMPLES,
    seed: int = SEED,
) -> dict:
    """
    Assemble the WebSocket `result` payload: single best + distinct "similarly good" scenarios
    (share, representative params, per-parameter credible range) + convergence readout + labeling.

    Takes the discrete `param_space` (the same dict used to build candidates) so that every
    reported number — `value`, `lo`, `hi`, and the single best — is snapped to an ACTUAL allowed
    value in that discrete space. Nothing off-grid is ever emitted.

    Param-agnostic: iterates `param_space`, so adding a dimension (e.g. Manning's n) flows through
    with no change here.

    On any failure (or too few comparisons) returns `_minimal_result` rather than raising, so the
    server's WS loop is never broken.
    """
    if model is None:
        return _minimal_result(best_config, total_comparisons)

    param_keys = list(param_space.keys())
    # Sorted unique allowed values per dimension (raw units), and the SAME min-max normalisation
    # build_candidate_tensor uses (constant dims guarded to range 1) so the grid matches the model.
    allowed = [torch.tensor(sorted(set(v)), dtype=torch.double) for v in param_space.values()]
    x_min = torch.stack([a.min() for a in allowed])
    x_range = torch.stack([a.max() - a.min() for a in allowed])
    x_range = torch.where(x_range == 0, torch.ones_like(x_range), x_range)
    allowed_norm = [(a - x_min[i]) / x_range[i] for i, a in enumerate(allowed)]

    def snap(raw_value: float, i: int) -> float:
        """Nearest actual allowed value in dimension i — guarantees an on-grid reconstruction."""
        return float(allowed[i][(allowed[i] - raw_value).abs().argmin()])

    try:
        grid, axes = build_report_grid(allowed_norm, n_per_dim=n_per_dim, max_points=max_points)
        F, _, _ = sample_surface(model, grid, n_samples=n_samples, seed=seed)
        scenarios_norm, dropped_share = cluster_scenarios(F, grid)
    except Exception as exc:  # noqa: BLE001 — reporting must never break the session
        print(f"Warning > reporting failed, returning minimal result: {exc}")
        return _minimal_result(best_config, total_comparisons)

    def params_block(center, lo, hi) -> dict:
        center_raw = _to_raw(center, x_min, x_range)
        lo_raw = _to_raw(lo, x_min, x_range)
        hi_raw = _to_raw(hi, x_min, x_range)
        block = {}
        for i, key in enumerate(param_keys):
            # Snap each endpoint to a real allowed value, then order and clamp the representative
            # into the (snapped) range so the UI never shows a value outside its own range.
            lo_i, hi_i = sorted((snap(float(lo_raw[i]), i), snap(float(hi_raw[i]), i)))
            value = min(max(snap(float(center_raw[i]), i), lo_i), hi_i)
            block[key] = {"value": _round(value), "lo": _round(lo_i), "hi": _round(hi_i)}
        return block

    scenarios = []
    for idx, s in enumerate(scenarios_norm):
        scenarios.append(
            {
                "id": chr(ord("A") + idx),
                "share": _round(s["share"], 3),
                "probNearBest": _round(s["prob_near_best"], 3),
                "params": params_block(s["center"], s["lo"], s["hi"]),
            }
        )

    # `best` is the reporting-based recommendation: the representative of the top-share scenario
    # (mode of the dominant basin). Flat {param: value}; ranges live in `scenarios`.
    if scenarios:
        best_params_flat = {k: v["value"] for k, v in scenarios[0]["params"].items()}
        best = {"params": best_params_flat, "scenarioId": scenarios[0]["id"]}
    else:
        best = {"params": best_config, "scenarioId": None}

    report_grid = {param_keys[i]: int(axes[i].numel()) for i in range(len(param_keys))}
    return {
        # Textbook L0 single best: argmax of the posterior MEAN over the full discrete grid
        # (exact, computed by the caller). Distinct from `best` above — see the module/schema docs.
        "optimalParameter": best_config,
        "type": "result",
        "totalComparison": total_comparisons,   # legacy field, kept
        "best": best,
        "scenarios": scenarios,
        "convergence": _convergence(scenarios_norm, dropped_share),
        "meta": {
            "posteriorSamples": n_samples,
            "reportGrid": report_grid,
            "toleranceFraction": TOL_FRAC,
            "shareFloor": SHARE_FLOOR,
            "droppedShare": _round(dropped_share, 3),
            "laplaceOverconfident": True,
        },
        "notes": NOTES,
    }
