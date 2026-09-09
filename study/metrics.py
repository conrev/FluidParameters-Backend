#!/usr/bin/env python
"""
study/metrics.py — statistics for an end-to-end human calibration session.

Computed offline by replaying a recorded session (study/recorder.py), so nothing here costs the
participant any waiting time and every number can be recomputed later, with a different definition,
without re-running the study.

The five families below are chosen to answer the questions the paper itself raises for a human
study ("the consistency of comparative judgements, task completion time, and user fatigue over
repeated comparisons", §5.3) and to make the case study quantitative rather than anecdotal.

  1. DESCRIPTIVES   who, how big a candidate space, what budget, how much of it was completed.

  2. EFFORT & FATIGUE
     Decision time per comparison (median/IQR), total time on task, warm-up vs BO phase, and the
     trend of decision time over the session (Spearman rho + OLS slope in seconds per comparison,
     plus a first-half/second-half split). A positive trend is the fatigue signal; a negative one
     is habituation. Client-reported decision times are preferred when present because the
     server-side clock also contains transport and rendering.

  3. JUDGEMENT CONSISTENCY  — how noisy are this participant's comparisons?
     * Prequential accuracy: for each comparison, the model fitted ONLY on the preceding
       comparisons predicts a winner; we score it against what the participant actually chose.
       One-step-ahead, so there is no peeking. Chance is 0.5. This is the human analogue of
       sweeping sigma_oracle in the synthetic evaluation.
       NOTE the phase split matters: warm-up pairs come from a Sobol design, independent of the
       model, whereas EUBO deliberately proposes pairs the model finds hard. Lower accuracy in the
       BO phase is the acquisition working, not the participant being inconsistent.
     * Implied noise-to-signal ratio: the comparison noise sigma that best explains the observed
       choices under the same probit link as Eq. 15, divided by the spread of the latent utility —
       directly comparable in spirit to the sigma_oracle fraction used in the synthetic study.
     * Circular triads: intransitive cycles (A>B, B>C, C>A) in the comparison graph, reported with
       the number of triads for which all three pairs were actually compared (the denominator).
     * Repeat-pair agreement: test-retest agreement when the same unordered pair appears twice.
     * Incumbent retention: in the BO phase the challenger duels the incumbent; the fraction of
       duels the incumbent survives (overall and in the final third) is a convergence signal.
     * Side bias: fraction of "A" choices with an exact binomial p-value. In the BO phase the
       incumbent is option B unless sides are randomised, so a side bias would be confounded with
       incumbent retention — this quantifies that risk.

  4. CONVERGENCE / OPTIMISATION BEHAVIOUR
     A real session has no ground truth, so regret is unavailable. What IS measurable is STABILITY:
     re-fit the GP after each comparison and track the recommendation it would have returned.
     Reported as the number of times the recommendation changed, the comparison after which it
     stopped changing ("settling point"), and the distance from each intermediate recommendation to
     the final one. Also posterior contraction (mean posterior sd over the grid) and the final
     per-parameter credible width as a fraction of its range — which parameters the session
     actually pinned down and which stayed effectively unconstrained.

  5. SYSTEM PERFORMANCE, IN SITU
     GP update / candidate selection / final result timings measured during the real session,
     rather than from a synthetic microbenchmark.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch

from optim.PBO import build_candidate_tensor, fit_preference_model

CHUNK = 2048  # posterior evaluations per chunk; the full-grid posterior is super-linear in N


# ── loading ─────────────────────────────────────────────────────────────────────────────────────


@dataclass
class Session:
    """One recorded session: the header plus the merged per-comparison table."""

    dir: Path
    header: dict
    events: list[dict]
    rows: list[dict] = field(default_factory=list)        # scored comparisons (drive the model)
    catch_rows: list[dict] = field(default_factory=list)  # repeat presentations (measurement only)

    @property
    def result(self) -> Optional[dict]:
        for ev in reversed(self.events):
            if ev.get("event") == "result":
                return ev.get("result")
        return None

    @property
    def result_compute_ms(self) -> Optional[float]:
        for ev in reversed(self.events):
            if ev.get("event") == "result":
                return ev.get("computeMs")
        return None

    @property
    def end(self) -> dict:
        for ev in reversed(self.events):
            if ev.get("event") == "session_end":
                return ev
        return {}

    @property
    def param_space(self) -> dict[str, list]:
        return self.header.get("space", {}).get("params", {})


def load_session(path: Path | str) -> Session:
    """Read a session directory and join each duel with the preference it produced."""
    path = Path(path)
    events = []
    with open(path / "events.jsonl", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # a torn final line after a crash: keep everything before it

    header = next((e for e in events if e.get("event") == "session_start"), {})
    duels = {e["duelId"]: e for e in events if e.get("event") == "duel"}

    rows: list[dict] = []
    for ev in events:
        if ev.get("event") != "preference":
            continue
        d = duels.get(ev["duelId"], {})
        pred = d.get("prediction") or {}
        client = ev.get("client") or {}
        client_ms = client.get("decisionMs") or client.get("decision_ms")
        rows.append(
            {
                "index": ev.get("index"),
                "phase": d.get("phase"),
                "duelId": ev["duelId"],
                "aIndex": d.get("aIndex"),
                "bIndex": d.get("bIndex"),
                "aParams": d.get("aParams"),
                "bParams": d.get("bParams"),
                "referenceIndex": ev.get("referenceIndex", d.get("referenceIndex")),
                "swapped": d.get("swapped"),
                "choice": ev.get("choice"),
                "winnerIndex": ev.get("winnerIndex"),
                "loserIndex": ev.get("loserIndex"),
                "winnerIsReference": ev.get("winnerIsReference"),
                "serverDecisionMs": ev.get("serverDecisionMs"),
                "clientDecisionMs": client_ms,
                "decisionMs": client_ms if client_ms is not None else ev.get("serverDecisionMs"),
                "muA": pred.get("muA"),
                "muB": pred.get("muB"),
                "varA": pred.get("varA"),
                "varB": pred.get("varB"),
                "covAB": pred.get("covAB"),
                "acqValue": d.get("acqValue"),
                "fitMs": (d.get("timings") or {}).get("fitMs"),
                "selectMs": (d.get("timings") or {}).get("selectMs"),
                "client": client,
            }
        )
    # Catch trials are repeat presentations used only to measure test-retest agreement; they never
    # entered the model, so they must not enter the comparison statistics either.
    scored = [r for r in rows if r["phase"] != "catch"]
    catches = [r for r in rows if r["phase"] == "catch"]
    scored.sort(key=lambda r: (r["index"] is None, r["index"]))
    return Session(dir=path, header=header, events=events, rows=scored, catch_rows=catches)


def find_sessions(root: Path | str) -> list[Path]:
    """Every session directory under `root` (a directory containing events.jsonl)."""
    root = Path(root)
    if (root / "events.jsonl").exists():
        return [root]
    return sorted(p.parent for p in root.glob("*/events.jsonl"))


# ── small statistics helpers (no scipy dependency) ──────────────────────────────────────────────


def _clean(xs: Iterable) -> np.ndarray:
    return np.array([x for x in xs if x is not None], dtype=float)


def _describe(xs: Iterable, scale: float = 1.0) -> dict:
    a = _clean(xs) * scale
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 3),
        "sd": round(float(a.std(ddof=1)), 3) if a.size > 1 else 0.0,
        "median": round(float(np.median(a)), 3),
        "q1": round(float(np.percentile(a, 25)), 3),
        "q3": round(float(np.percentile(a, 75)), 3),
        "min": round(float(a.min()), 3),
        "max": round(float(a.max()), 3),
        "total": round(float(a.sum()), 3),
    }


def _spearman(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Rank correlation; ties averaged. None when undefined."""
    if x.size < 3:
        return None

    def rank(v):
        order = v.argsort()
        r = np.empty(v.size, dtype=float)
        r[order] = np.arange(v.size, dtype=float)
        # average ranks within tied groups
        _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros(counts.size)
        np.add.at(sums, inv, r)
        return (sums / counts)[inv]

    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _norm_cdf(z: np.ndarray | float) -> np.ndarray:
    return 0.5 * (1.0 + torch.erf(torch.as_tensor(z, dtype=torch.double) / math.sqrt(2.0))).numpy()


def _binom_two_sided(k: int, n: int, p: float = 0.5) -> Optional[float]:
    """Exact two-sided binomial p-value (small n, so the direct sum is fine)."""
    if n == 0:
        return None
    from math import comb

    probs = [comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(n + 1)]
    obs = probs[k]
    return float(min(1.0, sum(q for q in probs if q <= obs * (1 + 1e-9))))


def posterior_mean_chunked(model, X: torch.Tensor, chunk: int = CHUNK) -> torch.Tensor:
    """Exact posterior mean over a large grid, evaluated in chunks (the joint call is superlinear)."""
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), chunk):
            outs.append(model.posterior(X[i : i + chunk]).mean.reshape(-1))
    return torch.cat(outs)


def posterior_std_chunked(model, X: torch.Tensor, chunk: int = CHUNK) -> torch.Tensor:
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), chunk):
            outs.append(model.posterior(X[i : i + chunk]).variance.reshape(-1).clamp_min(0).sqrt())
    return torch.cat(outs)


# ── 1. descriptives ─────────────────────────────────────────────────────────────────────────────


def descriptives(s: Session) -> dict:
    space = s.header.get("space", {})
    cfg = s.header.get("config", {})
    budget = cfg.get("totalDuels")
    done = len(s.rows)
    return {
        "sessionId": s.header.get("sessionId"),
        "participant": s.header.get("participant"),
        "condition": s.header.get("condition"),
        "startedAt": s.header.get("startedAt"),
        "gitCommit": (s.header.get("env") or {}).get("gitCommit"),
        "nCandidates": space.get("nCandidates"),
        "dims": space.get("dims"),
        "paramSizes": space.get("sizes"),
        "method": cfg.get("method"),
        "warmup": cfg.get("warmup"),
        "randomiseSides": cfg.get("randomiseSides"),
        "catchEvery": cfg.get("catchEvery"),
        "nInit": cfg.get("nInit"),
        "budget": budget,
        "comparisonsCompleted": done,
        "completionRate": round(done / budget, 3) if budget else None,
        "endReason": s.end.get("reason"),
        "wallClockMin": round(s.end.get("durationS", 0) / 60.0, 2) if s.end else None,
    }


# ── 2. effort & fatigue ─────────────────────────────────────────────────────────────────────────


def effort(s: Session) -> dict:
    """Task time per comparison, split by phase, plus the fatigue/habituation trend."""
    rows = [r for r in s.rows if r["decisionMs"] is not None]
    times = np.array([r["decisionMs"] for r in rows], dtype=float) / 1e3  # seconds
    idx = np.array([r["index"] for r in rows], dtype=float)

    out: dict[str, Any] = {
        "source": "client" if any(r["clientDecisionMs"] is not None for r in s.rows) else "server",
        "decisionTimeS": _describe(times),
        "byPhase": {
            phase: _describe([r["decisionMs"] for r in rows if r["phase"] == phase], 1e-3)
            for phase in ("warmup", "bo")
        },
    }
    if times.size >= 3:
        slope = float(np.polyfit(idx, times, 1)[0])
        half = times.size // 2
        out["trend"] = {
            "spearmanRho": round(_spearman(idx, times) or float("nan"), 3),
            "olsSlopeSecPerComparison": round(slope, 4),
            "firstHalfMedianS": round(float(np.median(times[:half])), 2),
            "secondHalfMedianS": round(float(np.median(times[half:])), 2),
            "interpretation": (
                "slower over time (fatigue)" if slope > 0 else "faster over time (habituation)"
            ),
        }
    # Any client-reported interaction counters are summarised generically, so the Unity client can
    # add new counters (viewpoint switches, playbacks, A/B toggles...) without touching this code.
    counters: dict[str, list] = {}
    for r in s.rows:
        for k, v in (r.get("client") or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and "decision" not in k.lower():
                counters.setdefault(k, []).append(v)
    if counters:
        out["interaction"] = {k: _describe(v) for k, v in counters.items()}
    return out


# ── 3. judgement consistency ────────────────────────────────────────────────────────────────────


def _delta(row: dict) -> Optional[tuple[float, float, int]]:
    """(mean utility difference A-B, variance of that difference, +1 if A chosen else -1)."""
    if row["muA"] is None or row["choice"] not in ("A", "B"):
        return None
    dmu = row["muA"] - row["muB"]
    dvar = max(0.0, (row["varA"] or 0.0) + (row["varB"] or 0.0) - 2.0 * (row["covAB"] or 0.0))
    return dmu, dvar, (1 if row["choice"] == "A" else -1)


def _implied_sigma(
    deltas: list[tuple[float, float, int]], include_posterior_var: bool = False
) -> Optional[tuple[float, bool]]:
    """
    MLE of the comparison-noise scale sigma under the paper's own link (Eq. 15),

        P(A > B) = Phi( (g_A - g_B) / (sqrt(2) sigma) ),

    with the model's one-step-ahead posterior means standing in for g. Posterior variance is
    deliberately EXCLUDED by default: including it lets the model's own uncertainty absorb every
    disagreement, which drives sigma to zero and destroys comparability with the synthetic study.
    The estimate therefore absorbs both genuine judgement noise and model misfit — it is an upper
    bound on the participant's inconsistency, and is labelled as such.
    """
    if len(deltas) < 5:
        return None
    dmu = np.array([d[0] for d in deltas])
    dvar = np.array([d[1] for d in deltas]) if include_posterior_var else np.zeros(len(deltas))
    sign = np.array([d[2] for d in deltas])
    scale = float(np.abs(dmu).mean()) or 1.0

    def nll(sigma: float) -> float:
        z = sign * dmu / np.sqrt(2.0 * sigma**2 + dvar + 1e-12)
        return float(-np.log(np.clip(_norm_cdf(z), 1e-12, 1.0)).sum())

    grid = np.logspace(-3, 2, 200) * scale
    k = int(np.argmin([nll(g) for g in grid]))
    # k == 0 means the likelihood is still rising as sigma -> 0: the model's ordering explains every
    # observed choice, so the data give only an upper bound on the noise. Flag it rather than
    # reporting a spuriously precise "sigma = 0".
    return float(grid[k]), bool(k == 0 or k == len(grid) - 1)


def _circular_triads(edges: list[tuple[int, int]]) -> dict:
    """Intransitive 3-cycles in the directed 'beats' graph, and how many triads were testable."""
    succ: dict[int, set] = {}
    undirected: dict[int, set] = {}
    for w, l in edges:
        succ.setdefault(w, set()).add(l)
        undirected.setdefault(w, set()).add(l)
        undirected.setdefault(l, set()).add(w)

    cycles = 0
    for i, outs in succ.items():
        for j in outs:
            for k in succ.get(j, ()):
                if i in succ.get(k, ()):
                    cycles += 1
    cycles //= 3  # each 3-cycle is found once from each of its three vertices

    nodes = sorted(undirected)
    testable = 0
    for a_i in range(len(nodes)):
        for b_i in range(a_i + 1, len(nodes)):
            a, b = nodes[a_i], nodes[b_i]
            if b not in undirected[a]:
                continue
            for c in nodes[b_i + 1 :]:
                if c in undirected[a] and c in undirected[b]:
                    testable += 1
    return {
        "circularTriads": cycles,
        "triadsFullyCompared": testable,
        "intransitivityRate": round(cycles / testable, 3) if testable else None,
    }


def consistency(s: Session) -> dict:
    """How reliable were this participant's comparisons? (see the module docstring)."""
    rows = s.rows
    out: dict[str, Any] = {}

    # -- prequential (one-step-ahead) predictive accuracy -----------------------------------------
    scored = [(r, _delta(r)) for r in rows]
    scored = [(r, d) for r, d in scored if d is not None and abs(d[0]) > 0]
    if scored:
        hits = [1.0 if (d[0] > 0) == (d[2] > 0) else 0.0 for _, d in scored]
        out["prequential"] = {
            "n": len(hits),
            "accuracy": round(float(np.mean(hits)), 3),
            "byPhase": {
                phase: round(
                    float(np.mean([h for (r, _), h in zip(scored, hits) if r["phase"] == phase])), 3
                )
                for phase in ("warmup", "bo")
                if any(r["phase"] == phase for r, _ in scored)
            },
            "note": (
                "Model fitted only on preceding comparisons. Chance = 0.5. BO-phase pairs are "
                "chosen by EUBO to be informative (i.e. hard), so a lower BO accuracy reflects the "
                "acquisition function, not participant inconsistency."
            ),
        }
        deltas = [d for _, d in scored]
        fitted = _implied_sigma(deltas)
        if fitted is not None:
            sigma, at_boundary = fitted
            mus = _clean([r["muA"] for r in rows] + [r["muB"] for r in rows])
            spread = float(mus.std(ddof=1)) if mus.size > 1 else None
            z = np.array([d[2] * d[0] for d in deltas]) / math.sqrt(2 * sigma**2 + 1e-12)
            out["impliedNoise"] = {
                "sigmaLatent": round(sigma, 6),
                "utilitySpread": round(spread, 4) if spread else None,
                "noiseToSignal": round(sigma / spread, 4) if spread else None,
                "meanLogLoss": round(float(-np.log(np.clip(_norm_cdf(z), 1e-12, 1)).mean()), 4),
                "atSearchBoundary": at_boundary,
                "note": (
                    "sigma is on the model's latent utility scale (identified only up to an affine "
                    "transform), so only the noise-to-signal ratio is comparable across sessions "
                    "or with the sigma_oracle fraction used in the synthetic evaluation. "
                    "It absorbs model misfit as well as judgement noise, so read it as an UPPER "
                    "bound on the participant's inconsistency. atSearchBoundary=true means every "
                    "choice agreed with the model's ordering, so the data bound the noise from "
                    "above but do not identify it — report '<value', not a point estimate."
                ),
            }

    # -- intransitivity, repeats, incumbent retention, side bias ----------------------------------
    edges = [(r["winnerIndex"], r["loserIndex"]) for r in rows if r["winnerIndex"] is not None]
    out["transitivity"] = _circular_triads(edges)

    # Test-retest agreement: every time the same unordered pair was judged more than once, whether
    # by an injected catch trial (sides swapped) or by chance, did the participant answer the same?
    seen_pairs: dict[frozenset, list] = {}
    for r in rows + s.catch_rows:
        if r["aIndex"] is None:
            continue
        seen_pairs.setdefault(frozenset((r["aIndex"], r["bIndex"])), []).append(r["winnerIndex"])
    repeats = {k: v for k, v in seen_pairs.items() if len(v) > 1}
    agree = sum(1 for v in repeats.values() if len(set(v)) == 1)
    out["repeatedPairs"] = {
        "nRepeatedPairs": len(repeats),
        "nCatchTrials": len(s.catch_rows),
        "agreementRate": round(agree / len(repeats), 3) if repeats else None,
        "note": (
            "Test-retest agreement on pairs judged twice (catch trials are re-presented with the "
            "sides swapped, so an agreement well below 1.0 indicates genuinely noisy judgements "
            "and a rate near 0.5 indicates the pair was indistinguishable to the participant). "
            "0 repeated pairs means no catch trials were configured (catch_every=0)."
        ),
    }

    bo = [r for r in rows if r["winnerIsReference"] is not None]
    if bo:
        held = [1.0 if r["winnerIsReference"] else 0.0 for r in bo]
        third = max(1, len(held) // 3)
        out["incumbentRetention"] = {
            "n": len(held),
            "overall": round(float(np.mean(held)), 3),
            "finalThird": round(float(np.mean(held[-third:])), 3),
            "note": (
                "Fraction of BO duels the incumbent survived. Rising towards the end of a session "
                "indicates the recommendation has stabilised."
            ),
        }

    choices = [r["choice"] for r in rows if r["choice"] in ("A", "B")]
    if choices:
        n_a = sum(1 for c in choices if c == "A")
        out["sideBias"] = {
            "n": len(choices),
            "propA": round(n_a / len(choices), 3),
            "binomialP": _binom_two_sided(n_a, len(choices)),
            "sidesRandomised": (s.header.get("config") or {}).get("randomiseSides"),
            "note": (
                "Without randomised sides the BO-phase incumbent is always option B, so a side "
                "bias is confounded with incumbent retention."
            ),
        }
    return out


# ── 4. convergence / optimisation behaviour ─────────────────────────────────────────────────────


def replay_trajectory(s: Session, stride: int = 1, chunk: int = CHUNK) -> list[dict]:
    """
    Re-fit the GP after each comparison and record the recommendation it would have returned.

    This is the session's *stability* curve: with a real participant there is no ground truth, so
    we measure how early and how firmly the recommendation settles rather than regret against a
    known optimum. `stride` subsamples the (expensive) full-grid posterior for long sessions.
    """
    param_space = s.param_space
    if not param_space or not s.rows:
        return []
    all_X, configs, _, _ = build_candidate_tensor(param_space)

    seen: list[int] = []
    comps: list[tuple[int, int]] = []

    def g2l(g: int) -> int:
        if g not in seen:
            seen.append(g)
        return seen.index(g)

    traj: list[dict] = []
    n = len(s.rows)
    for m, row in enumerate(s.rows, start=1):
        if row["winnerIndex"] is None:
            continue
        comps.append((g2l(row["winnerIndex"]), g2l(row["loserIndex"])))
        if len(comps) < 2:
            continue
        if stride > 1 and (m % stride) and m != n:
            continue
        model = fit_preference_model(all_X[seen], torch.tensor(comps, dtype=torch.long))
        mu = posterior_mean_chunked(model, all_X, chunk)
        sd = posterior_std_chunked(model, all_X, chunk)
        best = int(mu.argmax())
        # The preference-GP utility is identified only up to an affine transform, so its latent
        # scale drifts as comparisons accumulate and a raw posterior sd is NOT comparable across
        # fits. Normalising by the spread of the mean over the grid gives a scale-free measure of
        # how uncertain the surface is relative to how much structure it claims.
        spread = float(mu.std()) if mu.numel() > 1 else 0.0
        mean_sd = float(sd.mean())
        traj.append(
            {
                "comparison": m,
                "phase": row["phase"],
                "bestIndex": best,
                "bestParams": configs[best],
                "meanPosteriorSd": round(mean_sd, 4),
                "utilitySpread": round(spread, 4),
                "relativeUncertainty": round(mean_sd / spread, 4) if spread > 1e-12 else None,
                "distinctSeen": len(seen),
            }
        )

    if traj:  # distance from each intermediate recommendation to the final one (normalised space)
        final = all_X[traj[-1]["bestIndex"]]
        for t in traj:
            d = (all_X[t["bestIndex"]] - final).norm() / math.sqrt(all_X.shape[1])
            t["distanceToFinal"] = round(float(d), 4)
    return traj


def stability(traj: list[dict], total_comparisons: int) -> dict:
    """Turn the recommendation trajectory into: did it settle, and when?"""
    if not traj:
        return {"n": 0}
    idxs = [t["bestIndex"] for t in traj]
    changes = [traj[i]["comparison"] for i in range(1, len(idxs)) if idxs[i] != idxs[i - 1]]
    settled = changes[-1] if changes else traj[0]["comparison"]
    return {
        "nTrackedSteps": len(traj),
        "recommendationChanges": len(changes),
        "settledAtComparison": settled,
        "stableFraction": (
            round((total_comparisons - settled) / total_comparisons, 3)
            if total_comparisons
            else None
        ),
        "finalRelativeUncertainty": traj[-1].get("relativeUncertainty"),
        "initialRelativeUncertainty": traj[0].get("relativeUncertainty"),
        "posteriorContraction": (
            round(1 - traj[-1]["relativeUncertainty"] / traj[0]["relativeUncertainty"], 3)
            if traj[0].get("relativeUncertainty") and traj[-1].get("relativeUncertainty")
            else None
        ),
        "note": (
            "Stability, not regret: with a human participant there is no ground-truth optimum. "
            "'settledAtComparison' is the last comparison at which the recommendation changed. "
            "Uncertainty is reported RELATIVE to the spread of the fitted utility, because the "
            "preference-GP latent scale is identified only up to an affine transform and so a raw "
            "posterior sd is not comparable between fits."
        ),
    }


def identifiability(s: Session) -> dict:
    """
    Which parameters did the session actually constrain?

    For each parameter, the width of the top scenario's credible range as a fraction of the full
    calibrated range. Near 1 means the session left that parameter essentially unconstrained;
    small values mean the participant's judgements pinned it down.
    """
    result = s.result
    space = s.param_space
    if not result or not space:
        return {}
    scenarios = result.get("scenarios") or []
    if not scenarios:
        return {"note": "no scenarios in the final result"}
    params = scenarios[0].get("params", {})
    out: dict[str, Any] = {"perParameter": {}}
    for key, block in params.items():
        values = space.get(key) or []
        if not values:
            continue
        full = max(values) - min(values)
        width = block["hi"] - block["lo"]
        out["perParameter"][key] = {
            "value": block["value"],
            "lo": block["lo"],
            "hi": block["hi"],
            "relativeWidth": round(width / full, 3) if full else None,
        }
    widths = [v["relativeWidth"] for v in out["perParameter"].values() if v["relativeWidth"] is not None]
    if widths:
        ranked = sorted(out["perParameter"].items(), key=lambda kv: kv[1]["relativeWidth"] or 1)
        out["meanRelativeWidth"] = round(float(np.mean(widths)), 3)
        out["mostConstrained"] = ranked[0][0]
        out["leastConstrained"] = ranked[-1][0]
    out["convergence"] = result.get("convergence")
    out["scenarioShares"] = [sc.get("share") for sc in scenarios]
    out["optimalParameter"] = result.get("optimalParameter")
    out["best"] = result.get("best")
    return out


def coverage(s: Session) -> dict:
    """How much of the candidate space the participant actually saw, and how it concentrated."""
    space = s.param_space
    rows = s.rows
    if not rows:
        return {}
    shown = {r["aIndex"] for r in rows} | {r["bIndex"] for r in rows}
    shown.discard(None)
    n_total = s.header.get("space", {}).get("nCandidates")

    per_param: dict[str, dict] = {}
    for key, values in space.items():
        seen_values = {
            (r[side] or {}).get(key) for r in rows for side in ("aParams", "bParams") if r[side]
        }
        seen_values.discard(None)
        per_param[key] = {
            "distinctShown": len(seen_values),
            "available": len(values),
            "fraction": round(len(seen_values) / len(values), 3) if values else None,
        }

    # Mean pairwise separation of the two options in a duel: EUBO should propose closer pairs than
    # the space-filling warm-up once the model has localised the good region.
    all_X, _, _, _ = build_candidate_tensor(space) if space else (None, None, None, None)
    seps: dict[str, list] = {"warmup": [], "bo": []}
    if all_X is not None:
        d = math.sqrt(all_X.shape[1])
        for r in rows:
            if r["aIndex"] is None or r["phase"] not in seps:
                continue
            seps[r["phase"]].append(
                float((all_X[r["aIndex"]] - all_X[r["bIndex"]]).norm() / d)
            )
    return {
        "distinctCandidatesShown": len(shown),
        "fractionOfSpace": round(len(shown) / n_total, 5) if n_total else None,
        "perParameter": per_param,
        "duelSeparation": {k: _describe(v) for k, v in seps.items() if v},
    }


# ── 5. in-situ system performance ───────────────────────────────────────────────────────────────


def runtime(s: Session) -> dict:
    """Backend latency measured during the real session (upgrades the microbenchmark table)."""
    return {
        "updateGpMs": _describe([r["fitMs"] for r in s.rows]),
        "selectCandidateMs": _describe([r["selectMs"] for r in s.rows]),
        "finalResultMs": s.result_compute_ms,
        "note": (
            "Measured in situ during the session; 'select' is exhaustive EUBO over the candidate "
            "set and 'final result' is computed once, after the last comparison."
        ),
    }


# ── assembly ────────────────────────────────────────────────────────────────────────────────────


def summarise(s: Session, stride: int = 1, replay: bool = True) -> dict:
    """All five families for one session. `replay=False` skips the expensive GP re-fits."""
    out = {
        "session": descriptives(s),
        "effort": effort(s),
        "consistency": consistency(s),
        "coverage": coverage(s),
        "identifiability": identifiability(s),
        "runtime": runtime(s),
    }
    if replay:
        traj = replay_trajectory(s, stride=stride)
        out["stability"] = stability(traj, len(s.rows))
        out["_trajectory"] = traj
    return out
