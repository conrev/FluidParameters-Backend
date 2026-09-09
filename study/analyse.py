#!/usr/bin/env python
"""
study/analyse.py — turn recorded calibration sessions into study statistics.

Usage
-----
  python -m study.analyse sessions/                  # every session under a root, + cohort table
  python -m study.analyse sessions/20260909-..._P01  # a single session
  python -m study.analyse sessions/ --stride 2       # re-fit every 2nd comparison (faster replay)
  python -m study.analyse sessions/ --no-replay      # skip GP re-fits entirely (seconds, not minutes)
  python -m study.analyse sessions/ --latex          # also print LaTeX tables for the paper

Writes, next to each session's events.jsonl:
  analysis.json     every statistic computed for that session
  comparisons.csv   one row per comparison (the analysis-ready table)
  trajectory.csv    the recommendation/stability trajectory (when replay is enabled)

and, at the cohort root:
  cohort.csv        one row per session — the table to aggregate across participants
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from study.metrics import find_sessions, load_session, summarise

COMPARISON_COLUMNS = [
    "index", "phase", "choice", "decisionMs", "serverDecisionMs", "clientDecisionMs",
    "aIndex", "bIndex", "winnerIndex", "loserIndex", "referenceIndex", "winnerIsReference",
    "swapped", "muA", "muB", "varA", "varB", "covAB", "acqValue", "fitMs", "selectMs",
]


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def analyse_one(path: Path, stride: int, replay: bool) -> dict:
    s = load_session(path)
    summary = summarise(s, stride=stride, replay=replay)
    traj = summary.pop("_trajectory", [])

    # Per-comparison table, with the participant's chosen parameter values flattened out so the
    # CSV is directly usable in R/pandas without re-parsing nested JSON.
    rows = []
    for r in s.rows:
        row = {k: r.get(k) for k in COMPARISON_COLUMNS}
        for key, value in (r.get("aParams") or {}).items():
            row[f"A_{key}"] = value
        for key, value in (r.get("bParams") or {}).items():
            row[f"B_{key}"] = value
        for key, value in (r.get("winnerParams") or {}).items():
            row[f"win_{key}"] = value
        rows.append(row)
    columns = COMPARISON_COLUMNS + sorted({k for r in rows for k in r} - set(COMPARISON_COLUMNS))
    _write_csv(path / "comparisons.csv", rows, columns)

    if traj:
        flat = []
        for t in traj:
            row = {k: v for k, v in t.items() if k != "bestParams"}
            for key, value in (t.get("bestParams") or {}).items():
                row[f"best_{key}"] = value
            flat.append(row)
        cols = list(flat[0].keys())
        _write_csv(path / "trajectory.csv", flat, cols)

    with open(path / "analysis.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return summary


def _fmt(v, nd=2, dash="—"):
    if v is None:
        return dash
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def report(summary: dict) -> None:
    """Human-readable summary of one session."""
    d, e, c = summary["session"], summary["effort"], summary["consistency"]
    st, ident, rt = summary.get("stability", {}), summary["identifiability"], summary["runtime"]
    cov = summary["coverage"]

    print(f"\n{'=' * 78}")
    print(f"  {d['participant']}   {d['startedAt']}   ({d['sessionId']})")
    print(f"{'=' * 78}")
    print(
        f"  space {d['nCandidates']:,} candidates in {d['dims']}D | method {d['method']}"
        f" | budget {d['comparisonsCompleted']}/{d['budget']}"
        f" ({_fmt(d['completionRate'])}) | {d['endReason']} | {_fmt(d['wallClockMin'])} min"
    )

    dt = e["decisionTimeS"]
    print(f"\n  EFFORT ({e['source']} clock)")
    if dt.get("n"):
        print(
            f"    decision time  median {_fmt(dt['median'])}s "
            f"[IQR {_fmt(dt['q1'])}–{_fmt(dt['q3'])}], max {_fmt(dt['max'])}s, "
            f"total {_fmt(dt['total'] / 60)} min"
        )
        for phase, blk in e["byPhase"].items():
            if blk.get("n"):
                print(f"      {phase:<7} median {_fmt(blk['median'])}s (n={blk['n']})")
    if "trend" in e:
        t = e["trend"]
        print(
            f"    fatigue trend  rho={_fmt(t['spearmanRho'])}, "
            f"{_fmt(t['olsSlopeSecPerComparison'], 3)} s/comparison "
            f"({_fmt(t['firstHalfMedianS'])}s -> {_fmt(t['secondHalfMedianS'])}s) — "
            f"{t['interpretation']}"
        )

    print("\n  JUDGEMENT CONSISTENCY")
    if "prequential" in c:
        p = c["prequential"]
        phases = ", ".join(f"{k} {_fmt(v)}" for k, v in p["byPhase"].items())
        print(f"    prequential accuracy  {_fmt(p['accuracy'])} (n={p['n']}; {phases}) [chance 0.50]")
    if "impliedNoise" in c:
        n = c["impliedNoise"]
        bound = "< " if n.get("atSearchBoundary") else ""
        print(
            f"    implied noise-to-signal  {bound}{_fmt(n['noiseToSignal'], 3)}"
            f"   (log-loss {_fmt(n['meanLogLoss'], 3)})"
            + ("   [choices perfectly consistent — upper bound only]"
               if n.get("atSearchBoundary") else "")
        )
    tr = c["transitivity"]
    print(
        f"    circular triads  {tr['circularTriads']} of {tr['triadsFullyCompared']} testable"
        f" (rate {_fmt(tr['intransitivityRate'])})"
    )
    rp = c["repeatedPairs"]
    print(
        f"    repeated pairs   {rp['nRepeatedPairs']}"
        + (f", agreement {_fmt(rp['agreementRate'])}" if rp["nRepeatedPairs"] else " (no catch trials)")
    )
    if "incumbentRetention" in c:
        ir = c["incumbentRetention"]
        print(
            f"    incumbent held   {_fmt(ir['overall'])} overall, "
            f"{_fmt(ir['finalThird'])} in the final third"
        )
    if "sideBias" in c:
        sb = c["sideBias"]
        flag = "" if sb["sidesRandomised"] else "   [sides NOT randomised]"
        print(f"    side bias        P(A)={_fmt(sb['propA'])}, p={_fmt(sb['binomialP'], 3)}{flag}")

    if st.get("nTrackedSteps"):
        print("\n  CONVERGENCE (stability, not regret — no ground truth with a human)")
        print(
            f"    recommendation changed {st['recommendationChanges']}x, settled at comparison "
            f"{st['settledAtComparison']} (stable for the final {_fmt(st['stableFraction'])} "
            f"of the budget)"
        )
        print(f"    posterior contraction {_fmt(st['posteriorContraction'])}")

    if ident.get("perParameter"):
        print("\n  IDENTIFIABILITY (credible width as a fraction of the calibrated range)")
        for k, v in sorted(ident["perParameter"].items(), key=lambda kv: kv[1]["relativeWidth"] or 1):
            bar = "#" * int(round((v["relativeWidth"] or 0) * 30))
            print(
                f"    {k:<14} {_fmt(v['value'])}  [{_fmt(v['lo'])}, {_fmt(v['hi'])}]  "
                f"{_fmt(v['relativeWidth'])} {bar}"
            )
        conv = ident.get("convergence") or {}
        print(
            f"    -> {conv.get('distinctScenarios')} scenario(s), top share "
            f"{_fmt(conv.get('topShare'))}, status '{conv.get('status')}'"
        )

    print("\n  COVERAGE")
    print(
        f"    {cov.get('distinctCandidatesShown')} distinct candidates shown "
        f"({_fmt((cov.get('fractionOfSpace') or 0) * 100, 2)}% of the space)"
    )
    sep = cov.get("duelSeparation", {})
    if "warmup" in sep and "bo" in sep:
        print(
            f"    duel separation  warm-up {_fmt(sep['warmup']['median'], 3)} -> "
            f"BO {_fmt(sep['bo']['median'], 3)} (normalised)"
        )

    print("\n  RUNTIME (in situ)")
    for label, key in (("update GP", "updateGpMs"), ("select candidate", "selectCandidateMs")):
        b = rt[key]
        if b.get("n"):
            print(f"    {label:<17} {_fmt(b['mean'], 1)} ± {_fmt(b['sd'], 1)} ms (max {_fmt(b['max'], 1)})")
    print(f"    {'final result':<17} {_fmt(rt['finalResultMs'], 1)} ms")


COHORT_COLUMNS = [
    "participant", "sessionId", "startedAt", "condition", "endReason",
    "nCandidates", "dims", "budget", "comparisonsCompleted", "completionRate", "wallClockMin",
    "randomiseSides", "catchEvery", "nCatchTrials",
    "medianDecisionS", "totalDecisionMin", "fatigueSlopeSPerComp", "fatigueRho",
    "prequentialAccuracy", "prequentialWarmup", "prequentialBo", "noiseToSignal",
    "circularTriads", "triadsFullyCompared", "repeatAgreement",
    "incumbentHeldOverall", "incumbentHeldFinalThird", "propA", "sideBiasP",
    "recommendationChanges", "settledAtComparison", "stableFraction", "posteriorContraction",
    "distinctScenarios", "topShare", "convergenceStatus", "meanRelativeWidth",
    "mostConstrained", "leastConstrained",
    "distinctCandidatesShown", "fractionOfSpace",
    "updateGpMsMean", "selectCandidateMsMean", "finalResultMs",
]


def cohort_row(summary: dict) -> dict:
    d, e, c = summary["session"], summary["effort"], summary["consistency"]
    st, ident, rt = summary.get("stability", {}), summary["identifiability"], summary["runtime"]
    cov, conv = summary["coverage"], (summary["identifiability"].get("convergence") or {})
    dt, trend = e["decisionTimeS"], e.get("trend", {})
    pq, noise = c.get("prequential", {}), c.get("impliedNoise", {})
    tr, rp, ir, sb = (
        c["transitivity"], c["repeatedPairs"], c.get("incumbentRetention", {}), c.get("sideBias", {})
    )
    return {
        **{k: d.get(k) for k in ("participant", "sessionId", "startedAt", "condition", "endReason",
                                 "nCandidates", "dims", "budget", "comparisonsCompleted",
                                 "completionRate", "wallClockMin",
                                 "randomiseSides", "catchEvery")},
        "nCatchTrials": rp.get("nCatchTrials"),
        "medianDecisionS": dt.get("median"),
        "totalDecisionMin": round(dt["total"] / 60, 2) if dt.get("total") else None,
        "fatigueSlopeSPerComp": trend.get("olsSlopeSecPerComparison"),
        "fatigueRho": trend.get("spearmanRho"),
        "prequentialAccuracy": pq.get("accuracy"),
        "prequentialWarmup": (pq.get("byPhase") or {}).get("warmup"),
        "prequentialBo": (pq.get("byPhase") or {}).get("bo"),
        "noiseToSignal": noise.get("noiseToSignal"),
        "circularTriads": tr.get("circularTriads"),
        "triadsFullyCompared": tr.get("triadsFullyCompared"),
        "repeatAgreement": rp.get("agreementRate"),
        "incumbentHeldOverall": ir.get("overall"),
        "incumbentHeldFinalThird": ir.get("finalThird"),
        "propA": sb.get("propA"),
        "sideBiasP": sb.get("binomialP"),
        "recommendationChanges": st.get("recommendationChanges"),
        "settledAtComparison": st.get("settledAtComparison"),
        "stableFraction": st.get("stableFraction"),
        "posteriorContraction": st.get("posteriorContraction"),
        "distinctScenarios": conv.get("distinctScenarios"),
        "topShare": conv.get("topShare"),
        "convergenceStatus": conv.get("status"),
        "meanRelativeWidth": ident.get("meanRelativeWidth"),
        "mostConstrained": ident.get("mostConstrained"),
        "leastConstrained": ident.get("leastConstrained"),
        "distinctCandidatesShown": cov.get("distinctCandidatesShown"),
        "fractionOfSpace": cov.get("fractionOfSpace"),
        "updateGpMsMean": rt["updateGpMs"].get("mean"),
        "selectCandidateMsMean": rt["selectCandidateMs"].get("mean"),
        "finalResultMs": rt.get("finalResultMs"),
    }


def latex_tables(summaries: list[dict]) -> str:
    """Two paper-ready tables: session conduct/consistency, and in-situ runtime."""
    lines = [
        r"% --- session conduct and judgement consistency -------------------------------",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Calibration sessions. Decision time is the median per comparison; "
        r"prequential accuracy is the one-step-ahead agreement between the fitted preference model "
        r"and the participant's next judgement (chance $=0.5$); the recommendation is reported as "
        r"the comparison after which it no longer changed.}",
        r"\label{tab:sessions}",
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        r"Participant & Comparisons & Median time (s) & Preq.\ acc. & Settled at & Top share \\",
        r"\hline",
    ]
    for s in summaries:
        r = cohort_row(s)
        lines.append(
            f"{r['participant']} & {r['comparisonsCompleted']} & "
            f"{_fmt(r['medianDecisionS'], 1)} & {_fmt(r['prequentialAccuracy'])} & "
            f"{_fmt(r['settledAtComparison'], 0)} & {_fmt(r['topShare'])} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]

    lines += [
        r"% --- in-situ runtime (replaces the synthetic Table 3) ------------------------",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Runtime of the principal optimisation operations, measured during the "
        r"calibration sessions themselves (mean $\pm$ s.d.\ across comparisons).}",
        r"\label{tab:runtime-insitu}",
        r"\begin{tabular}{lr}",
        r"\hline",
        r"Operation & Time \\",
        r"\hline",
    ]
    gp = [s["runtime"]["updateGpMs"] for s in summaries if s["runtime"]["updateGpMs"].get("n")]
    sel = [s["runtime"]["selectCandidateMs"] for s in summaries
           if s["runtime"]["selectCandidateMs"].get("n")]
    fin = [s["runtime"]["finalResultMs"] for s in summaries if s["runtime"].get("finalResultMs")]
    if sel:
        lines.append(
            f"Select next candidate using EUBO & "
            f"{sum(b['mean'] for b in sel) / len(sel) / 1e3:.3f} s \\\\"
        )
    if gp:
        lines.append(
            f"Update GP with new comparison data & "
            f"{sum(b['mean'] for b in gp) / len(gp) / 1e3:.3f} s \\\\"
        )
    if fin:
        lines.append(f"Compute final calibration result & {sum(fin) / len(fin) / 1e3:.3f} s \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyse recorded PBO calibration sessions.")
    ap.add_argument("path", help="a session directory, or a root containing session directories")
    ap.add_argument("--stride", type=int, default=1,
                    help="re-fit the GP every Nth comparison during replay (default 1 = every one)")
    ap.add_argument("--no-replay", action="store_true",
                    help="skip GP re-fits (no stability/convergence statistics, but much faster)")
    ap.add_argument("--latex", action="store_true", help="also print LaTeX tables for the paper")
    args = ap.parse_args()

    root = Path(args.path)
    sessions = find_sessions(root)
    if not sessions:
        print(f"No sessions found under {root}", file=sys.stderr)
        raise SystemExit(1)

    summaries = []
    for path in sessions:
        try:
            summary = analyse_one(path, stride=args.stride, replay=not args.no_replay)
        except Exception as exc:  # noqa: BLE001 — one bad session must not sink the cohort
            print(f"  ! skipped {path.name}: {exc}", file=sys.stderr)
            continue
        report(summary)
        summaries.append(summary)

    if not summaries:
        raise SystemExit(1)

    out_root = root if not (root / "events.jsonl").exists() else root.parent
    cohort_path = out_root / "cohort.csv"
    _write_csv(cohort_path, [cohort_row(s) for s in summaries], COHORT_COLUMNS)
    print(f"\n{len(summaries)} session(s) analysed -> {cohort_path}")
    print("  per session: analysis.json, comparisons.csv" +
          ("" if args.no_replay else ", trajectory.csv"))

    if args.latex:
        print("\n" + latex_tables(summaries))


if __name__ == "__main__":
    main()
