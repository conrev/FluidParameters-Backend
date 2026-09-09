# Running an end-to-end user study

Instrumentation for human calibration sessions: what is recorded, which statistics come out, and
how they map onto the case-study section of the paper.

Design rule: **the optimiser is untouched.** `optim/` never imports `study/`, a session with no
recorder attached behaves exactly as before, and nothing in the analysis runs while a participant
is waiting. Everything expensive is computed afterwards by replaying the log.

---

## 1. What gets recorded

One self-contained directory per session:

```
sessions/20260909-141233_P01_a1b2c3/
    session.json     participant, protocol, candidate space, library versions, git commit
    events.jsonl     every event, appended and flushed as it happens
    result.json      the final calibration result payload
```

`events.jsonl` holds, in order:

| event | carries |
|---|---|
| `session_start` | participant / condition / notes, budget, method, warm-up design, full parameter space, `\|Θ\|`, torch+botorch versions, git commit |
| `duel` | comparison index, phase (`warmup` / `bo` / `catch`), both candidates' parameter values, which one is the incumbent, whether sides were swapped, the model's belief about this pair, the EUBO value, and the GP-update / candidate-selection timings |
| `preference` | the choice, winner and loser, decision time, and any client-side interaction telemetry |
| `result` | the full result payload and how long it took to compute |
| `session_end` | why the session ended (`completed`, `disconnected`) and its duration |

The `duel` event's *belief* block (`muA, muB, varA, varB, covAB`) is the important one: it is the
posterior over the pair **as it stood before the participant answered**, i.e. fitted only on the
preceding comparisons. That makes every consistency statistic a genuine one-step-ahead test rather
than a fit to data the model has already seen. It costs two posterior look-ups per comparison.

Recording is on by default. `PBO_SESSION_LOG_DIR` sets the location; `PBO_DISABLE_SESSION_LOG=1`
turns it off.

---

## 2. Client-side fields (optional, recommended)

The backend works unchanged if the Unity client sends nothing extra, but two additions make the
study substantially stronger:

**On `init`** — identify the participant and set the protocol. Everything session-level goes at the
top level of `init`; `client{}` (below) is per-comparison only.
```jsonc
{ "type": "init",
  "parameters": "...",        // JSON *string*, as before
  "n_init": 10,
  "n_bo": 30,
  "randomiseSides": true,     // shuffle A/B display order      (default false)
  "catchEvery": 8,            // repeat an earlier pair every N (default 0 = off)
  "study": { "participant": "P01", "condition": "baseline", "notes": "Budj Bim, facilitator AH" } }
```
All four study fields are optional and default to the previous behaviour, so an old client keeps
working unchanged.

**On `duel`** — report what the participant actually did:
```jsonc
{ "type": "duel", "duelId": "...", "choice": "A",
  "client": { "decisionMs": 24310, "viewpointSwitches": 3, "abToggles": 5,
              "playbacks": 2, "checkpointsInspected": [12, 480, 940] } }
```
`decisionMs` should be measured from the first rendered frame to the click. The backend also times
this itself, but the server clock includes transport and scene loading, so the client value is the
better one for task-time and fatigue analysis — it is used automatically when present. Any numeric
counter you add is summarised automatically; you do not need to change the backend to add one.

---

## 3. Two protocol options worth enabling

Both default to **off**, so production behaviour is unchanged, and both are set at `init`.

### `randomiseSides: true` — remove a position confound
In the BO phase the challenger duels the incumbent, and the incumbent is **always option B**. If a
participant has any side bias, it is indistinguishable from a preference for incumbents, and it
biases the optimisation itself, not just the statistics. Randomising which candidate is displayed
as A removes this. The analysis reports the side bias either way (`propA` with an exact binomial
p-value) so you can show the confound was controlled.

### `catchEvery: N` — measure judgement consistency directly
Re-presents a previously answered pair, **sides swapped**, after every N
scored comparisons. A catch trial is pure measurement: it never enters the model, never changes
the recommendation, and never counts against the comparison budget — but it does cost the
participant a comparison's worth of time, so budget for it (`catch_every=8` over 40 comparisons
adds ~5 trials, roughly 2 extra minutes).

Test–retest agreement from catch trials is the most direct answer to the paper's own open question
about "the consistency of comparative judgements". An agreement well below 1.0 means genuinely
noisy judgements; near 0.5 means the pair was simply indistinguishable to the participant, which is
itself the "perceptible differences" precondition from §5.2 being tested empirically.

Both settings are recorded in `session.json` and carried through to `cohort.csv`
(`randomiseSides`, `catchEvery`, `nCatchTrials`), so each session's protocol is self-documenting.

---

## 4. The statistics

```bash
python -m study.analyse sessions/            # every session + cohort.csv
python -m study.analyse sessions/ --latex    # also print paper-ready LaTeX tables
python -m study.analyse sessions/ --no-replay  # skip GP re-fits (seconds instead of minutes)
python -m study.analyse sessions/ --stride 2   # re-fit every 2nd comparison
```

Writes `analysis.json`, `comparisons.csv` and `trajectory.csv` beside each session, and one
`cohort.csv` row per session for aggregation across participants.

### Effort and fatigue → §5.3 "task completion time, and user fatigue"
Median and IQR of decision time, total time on task, warm-up versus BO phase, and the trend over
the session (Spearman ρ, OLS slope in seconds per comparison, first-half versus second-half
medians). A positive slope is the fatigue signal; a negative one is habituation. Client interaction
counters are summarised alongside.

### Judgement consistency → §5.3 "consistency of comparative judgements"
- **Prequential accuracy** — how often the model, fitted only on the preceding comparisons,
  predicted the participant's next choice. Chance is 0.5. Reported per phase, and this split
  matters: warm-up pairs come from a Sobol design and are model-independent, whereas EUBO
  deliberately proposes pairs the model finds hard, so *lower BO accuracy is the acquisition
  function working*, not the participant being inconsistent. Say so in the caption.
- **Implied noise-to-signal ratio** — the comparison noise σ that best explains the observed
  choices under the same probit link as Eq. 15, divided by the spread of the fitted utility. This
  is the bridge between the case study and the synthetic evaluation: it is directly comparable to
  the σ = 0.05 *fraction* used there. If the participant's choices are perfectly consistent with
  the model's ordering the estimate runs to the search boundary and is flagged
  (`atSearchBoundary`) — report it as an upper bound, not a point estimate.
  *Measured behaviour, from a synthetic sweep (2 participants each, Budj Bim space, 10 + 30):*

  | true σ | prequential accuracy | implied noise-to-signal |
  |---|---|---|
  | 0.05 | 0.82, 0.87 | 0.22, 0.12 |
  | 0.15 | 0.82, 0.87 | 0.17, 0.12 |
  | 0.35 | 0.74, 0.74 | 0.31, 0.20 |

  The estimator separates a sloppy participant from a careful one, but it **saturates at the low
  end**: σ = 0.05 and σ = 0.15 are indistinguishable, because at those levels the probit oracle is
  effectively deterministic, and the residual ~0.1–0.2 is model misfit rather than judgement noise.
  Treat it as an upper bound with a floor around 0.1, not a calibrated measurement.

- **Test–retest agreement** on repeated pairs (see catch trials above). This is the one consistency
  measure with no model in the loop, which is why it is worth spending budget on.
- **Circular triads** — intransitive cycles (A≻B, B≻C, C≻A), reported with the number of triads for
  which all three pairs were actually compared. With 40 comparisons over 6000 candidates most
  candidates are seen once, so the denominator is often small; report it honestly rather than
  claiming perfect transitivity from an empty test.
- **Incumbent retention** — the fraction of BO duels the incumbent survived, overall and in the
  final third. Rising retention late in a session is a convergence signal.
- **Side bias** — `P(A)` with an exact binomial p-value.

### Convergence → makes §4.2 quantitative
With a real participant there is no ground truth, so **regret is unavailable and must not be
claimed**. What is measurable is *stability*: the GP is re-fitted after every comparison and the
recommendation it would have returned is tracked. Reported as the number of times the
recommendation changed, the comparison after which it stopped changing, and the fraction of the
budget for which it was already final — which answers "was 40 comparisons enough?" with evidence.

Uncertainty is reported **relative** to the spread of the fitted utility, because the preference-GP
latent scale is identified only up to an affine transform: a raw posterior standard deviation is
not comparable between two fits.

### Identifiability → a genuinely new result for the case study
For each of the five boundary-condition parameters, the width of the top scenario's credible range
as a fraction of its calibrated range. Near 1 means the session left that parameter essentially
unconstrained; small values mean the participant's judgements pinned it down. This lets §4.2 say
something concrete — *which* aspects of the hydrograph the archaeologist's knowledge actually
constrained, and which remained open — instead of only reporting that a session completed.

### Runtime in situ → replaces Table 3
GP update, candidate selection and final-result timings measured during the real sessions
(mean ± s.d. across comparisons) rather than from a synthetic microbenchmark. `--latex` emits this
as a drop-in replacement table.

---

## 5. Piloting before you run people

`study/simulate.py` drives a complete session with a synthetic participant that answers with
exactly the Eq. 15 probit oracle and spends a plausible amount of time doing it (log-normal
decision times that lengthen for hard pairs and drift upward with fatigue).

```bash
python -m study.simulate                          # Budj Bim space (6000), 10 + 30, σ = 0.05
python -m study.simulate --participants 5         # a synthetic cohort
python -m study.simulate --noise 0.20             # a noisier participant
python -m study.simulate --catch-every 8 --randomise-sides
python -m study.simulate --space small --n-init 4 --n-bo 8   # fast smoke test
```

The `budj` preset is the case study's own candidate space — 8 × 6 × 5 × 5 × 5 = 6000 candidates,
matching Table 2 exactly. Use it to check the whole pipeline, and to choose a comparison budget:
run several synthetic participants and look at `settledAtComparison` in `cohort.csv`.

Because the simulated participant has a *known* hidden optimum, a pilot cohort is also the honest
way to answer "does the recommendation land near the truth at this budget and this noise level?" —
a question no real session can answer.

---

## 6. Ethics and data handling

- `sessions/` and `cohort.csv` are in `.gitignore`. Session logs are participant data; keep them
  out of the repository and inside whatever your ethics approval specifies.
- Use de-identified participant codes (`P01`), never names or emails, in the `study` block.
- The log records parameter values, choices, timings and model state — not video, audio, or
  anything about the participant beyond the code you supply.
- Sessions that end early are recorded too, with `reason: "disconnected"`. Withdrawal means
  deleting the directory; nothing is written anywhere else.
