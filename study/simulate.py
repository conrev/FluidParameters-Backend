#!/usr/bin/env python
"""
study/simulate.py — drive a complete session with a *simulated* participant.

Two uses:

  1. End-to-end test of the study pipeline (recorder -> events.jsonl -> study.analyse) without
     needing a person or the Unity client.
  2. Piloting the protocol before running real participants: how many comparisons before the
     recommendation settles at this candidate-space size, how much a given catch-trial rate costs,
     what a session looks like at sigma = 0.05 versus a sloppier participant.

The simulated participant answers with exactly the oracle of Eq. 15 of the paper,

    Pr(A > B) = Phi( (g(A) - g(B)) / (sqrt(2) * sigma) ),   sigma = noise * sd(g over Theta)

and spends a plausible amount of time doing it: log-normal decision times that grow with a fatigue
drift and stretch when the two candidates are close in latent goodness (hard comparisons take
longer). None of this reaches the optimiser — it only fills the log the way a person would.

Usage
-----
  python -m study.simulate                                   # Budj Bim space (6000), 10+30, sigma 0.05
  python -m study.simulate --space small --n-init 4 --n-bo 8 # fast smoke test
  python -m study.simulate --noise 0.20 --participant P07    # a noisier participant
  python -m study.simulate --catch-every 8                   # add test-retest catch trials
  python -m study.simulate --participants 5                  # a synthetic cohort
"""

from __future__ import annotations

import argparse
import math
import random

import torch

from optim.PBO import PARAM_SPACE, PreferentialBOSession, build_candidate_tensor
from study.recorder import SessionRecorder

#: The Budj Bim candidate space of the case study (Table 2): 8 x 6 x 5 x 5 x 5 = 6000 candidates.
BUDJ_BIM_SPACE: dict[str, list] = {
    "baseStage": [round(-0.5 + 0.5 * i, 2) for i in range(8)],      # [-0.5, 3.0] m, step 0.5
    "amplitude": [round(0.5 + 0.5 * i, 2) for i in range(6)],       # [ 0.5, 3.0] m, step 0.5
    "phase": [round(0.1 + 0.2 * i, 2) for i in range(5)],           # [ 0.1, 0.9], step 0.2
    "riseFraction": [round(0.1 + 0.2 * i, 2) for i in range(5)],    # [ 0.1, 0.9], step 0.2
    "peakDuration": [round(0.1 + 0.2 * i, 2) for i in range(5)],    # [ 0.1, 0.9], step 0.2
}

SMALL_SPACE: dict[str, list] = {
    "baseStage": [0.0, 0.5, 1.0, 1.5, 2.0],
    "amplitude": [0.5, 1.0, 1.5, 2.0],
    "phase": [0.1, 0.5, 0.9],
}

SPACES = {"budj": BUDJ_BIM_SPACE, "small": SMALL_SPACE, "prod": PARAM_SPACE}


class SimulatedParticipant:
    """
    A synthetic knowledge holder with a fixed (hidden) idea of how the system behaved.

    The latent goodness is a smooth single-peaked function of the normalised parameters, with
    per-parameter sensitivity: some parameters matter a lot to this participant and some barely at
    all — which is what makes the identifiability read-out in study.metrics meaningful.
    """

    def __init__(self, all_X: torch.Tensor, noise: float = 0.05, seed: int = 0) -> None:
        rng = torch.Generator().manual_seed(seed)
        dim = all_X.shape[1]
        self.target = torch.rand(dim, generator=rng, dtype=torch.double)
        # Sensitivity per parameter: 1.0 = strongly constrains the judgement, ~0 = ignored.
        self.weights = 0.15 + 0.85 * torch.rand(dim, generator=rng, dtype=torch.double)
        self.g = torch.exp(-(((all_X - self.target) ** 2) * self.weights).sum(-1) / 0.25)
        self.sigma = float(noise * self.g.std())
        self.rng = random.Random(seed)

    def prefers_a(self, i: int, j: int) -> bool:
        """Probit choice rule — Eq. 15 of the paper."""
        d = float(self.g[i] - self.g[j])
        p = 0.5 * (1.0 + math.erf(d / (math.sqrt(2.0) * math.sqrt(2.0) * max(self.sigma, 1e-12))))
        return self.rng.random() < p

    def decision_ms(self, i: int, j: int, step: int) -> float:
        """Log-normal decision time: harder pairs take longer, and the session slowly tires."""
        gap = abs(float(self.g[i] - self.g[j])) / max(float(self.g.std()), 1e-9)
        base = 14.0 + 16.0 * math.exp(-2.0 * gap)          # 14 s easy ... 30 s indistinguishable
        fatigue = 1.0 + 0.012 * step                        # ~+1.2% per comparison
        return max(1500.0, self.rng.lognormvariate(math.log(base * fatigue), 0.35) * 1000.0)


def run_one(
    space: dict[str, list],
    participant: str,
    n_init: int,
    n_bo: int,
    noise: float,
    seed: int,
    catch_every: int,
    randomise_sides: bool,
    root: str | None,
) -> str:
    all_X, _, _, _ = build_candidate_tensor(space)
    sim = SimulatedParticipant(all_X, noise=noise, seed=seed)

    recorder = SessionRecorder(
        participant=participant,
        condition="simulated",
        notes=f"synthetic participant (probit oracle, noise={noise})",
        extra={"simulated": True, "noise": noise, "seed": seed},
        root=root,
    )
    session = PreferentialBOSession(
        space,
        n_init=n_init,
        n_iterations=n_bo,
        warmup="sobol",
        seed=seed,
        recorder=recorder,
        randomise_sides=randomise_sides,
        catch_every=catch_every,
    )

    lookup = {tuple(cfg.items()): k for k, cfg in enumerate(session.configs)}
    msg = session.start()
    step = 0
    while msg.get("type") == "duel":
        i = lookup[tuple(msg["optionA"].items())]
        j = lookup[tuple(msg["optionB"].items())]
        step += 1
        choice = "A" if sim.prefers_a(i, j) else "B"
        client = {
            "decisionMs": round(sim.decision_ms(i, j, step), 1),
            "viewpointSwitches": sim.rng.randint(0, 4),
            "playbacks": sim.rng.randint(0, 3),
            "abToggles": sim.rng.randint(1, 6),
        }
        msg = session.submit_preference(msg["duelId"], choice, client)
        print(
            f"    {step:>3} {msg.get('type'):<6}"
            f" {'' if msg.get('type') != 'duel' else msg.get('phase', '')}",
            end="\r",
            flush=True,
        )
    recorder.close("completed")

    best = msg.get("optimalParameter")
    truth = {
        k: space[k][int(torch.argmin((torch.tensor(space[k], dtype=torch.double)
                                      - (min(space[k]) + sim.target[d] * (max(space[k]) - min(space[k]))))
                                     .abs()))]
        for d, k in enumerate(space)
    }
    print(f"\n  {participant}: recommended {best}")
    print(f"  {' ' * len(participant)}  hidden truth {truth}")
    print(f"  -> {recorder.dir}")
    return str(recorder.dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a simulated user-study session end to end.")
    ap.add_argument("--space", choices=sorted(SPACES), default="budj")
    ap.add_argument("--n-init", type=int, default=10, help="warm-up comparisons (paper: 10)")
    ap.add_argument("--n-bo", type=int, default=30, help="EUBO comparisons (paper: 40 total)")
    ap.add_argument("--noise", type=float, default=0.05,
                    help="oracle noise as a fraction of the utility spread (paper: 0.05)")
    ap.add_argument("--catch-every", type=int, default=0,
                    help="insert a repeat (test-retest) trial every N scored comparisons")
    ap.add_argument("--randomise-sides", action="store_true",
                    help="randomise A/B presentation order (recommended for human studies)")
    ap.add_argument("--participants", type=int, default=1, help="number of synthetic participants")
    ap.add_argument("--participant", default=None, help="participant code (single run)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default=None, help="session root (default $PBO_SESSION_LOG_DIR)")
    args = ap.parse_args()

    space = SPACES[args.space]
    n = 1
    for values in space.values():
        n *= len(values)
    print(f"space '{args.space}': {n:,} candidates in {len(space)}D | "
          f"budget {args.n_init}+{args.n_bo} | noise {args.noise}")

    for p in range(args.participants):
        name = args.participant or f"S{p + 1:02d}"
        run_one(
            space=space,
            participant=name,
            n_init=args.n_init,
            n_bo=args.n_bo,
            noise=args.noise,
            seed=args.seed + p,
            catch_every=args.catch_every,
            randomise_sides=args.randomise_sides,
            root=args.root,
        )


if __name__ == "__main__":
    main()
