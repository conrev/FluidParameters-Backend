"""
User-study instrumentation for the result-driven calibration framework.

`study` is a *sidecar*: production optimisation (optim/) never imports it, and a session runs
identically whether or not a recorder is attached. It provides

    recorder.py — append-only session telemetry written during a live session (no added latency)
    metrics.py  — statistics computed offline by replaying a recorded session
    analyse.py  — CLI: one session or a whole cohort -> CSV/JSON/LaTeX for the paper
"""
