"""Global random seed for the project.

Imported everywhere stochastic. Do not change this value after the first results
land; it is pinned in the manuscript and in committed result tables.
"""

GLOBAL_SEED: int = 20260426
"""Fixed seed for all random-sample draws in this project.

Set at project start on 2026-04-26 (manuscript-original draft date). Used by:
- numpy.random.default_rng for random-instance sampling per signal
- subject-clustered bootstrap in src/ppgfeas/analysis/bootstrap.py
- any deterministic shuffle in the figure pipeline
"""
