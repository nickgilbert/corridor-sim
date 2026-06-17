"""Distribution samplers over an injected RNG (spec §4, §6.4).

All randomness flows through a :class:`numpy.random.Generator` passed in by the
caller — never the global ``np.random`` state — so runs are reproducible from the
master seed.
"""

from __future__ import annotations

import numpy as np

from .config import DistSpec


def sample(spec: DistSpec, rng: np.random.Generator, n: int) -> np.ndarray:
    """Draw ``n`` samples from ``spec`` using ``rng``.

    Supported families: ``fixed``, ``triangular``, ``normal`` (optionally
    truncated via ``lo``/``hi``), ``lognormal``.
    """

    a = spec.args
    fam = spec.family

    if fam == "fixed":
        return np.full(n, float(a["value"]), dtype=float)

    if fam == "triangular":
        lo, mode, hi = float(a["lo"]), float(a["mode"]), float(a["hi"])
        if not (lo <= mode <= hi):
            raise ValueError(f"triangular requires lo<=mode<=hi, got {lo},{mode},{hi}")
        if lo == hi:
            return np.full(n, lo, dtype=float)
        return rng.triangular(lo, mode, hi, size=n)

    if fam == "normal":
        mean, sd = float(a["mean"]), float(a["sd"])
        out = rng.normal(mean, sd, size=n)
        lo = a.get("lo")
        hi = a.get("hi")
        if lo is not None or hi is not None:
            out = np.clip(out, lo if lo is not None else -np.inf, hi if hi is not None else np.inf)
        return out

    if fam == "lognormal":
        mean, sd = float(a["mean"]), float(a["sd"])
        return rng.lognormal(mean, sd, size=n)

    raise ValueError(f"Unknown distribution family: {fam!r}")


def point(spec: DistSpec) -> float:
    """Return a representative point value for a distribution (its mode/mean)."""

    a = spec.args
    fam = spec.family
    if fam == "fixed":
        return float(a["value"])
    if fam == "triangular":
        return float(a["mode"])
    if fam == "normal":
        return float(a["mean"])
    if fam == "lognormal":
        # mode of a lognormal in linear space
        return float(np.exp(a["mean"] - a["sd"] ** 2))
    raise ValueError(f"Unknown distribution family: {fam!r}")
