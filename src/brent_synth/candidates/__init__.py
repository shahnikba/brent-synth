"""The ladder: scenario generators in increasing order of mechanism.

Each rung adds one idea to the one below it, so a rung that fails to
beat its predecessor out of sample has not earned its parameters.

    0  iid_t          fat tails, nothing else
    1  garch_normal   clustering, Gaussian shock
    2  gjr_skewt      + leverage and a skewed fat-tailed shock
    3  gjr_skewt_vt   + long-run variance pinned to the sample
    4  figarch_skewt  + long memory in volatility
    5  ms_variance    a different story: regimes, Gaussian within each
"""

from __future__ import annotations

from brent_synth.candidates.arch_backed import (
    ArchCandidate,
    FigarchSkewT,
    GarchNormal,
)
from brent_synth.candidates.base import (
    DensityForecast,
    FittedModel,
    LocationScaleForecast,
    MixtureForecast,
    ScenarioModel,
)
from brent_synth.candidates.gjr_skewt import GjrSkewT, GjrSkewTVarianceTarget
from brent_synth.candidates.iid_t import IidStudentT
from brent_synth.candidates.ms_variance import MarkovSwitchingVariance

CANDIDATES: tuple[ScenarioModel, ...] = (
    IidStudentT(),
    GarchNormal(),
    GjrSkewT(),
    GjrSkewTVarianceTarget(),
    FigarchSkewT(),
    MarkovSwitchingVariance(),
)

CANDIDATES_BY_NAME = {c.name: c for c in CANDIDATES}

__all__ = [
    "CANDIDATES",
    "CANDIDATES_BY_NAME",
    "ArchCandidate",
    "DensityForecast",
    "FigarchSkewT",
    "FittedModel",
    "GarchNormal",
    "GjrSkewT",
    "GjrSkewTVarianceTarget",
    "IidStudentT",
    "LocationScaleForecast",
    "MarkovSwitchingVariance",
    "MixtureForecast",
    "ScenarioModel",
]
