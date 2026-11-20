"""Blend of SRS and Elo: margin = w * SRS + (1 - w) * Elo. The total comes from SRS.

The two models are wrong for different reasons (SRS weighs the whole season
equally; Elo reacts to recent games and remembers the previous year), so
averaging them usually reduces the error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import SeasonArrays
from .elo import EloModel, EloState
from .srs import SrsModel, SrsState


@dataclass
class BlendState:
    srs: SrsState
    elo: EloState

    @property
    def games(self) -> np.ndarray:
        return self.srs.games


class BlendModel:
    name = "blend"
    label = "SRS + Elo blend"

    def __init__(self, srs: SrsModel, elo: EloModel, weight: float = 0.5) -> None:
        self.srs, self.elo, self.weight = srs, elo, weight

    def prior(
        self, prev: BlendState | None, prev_sa: SeasonArrays | None, sa: SeasonArrays
    ) -> dict:
        return {
            "srs": self.srs.prior(prev.srs if prev else None, prev_sa, sa),
            "elo": self.elo.prior(prev.elo if prev else None, prev_sa, sa),
        }

    def fit(
        self, sa: SeasonArrays, mask: np.ndarray, prior: dict, now_week: int | None = None
    ) -> BlendState:
        return BlendState(
            srs=self.srs.fit(sa, mask, prior["srs"], now_week),
            elo=self.elo.fit(sa, mask, prior["elo"], now_week),
        )

    def predict(self, state: BlendState, h, a, neutral) -> tuple[np.ndarray, np.ndarray]:
        m_srs, total = self.srs.predict(state.srs, h, a, neutral)
        m_elo, _ = self.elo.predict(state.elo, h, a, neutral)
        return self.weight * m_srs + (1.0 - self.weight) * m_elo, total

    def describe(self) -> dict:
        return {
            "model": self.name,
            "srs_weight": self.weight,
            "srs": self.srs.describe(),
            "elo": self.elo.describe(),
        }
