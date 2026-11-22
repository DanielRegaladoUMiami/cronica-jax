"""Load the tuned hyperparameters and build the candidate models."""

from __future__ import annotations

import json

from .. import config
from .baseline import BaselineModel
from .blend import BlendModel
from .elo import EloModel, EloParams
from .srs import SrsModel, SrsParams


def load_params(league: config.League = config.CFB) -> dict:
    if league.params_path.exists():
        return json.loads(league.params_path.read_text())
    return {}


def build_models(params: dict | None = None) -> dict:
    """The candidates, in order: baseline, SRS, Elo and (when it adds something) the blend."""
    params = load_params() if params is None else params
    srs = SrsModel(SrsParams(**params["srs"])) if "srs" in params else SrsModel()
    elo = EloModel(EloParams(**params["elo"])) if "elo" in params else EloModel()
    models = {"baseline": BaselineModel(), "srs": srs, "elo": elo}
    weight = float(params.get("srs_blend_weight", 0.5))
    # If the best blend is essentially 100% SRS, it is not a different model.
    if 0.05 < weight < 0.95:
        models["blend"] = BlendModel(srs, elo, weight)
    return models


def selected_name(params: dict | None = None) -> str:
    params = load_params() if params is None else params
    return params.get("selected", "srs")
