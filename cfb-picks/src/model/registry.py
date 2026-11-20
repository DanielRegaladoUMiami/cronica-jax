"""Carga los hiperparámetros calibrados y construye los modelos candidatos."""

from __future__ import annotations

import json

from .. import config
from .baseline import BaselineModel
from .blend import BlendModel
from .elo import EloModel, EloParams
from .srs import SrsModel, SrsParams


def load_params() -> dict:
    if config.PARAMS_PATH.exists():
        return json.loads(config.PARAMS_PATH.read_text())
    return {}


def build_models(params: dict | None = None) -> dict:
    """Los candidatos, en orden: baseline, SRS, Elo y (si aporta) la mezcla."""
    params = load_params() if params is None else params
    srs = SrsModel(SrsParams(**params["srs"])) if "srs" in params else SrsModel()
    elo = EloModel(EloParams(**params["elo"])) if "elo" in params else EloModel()
    models = {"baseline": BaselineModel(), "srs": srs, "elo": elo}
    weight = float(params.get("peso_srs_en_mezcla", 0.5))
    # Si la mezcla óptima es prácticamente 100% SRS, no es un modelo distinto.
    if 0.05 < weight < 0.95:
        models["blend"] = BlendModel(srs, elo, weight)
    return models


def selected_name(params: dict | None = None) -> str:
    params = load_params() if params is None else params
    return params.get("seleccionado", "srs")
