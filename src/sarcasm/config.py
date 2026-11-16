"""Configuration loading and path resolution.

Config files (``configs/*.json``) contain no machine-specific paths. String values may use placeholders:
  ${DATA}       data root          (env SARCASM_DATA,       default <repo>/data)
  ${MODELS}     pretrained models  (env SARCASM_MODELS,     default <repo>/models)
  ${OUTPUTS}    run outputs        (env SARCASM_OUTPUTS,    default <repo>/outputs)
  ${ARROW_ROOT} MMSD2.0 arrow files (env SARCASM_ARROW_ROOT, default: path written by scripts/download_assets.py
                to ${DATA}/arrow_root.txt)
A config has the form {"common": {...}, "runs": [{"name": ..., ...}, ...]}; the settings of one run are
common updated with the run entry.
"""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def roots():
    data = Path(os.environ.get("SARCASM_DATA", REPO / "data"))
    out = dict(DATA=data, MODELS=Path(os.environ.get("SARCASM_MODELS", REPO / "models")),
               OUTPUTS=Path(os.environ.get("SARCASM_OUTPUTS", REPO / "outputs")))
    arrow = os.environ.get("SARCASM_ARROW_ROOT")
    if arrow is None and (data / "arrow_root.txt").exists():
        arrow = (data / "arrow_root.txt").read_text().strip()
    out["ARROW_ROOT"] = Path(arrow) if arrow else data / "MISSING_ARROW_ROOT_run_scripts_download_assets"
    return {k: str(v) for k, v in out.items()}


def resolve(value, table=None):
    table = table or roots()
    if isinstance(value, str):
        for key, path in table.items():
            value = value.replace("${" + key + "}", path)
        return value
    if isinstance(value, list):
        return [resolve(v, table) for v in value]
    if isinstance(value, dict):
        return {k: resolve(v, table) for k, v in value.items()}
    return value


def load_config(path):
    """Load a config file and resolve its placeholders."""
    return resolve(json.loads(Path(path).read_text()))


def run_config(protocol, name):
    """Settings of one named run: common updated with the run entry."""
    run = next((r for r in protocol["runs"] if r["name"] == name), None)
    if run is None:
        raise KeyError(f"run {name!r} not in config; available: {[r['name'] for r in protocol['runs']]}")
    return dict(protocol["common"], **run)


def default_output(config_path, name):
    return Path(roots()["OUTPUTS"]) / Path(config_path).stem / name
