"""Every public module of slenderpy appears in the API reference."""

import importlib
import pkgutil
from pathlib import Path

import yaml

import slenderpy

QUARTO = Path(__file__).parents[1] / "doc" / "_quarto.yml"


def _public_modules():
    """Leaf modules, and packages that define names of their own."""
    names = []
    for info in pkgutil.walk_packages(slenderpy.__path__, "slenderpy."):
        parts = info.name.split(".")
        if parts[1] == "legacy" or any(p.startswith("_") for p in parts):
            continue
        if info.ispkg:
            module = importlib.import_module(info.name)
            own = [
                name
                for name, value in vars(module).items()
                if not name.startswith("_")
                and getattr(value, "__name__", "").rsplit(".", 1)[0] != info.name
            ]
            if not own:
                continue
        names.append(info.name.removeprefix("slenderpy."))
    return names


def _documented():
    config = yaml.safe_load(QUARTO.read_text(encoding="utf-8"))["quartodoc"]
    entries = []
    for section in config["sections"]:
        for item in section["contents"]:
            entries.append(item if isinstance(item, str) else item["name"])
    return entries


def test_every_public_module_is_documented():
    documented = _documented()
    missing = [m for m in _public_modules() if m not in documented]
    assert missing == []
