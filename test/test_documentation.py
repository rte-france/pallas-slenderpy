"""The API reference covers every public module, and every link resolves."""

import dataclasses
import hashlib
import importlib
import json
import pkgutil
import re
from pathlib import Path

import yaml

import slenderpy

ROOT = Path(__file__).parents[1]
QUARTO = ROOT / "doc" / "_quarto.yml"
LINK = re.compile(r"\[\]\(`~?([^`]+)`\)")


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


def _links():
    """Interlink targets of the docstrings and of the documentation pages."""
    sources = [
        p for p in (ROOT / "src" / "slenderpy").rglob("*.py") if "legacy" not in p.parts
    ]
    sources += list((ROOT / "doc").glob("*.qmd"))
    sources += list((ROOT / "doc" / "examples").glob("*.qmd"))
    return {
        target: path.relative_to(ROOT).as_posix()
        for path in sources
        for target in LINK.findall(path.read_text(encoding="utf-8"))
    }


def _resolves(target, documented):
    """True if the target is a documented module or an object defined in one."""
    parts = target.split(".")
    for n in range(len(parts), 1, -1):
        module = ".".join(parts[1:n])
        if module in documented:
            break
    else:
        return False
    obj = importlib.import_module(target[: target.find(module) + len(module)])
    rest = parts[n:]
    if rest:
        defined_in = getattr(getattr(obj, rest[0], None), "__module__", None)
        if defined_in != obj.__name__:
            return False
    for i, name in enumerate(rest):
        if hasattr(obj, name):
            obj = getattr(obj, name)
        elif i == len(rest) - 1 and dataclasses.is_dataclass(obj):
            return name in {f.name for f in dataclasses.fields(obj)}
        else:
            return False
    return True


def test_every_interlink_resolves():
    documented = _documented()
    broken = {t: f for t, f in _links().items() if not _resolves(t, documented)}
    assert broken == {}


def test_every_executed_page_is_frozen_and_current():
    """CI renders from doc/_freeze: a missing or stale freeze would re-run a page."""
    doc = ROOT / "doc"
    pages = list(doc.glob("*.qmd")) + list((doc / "examples").glob("*.qmd"))
    stale = []
    for page in pages:
        text = page.read_text(encoding="utf-8")
        if "```{python}" not in text:
            continue
        name = page.relative_to(doc).with_suffix("").as_posix()
        freeze = doc / "_freeze" / name / "execute-results" / "html.json"
        # Quarto hashes the page source with LF line endings, as read_text gives
        if (
            not freeze.exists()
            or json.loads(freeze.read_text(encoding="utf-8"))["hash"]
            != hashlib.md5(text.encode("utf-8")).hexdigest()
        ):
            stale.append(name)
    assert stale == []
