"""The legacy package stays importable, module by module."""

import importlib

import pytest

LEGACY = [
    "beam",
    "cable",
    "cable_wakeosc",
    "fatigue",
    "fdm_utils",
    "force",
    "simtools",
    "turbwind",
    "wind",
    "_cable_utils",
]


@pytest.mark.parametrize("name", LEGACY)
def test_legacy_modules_import(name):
    importlib.import_module(f"slenderpy.legacy.{name}")
