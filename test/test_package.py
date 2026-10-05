"""The package root exposes what slenderpy.future exposed."""

import slenderpy


def test_package_root_exports():
    from slenderpy.components import Conductor, Span

    assert slenderpy.Conductor is Conductor
    assert slenderpy.Span is Span
    assert slenderpy.floatArrayLike is not None
