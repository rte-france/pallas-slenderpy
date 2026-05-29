"""Side of the stockbridge damper (left or right of the clamp)."""

from enum import Enum


class Side(str, Enum):
    """Side of a damper mass relative to the clamp.

    The :attr:`epsilon` value carries the geometric sign that appears in the
    equations of motion: ``-1`` for the left mass, ``+1`` for the right one.
    """

    LEFT = "left"
    RIGHT = "right"

    @property
    def epsilon(self) -> int:
        """Sign convention: ``-1`` on the left, ``+1`` on the right."""
        return -1 if self is Side.LEFT else 1
