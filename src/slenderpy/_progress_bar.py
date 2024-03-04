"""Helpers for using the progress meter functionalities of the tqdm library."""
import tqdm


class PBar(tqdm.tqdm):
    """Simplification of tqdm.tqdm progress bar."""

    def __init__(self, it=None, desc='', total=None, leave=True, disable=False,
                 unit_scale=1, **kwargs):
        """Init with specific treatment for some args."""
        super().__init__(it, desc=desc, total=total, leave=leave,
                         disable=disable, unit_scale=unit_scale, **kwargs)


def generate(pp, nt, desc=''):
    """Generate a progress bar."""
    if isinstance(pp, bool):
        pb = PBar(total=nt, desc=desc, disable=not pp)
    else:
        pb = PBar(total=nt, desc=desc, **pp)

    return pb
