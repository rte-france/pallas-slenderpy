"""A set of function for fatigue post-processing."""

import numpy as np
import pandas as pd

from slenderpy import cable


def _compress(s):
    """Compress signal to remove unnecessary points in cycle count process."""
    dm = np.sign(s[1:-1] - s[:-2])
    dp = np.sign(s[2:] - s[1:-1])
    ix = np.where(dm != dp)[0]
    return np.concatenate(([0], 1 + ix, [len(s) - 1]))


def _rainflow(array_ext, flm=0, l_ult=1e16, uc_mult=0.5):
    """Rainflow counting of a signal's turning points with Goodman correction.

    From https://stackoverflow.com/questions/6107702/rainflow-counting-algorithm?,
    slightly modified.

    Parameters
    ----------
    array_ext : numpy.ndarray
        Array of turning points.
    flm : float, optional
        Fixed-load mean. The default is 0.
    l_ult : float, optional
        Ultimate load. The default is 1e16.
    uc_mult : float, optional
        Partial-load scaling. The default is 0.5.

    Returns
    -------
    array_out : numpy.ndarray
        (5, n_cycle) array of rainflow values (1) load range, (2) range mean,
        (3) Goodman-adjusted range, (4) cycle count, (5) Goodman-adjusted range
        with flm.

    """
    flmargin = l_ult - np.fabs(flm)  # fixed load margin
    tot_num = array_ext.size  # total size of input array
    array_out = np.zeros((5, tot_num - 1))  # initialize output array

    pr = 0  # index of input array
    po = 0  # index of output array
    j = -1  # index of temporary array "a"
    a = np.empty(array_ext.shape)  # temporary array for algorithm

    # loop through each turning point stored in input array
    for i in range(tot_num):

        j += 1  # increment "a" counter
        a[j] = array_ext[pr]  # put turning point into temporary array
        pr += 1  # increment input array pointer

        while (j >= 2) & (np.fabs(a[j - 1] - a[j - 2]) <= np.fabs(a[j] - a[j - 1])):
            lrange = np.fabs(a[j - 1] - a[j - 2])

            # partial range
            if j == 2:
                mean = (a[0] + a[1]) / 2.0
                adj_range = lrange * flmargin / (l_ult - np.fabs(mean))
                adj_zero_mean_range = lrange * l_ult / (l_ult - np.fabs(mean))
                a[0] = a[1]
                a[1] = a[2]
                j = 1
                if lrange > 0:
                    array_out[0, po] = lrange
                    array_out[1, po] = mean
                    array_out[2, po] = adj_range
                    array_out[3, po] = uc_mult
                    array_out[4, po] = adj_zero_mean_range
                    po += 1

            # full range
            else:
                mean = (a[j - 1] + a[j - 2]) / 2.0
                adj_range = lrange * flmargin / (l_ult - np.fabs(mean))
                adj_zero_mean_range = lrange * l_ult / (l_ult - np.fabs(mean))
                a[j - 2] = a[j]
                j = j - 2
                if lrange > 0:
                    array_out[0, po] = lrange
                    array_out[1, po] = mean
                    array_out[2, po] = adj_range
                    array_out[3, po] = 1.00
                    array_out[4, po] = adj_zero_mean_range
                    po += 1

    # partial range
    for i in range(j):
        lrange = np.fabs(a[i] - a[i + 1])
        mean = (a[i] + a[i + 1]) / 2.0
        adj_range = lrange * flmargin / (l_ult - np.fabs(mean))
        adj_zero_mean_range = lrange * l_ult / (l_ult - np.fabs(mean))
        if lrange > 0:
            array_out[0, po] = lrange
            array_out[1, po] = mean
            array_out[2, po] = adj_range
            array_out[3, po] = uc_mult
            array_out[4, po] = adj_zero_mean_range
            po += 1

    # get rid of unused entries
    array_out = array_out[:, :po]

    return array_out


def count_cycles(res, cb, xb=89.0e-03, pos="left", var="un", add_cat=True):
    """Count vibration cycles in simulation outputs.

    Given a cable object and simulation results, a post-processing chain is
    applied to get a cycle history for a variable and a position. The steps in
    the chain are: projection in a fixed triad, signal interpolation at given
    value, adding equilibrium position (catenary equation), signal compression
    and cycle count.

    If the of space points in the input results are too far from xb, the
    interpolation will contain large errors.

    Parameters
    ----------
    res : slenderpy.simulation.Results
        Input results from a cable simulation.
    cb : slenderpy.cable.SCable
        The cable object used to generate the results.
    xb : float, optional
        Distance (m) from end where to estimate offsets. The default is 89 mm,
        a common value in fatigue experiments.
    var : str, optional
        Key which indicates on which variable to perform the cycle count. The
        default is 'un'.
    add_cat : bool, optional
        Add or not catenary equation for cycle range mean. The default is True.

    Raises
    ------
    RuntimeError
        DESCRIPTION.

    Returns
    -------
    dc : pandas.DataFrame
        A dataframe of length the number of cycles computed; the columns are (for
        for each cyle): the load range (m), the range mean (m) and the cycle count.

    """
    prj = cable.tnb2xyz(res, cb)
    if var == "un":
        prv = "uz"
    elif var == "ub":
        prv = "uy"

    # normalize xb
    if pos == "left":
        left = True
    elif pos == "right":
        left = False
    else:
        raise ValueError("Input pos must be left or right")

    if left:
        sb = xb / cb.Lp
    else:
        sb = 1.0 - xb / cb.Lp

    # get offset at xb
    los = np.array(prj.los())
    if left:
        tmp = np.where(np.logical_and(los > sb, los < 0.5))[0]
    else:
        tmp = np.where(np.logical_and(los < sb, los > 0.5))[0]

    # ..
    if len(tmp) == 0:
        raise RuntimeError("not enough data in res.los()")
    else:
        if left:
            tmp = tmp[0]
        else:
            tmp = tmp[-1]
    # ..
    if left:
        sp = los[tmp]
        yp = prj.data[prv][:, tmp].values
        if tmp == 0:
            sm = 0.0
            ym = 0.0
        else:
            sm = los[tmp - 1]
            ym = prj.data[prv][:, tmp - 1].values
    else:
        sm = los[tmp]
        ym = prj.data[prv][:, tmp].values
        imx = len(los) - 1
        if tmp == imx:
            sp = 0.0
            yp = 0.0
        else:
            sp = los[tmp + 1]
            yp = prj.data[prv][:, tmp + 1].values

    yb = (yp - ym) / (sp - sm) * (sb - sm) + ym

    # add catenary
    if add_cat:
        yb -= cb.altitude_1s(sb) - cb.h * sb

    # compress
    i = _compress(yb)
    yb = yb[i]

    # count
    count = _rainflow(yb)
    dc = pd.DataFrame(
        data=count[[0, 1, 3], :].T, columns=["load_range", "range_mean", "cycle_count"]
    )

    return dc
