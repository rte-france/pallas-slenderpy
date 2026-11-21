"""Wake-induced aerodynamic loads for bundled conductors.

Implements the wind-load model of sections 4.1 to 4.13 of the Eurobios/RTE note
``ESCB-2020-11-12`` (September 2021), in the simplified quasi-steady variant
described in section 4.10 where the wind force is treated explicitly in time
(no Aw1/Aw2/Bw linearisation). This matches the way :meth:`Bundle.solve`
already handles user-supplied force callbacks: forces are evaluated at the
current state at both t and t+dt and averaged.

Aerodynamic coefficients for the leeward cable come from the Wu et al. (2018)
fit reproduced in Annex A of the note (equations A.1 and A.2 with the
coefficients listed page 35). The fit is valid for y/d in [4, 20] and
|z/d| <= 5; outside that envelope the leeward cable is treated as an isolated
cylinder. Closer arrangements (|y/d| < 4) are not supported here; the Wu fit
does not cover them and the Snegovskiy polynomials would be required.

The module exposes:

- :func:`wu2018_coefs`: pure aerodynamic coefficients cd(y/d, z/d),
  cl(y/d, z/d) for the leeward cable.
- :func:`aero_force_bundle`: factory returning a force callback compatible
  with :meth:`slenderpy.cable.dynamic.bundle.Bundle.solve`.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple, Union

import numpy as np

import slenderpy.force.wind

# Wu 2018 fit coefficients (Annex A of the note, p. 35).
_WU_CD_A = 1.2
_WU_CD_a = 0.27
_WU_CD_b = 0.0272
_WU_CD_c = -8.6745
_WU_CL_A = -0.75
_WU_CL_a = 0.4
_WU_CL_b = 0.6
_WU_CL_c = 0.1

# Validity envelope of the Wu fit (note section 4.12.1 and Annex A).
_WU_Y_MIN = 4.0
_WU_Y_MAX = 20.0
_WU_Z_MAX = 5.0


def wu2018_coefs(
    y_over_d: Union[float, np.ndarray],
    z_over_d: Union[float, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Drag and lift coefficients of a leeward cylinder behind a windward one.

    Parameters
    ----------
    y_over_d : float or numpy.ndarray
        Streamwise distance from the windward cable, divided by the cable
        diameter. Must be positive (cable is downstream).
    z_over_d : float or numpy.ndarray
        Cross-stream distance from the windward cable axis, divided by the
        cable diameter.

    Returns
    -------
    cd, cl : numpy.ndarray
        Drag and lift coefficients of the leeward cable. Both arrays have
        the broadcasted shape of the inputs. The reference orientation is
        the freestream wind frame (drag along +y, lift along +z).

    Notes
    -----
    Implements equations A.1-A.2 of the note (Wu et al. 2018 fit)::

        cd = A  * (1 - exp(-a  * z^2 - b  * (y - c)))
        cl = A' * z * exp(-a' * (z - b') * (z + b') - c' * y)

    with A = 1.2, a = 0.27, b = 0.0272, c = -8.6745 for drag and
    A' = -0.75, a' = 0.4, b' = 0.6, c' = 0.1 for lift. The fit is valid for
    y/d in [4, 20] and |z/d| <= 5; values are still returned outside that
    envelope but their physical relevance is limited.
    """
    y = np.asarray(y_over_d, dtype=float)
    z = np.asarray(z_over_d, dtype=float)
    cd = _WU_CD_A * (1.0 - np.exp(-_WU_CD_a * z * z - _WU_CD_b * (y - _WU_CD_c)))
    cl = _WU_CL_A * z * np.exp(
        -_WU_CL_a * (z - _WU_CL_b) * (z + _WU_CL_b) - _WU_CL_c * y
    )
    return cd, cl




# Parameters from Annex A (page 150) of [Snegovskiy2010] to compute
# leeward drag coefficient from cable position relative to windward cable.
_b01 = 2.323294982163092e-1
_b02 = 2.844681316572165e-2
_b03 = -3.831362667418982e-4

_b21 = 7.264297381642306e-1
_b22 = -7.129965014332403e-2
_b23 = 2.002378704380350e-3

_b41 = -1.628227576498350e-1
_b42 = 1.950084905957192e-2
_b43 = -5.952414931982878e-4

_b61 = 1.536080273957189e-2
_b62 = -1.930079534172798e-3
_b63 = 5.979023476687854e-5

_b81 = -6.428424055644690e-4
_b82 = 7.929895962013830e-5
_b83 = -2.403369109975760e-6

_b101 = 9.761759132226317e-6
_b102 = -1.116095087816932e-6
_b103 = 3.173009934540457e-8

# Parameters from Annex A (page 151) of [Snegovskiy2010] to compute
# leeward lift coefficient from cable position relative to windward cable.
_q11 = -9.366116523539152e-1
_q12 = 9.453571110216909e-2
_q13 = -2.571093910023341e-3

_q31 = 3.622073725080496e-1
_q32 = -4.049373525837849e-2
_q33 = 1.147815561818735e-3

_q51 = -4.904551885931337e-2
_q52 = 5.781434477530589e-3
_q53 = -1.684462803797870e-4

_q71 = 2.815451239687794e-3
_q72 = -3.411751082960935e-4
_q73 = 1.009254549253525e-5

_q91 = -5.853126807000320e-5
_q92 = 7.187070581552582e-6
_q93 = -2.140673043868416e-7




def price_coefs(y_over_d: np.ndarray, z_over_d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Drag and lift coefficients based on Price (1975)"""
    y, z = y_over_d, z_over_d
    
    # Drag
    a0 = _b01 + _b02 * y + _b03 * (y**2)
    a2 = _b21 + _b22 * y + _b23 * (y**2)
    a4 = _b41 + _b42 * y + _b43 * (y**2)
    a6 = _b61 + _b62 * y + _b63 * (y**2)
    a8 = _b81 + _b82 * y + _b83 * (y**2)
    a10 = _b101 + _b102 * y + _b103 * (y**2)
    cd = a10 * (z**10) + a8 * (z**8) + a6 * (z**6) + a4 * (z**4) + a2 * (z**2) + a0
    cd = np.where(np.abs(z) > 4, 1.2, cd)

    # Lift
    a1 = _q11 + _q12 * y + _q13 * (y**2)
    a3 = _q31 + _q32 * y + _q33 * (y**2)
    a5 = _q51 + _q52 * y + _q53 * (y**2)
    a7 = _q71 + _q72 * y + _q73 * (y**2)
    a9 = _q91 + _q92 * y + _q93 * (y**2)
    cl = a1 * z + a3 * (z**3) + a5 * (z**5) + a7 * (z**7) + a9 * (z**9)
    cl = np.where(np.abs(z) > 4, 0.0, cl)

    return cd, cl




def diana_coefs(y_over_d: np.ndarray, z_over_d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Drag and lift coefficients based on Diana (1972)"""
    xi, eta = y_over_d, np.abs(z_over_d)
    
    # Drag
    case1_d = (1.2 - 1.605e-2 * (1.215 - 7.2e-2 * xi + 1.44e-3 * xi**2) * 
               (74 - 0.712 * eta**2 - 10.5 * eta**3 + 2.66 * eta**4))
    case2_d = (1.2 - 0.5056e-2 * (74 - 0.712 * eta**2 - 10.5 * eta**3 + 2.66 * eta**4))
    cd = np.where((eta < 3) & (np.abs(xi) < 25), case1_d, 
                  np.where((eta < 3) & (np.abs(xi) >= 25), case2_d, 1.2))

    # Lift
    case1_l = ((-1.78 + 0.127 * xi - 2.38e-3 * (xi**2)) * 
               (0.928 * (eta**2) - 0.827 * (eta**3) + 0.244 * (eta**4) - 0.0239 * (eta**5)) * np.sign(z_over_d))
    case2_l = (-0.08 * (0.928 * (eta**2) - 0.827 * (eta**3) + 0.244 * (eta**4) - 0.0239 * (eta**5)) * np.sign(z_over_d))
    cl = np.where((eta < 3.66) & (xi < 26.6), case1_l,
                  np.where((eta < 3.66) & (xi >= 26.6), case2_l, 0.0))

    return cd, cl



def aero_force_bundle(
    bundle_geometry: Sequence[Tuple[float, float]],
    diameter: float,
    wind: object,  # Objet vent issu de wind.py (ConstantWind, etc.)
    aero_model: str = "Price",  # "Wu", "Price", ou "Diana" (cf bundleconductors)
    air_density: float = 1.225,
    cd_windward: float = 1.0,
    cl_windward: float = 0.0,
) -> Callable:
    """Build a force callback that applies wake-coupled aerodynamic load.

    Mise à jour : intègration des objets de wind.py et les modèles de bundleconductors

    The returned callback has the signature expected by
    :meth:`slenderpy.cable.dynamic.bundle.Bundle.solve`. At each evaluation
    it determines, per sub-conductor and per spatial node, whether the
    sub-conductor sits in the wake of another one according to the
    instantaneous relative position in the wind frame (equations 159-160
    of the note). If so, the drag/lift coefficients come from
    :func:`wu2018_coefs`; otherwise the user-supplied isolated-cylinder
    coefficients are used. The aerodynamic force per unit length is then
    obtained from equation 147 of the note in the cable's normal-binormal
    frame.

    Parameters
    ----------
    bundle_geometry : sequence of (float, float)
        For each sub-conductor, the rest position offset (binormal, normal)
        in metres relative to a common origin. Matches the convention used
        by ``examples/bundle.py:visualize_bundle_3d``: the first coordinate
        is the binormal offset (cross-span horizontal) and the second is
        the normal offset (vertical for a horizontal span). Length must
        match the bundle's ``n_cables``.
    diameter : float
        Cable diameter (m). Common to all sub-conductors (matches the
        Bundle restriction that all sub-conductors share the same
        geometry).
    wind_speed : float or callable
        Freestream wind speed (m/s). If a callable, signature is
        ``wind_speed(s, t) -> ndarray`` where ``s`` is the normalised
        curvilinear grid in [0, 1] and ``t`` is dimensional time (s);
        the returned array must have the shape of ``s``.
    wind_direction : float, optional
        Wind direction in the (binormal, normal) plane, in radians.
        ``0.0`` (default) means the wind is along the +binormal axis
        (horizontal cross-span). ``pi/2`` means along +normal (vertical).
    air_density : float, optional
        Air mass density (kg/m^3). Default 1.225 (sea level, 15 C).
    cd_windward : float, optional
        Drag coefficient applied to sub-conductors that are not in the wake
        of any other sub-conductor. Default 1.0 (typical for ACSR
        conductors at relevant Reynolds numbers).
    cl_windward : float, optional
        Lift coefficient applied to sub-conductors that are not in the wake
        of any other sub-conductor. Default 0.0.

    Returns
    -------
    force : callable
        ``force(s, t, un_list, ub_list, vn_list, vb_list)``
        returning ``(fn_list, fb_list)`` where each list has ``n_cables``
        entries of length-``len(s)`` arrays containing the aerodynamic
        force per unit length (N/m) in the normal and binormal directions.

    Notes
    -----
    Wake detection: a sub-conductor i is considered in the wake of j at a
    given node if the streamwise distance (projected onto the wind axis)
    satisfies ``4 <= y_rel/d <= 20`` and the cross-stream distance
    ``|z_rel|/d <= 5``. If several upstream sub-conductors qualify, the
    closest one (smallest streamwise distance) is selected. The Wu fit is
    only valid in this envelope; outside, the isolated-cylinder
    coefficients are used.

    The model is an approximation in two ways:

    1. Coefficients ``cd``, ``cl`` come from a single upstream cable.
       Cascading wakes (e.g. third cable in the doubly-shielded zone of
       the first two) are approximated by the closest upstream cable only.
    2. The relative-wind direction used to project the lift is the
       cable-relative wind ``w - v_cable`` (equation 147 of the note),
       while the coefficients are evaluated using the freestream wind
       direction to project the inter-cable offset. This is the standard
       quasi-steady approximation; consistent with the note section 4.5.
    """
    geom = np.asarray(bundle_geometry, dtype=float)
    if geom.ndim != 2 or geom.shape[1] != 2:
        raise ValueError("bundle_geometry must be a sequence of (b, n) pairs")
    if diameter <= 0.0:
        raise ValueError("diameter must be positive")
    if air_density <= 0.0:
        raise ValueError("air_density must be positive")


    # Vérification du modèle choisi
    aero_model = aero_model.lower()
    if aero_model not in ["wu", "price", "diana"]:
        raise ValueError(f"Modèle aérodynamique inconnu : {aero_model}. Choisissez 'Wu', 'Price' ou 'Diana'.")

    n_cables = geom.shape[0]
    offset_B = geom[:, 0]
    offset_N = geom[:, 1]

    def force(
        s: np.ndarray,
        t: float,
        un_list: List[np.ndarray],
        ub_list: List[np.ndarray],
        vn_list: List[np.ndarray],
        vb_list: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if len(un_list) != n_cables:
            raise ValueError(
                f"bundle_geometry has {n_cables} cables but solver passed "
                f"{len(un_list)} state lists"
            )

        s_arr = np.asarray(s, dtype=float)

        # On interroge l'objet wind (issu de wind.py) pour obtenir la vitesse à l'instant t
        # wy = vent longitudinal (qui correspond à l'axe Binormal horizontal du câble)
        # wz = vent vertical (qui correspond à l'axe Normal vertical du câble)
        wy, wz = wind.velocity(s_arr, t)
        
        # Si le vent renvoie des scalaires (ex: ConstantWind), on les convertit en tableaux
        wB = np.asarray(wy, dtype=float) * np.ones_like(s_arr)
        wN = np.asarray(wz, dtype=float) * np.ones_like(s_arr)
        
        w_norm = np.hypot(wN, wB)
        valid_w = w_norm > 1e-12

        w_norm_safe = np.where(valid_w, w_norm, 1.0)

        fn_list = [np.zeros_like(s_arr) for _ in range(n_cables)]
        fb_list = [np.zeros_like(s_arr) for _ in range(n_cables)]

        if not np.any(valid_w):
            return fn_list, fb_list

        # Determine, per cable i and per node, the closest upstream cable.
        upstream_idx = -np.ones((n_cables, s_arr.size), dtype=int)
        y_rel_min = np.full((n_cables, s_arr.size), np.inf)
        z_rel_eff = np.zeros((n_cables, s_arr.size))

        for i in range(n_cables):
            uN_i = np.asarray(un_list[i], dtype=float)
            uB_i = np.asarray(ub_list[i], dtype=float)
            for j in range(n_cables):
                if i == j:
                    continue
                uN_j = np.asarray(un_list[j], dtype=float)
                uB_j = np.asarray(ub_list[j], dtype=float)
                dN = (offset_N[i] - offset_N[j]) + uN_i - uN_j
                dB = (offset_B[i] - offset_B[j]) + uB_i - uB_j
                y_rel = (dN * wN + dB * wB) / w_norm_safe
                z_rel = (-dN * wB + dB * wN) / w_norm_safe
                yd = y_rel / diameter
                zd = z_rel / diameter
                
                # On élargit la zone de détection pour englober les limites de Wu, Price et Diana
                in_wake = (
                    valid_w
                    & (yd > 0.0)
                    & (yd <= 30.0)
                    & (np.abs(zd) <= 6.0)
                )
                
                closer = in_wake & (y_rel < y_rel_min[i, :])
                upstream_idx[i, closer] = j
                y_rel_min[i, closer] = y_rel[closer]
                z_rel_eff[i, closer] = z_rel[closer]

        for i in range(n_cables):
            vN_i = np.asarray(vn_list[i], dtype=float)
            vB_i = np.asarray(vb_list[i], dtype=float)
            w_rel_N = wN - vN_i
            w_rel_B = wB - vB_i
            w_rel = np.hypot(w_rel_N, w_rel_B)

            cd = np.full_like(s_arr, float(cd_windward))
            cl = np.full_like(s_arr, float(cl_windward))
            mask = upstream_idx[i, :] >= 0
            if np.any(mask):
                yd_wake = y_rel_min[i, mask] / diameter
                zd_wake = z_rel_eff[i, mask] / diameter
                
                if aero_model == "wu":
                    cd_wake, cl_wake = wu2018_coefs(yd_wake, zd_wake)
                elif aero_model == "price":
                    cd_wake, cl_wake = price_coefs(yd_wake, zd_wake)
                elif aero_model == "diana":
                    cd_wake, cl_wake = diana_coefs(yd_wake, zd_wake)
                
                cd[mask] = cd_wake
                cl[mask] = cl_wake

            q = 0.5 * air_density * diameter * w_rel
            fn_list[i] = q * (w_rel_N * cd - w_rel_B * cl)
            fb_list[i] = q * (w_rel_B * cd + w_rel_N * cl)

        return fn_list, fb_list

    return force
