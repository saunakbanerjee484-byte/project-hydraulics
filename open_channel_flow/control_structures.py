"""
control_structures.py
======================
Derive a GVF boundary depth from a real hydraulic structure.
"""
from __future__ import annotations

import math

from .geometry import GeometryError


def sluice_gate_depth(a: float, Cc: float) -> float:
    """
    Depth just downstream of a sluice gate: the vena-contracta depth
    y = Cc * a, where a is the gate opening and Cc the contraction
    coefficient.
    """
    if a <= 0:
        raise GeometryError(f"Gate opening a must be positive, got a={a!r}")
    if not (0 < Cc <= 1):
        raise GeometryError(f"Contraction coefficient Cc must be in (0, 1], got Cc={Cc!r}")
    return Cc * a


def weir_downstream_depth(Q: float, L: float, P: float, Cw: float) -> float:
    """
    Depth immediately downstream of a sharp-crested weir (pool
    elevation), using the standard weir equation Q = Cw * L * H^(3/2)
    solved for head H, then added to the crest height P above the bed.
    """
    if Q <= 0:
        raise GeometryError(f"Discharge Q must be positive, got Q={Q!r}")
    if L <= 0:
        raise GeometryError(f"Weir length L must be positive, got L={L!r}")
    if P < 0:
        raise GeometryError(f"Weir crest height P must be non-negative, got P={P!r}")
    if Cw <= 0:
        raise GeometryError(f"Weir coefficient Cw must be positive, got Cw={Cw!r}")

    H = (Q / (Cw * L)) ** (2.0 / 3.0)
    return P + H
