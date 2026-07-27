"""
energy.py
=========
Specific energy diagram: E(y) = y + Q^2 / (2 g A(y)^2).
"""
from __future__ import annotations

from .core import ConvergenceError
from .geometry import CrossSection


def specific_energy(y: float, Q: float, g: float, section: CrossSection) -> float:
    a = section.area(y)
    return y + (Q ** 2) / (2 * g * a ** 2)


def alternate_depth(y1: float, Q: float, g: float, section: CrossSection, yc: float,
                     tol: float = 1e-9, max_iter: int = 200) -> float:
    """
    Given a depth y1 (on one side of critical depth), find the alternate
    depth y2 on the other side with the same specific energy.
    """
    e_target = specific_energy(y1, Q, g, section)

    def f(y):
        return specific_energy(y, Q, g, section) - e_target

    if y1 < yc:
        # y1 is supercritical branch -> search subcritical branch (y > yc)
        lo, hi = yc * 1.0000001, max(yc * 5, y1 * 5, 1.0)
    else:
        # y1 is subcritical branch -> search supercritical branch (y < yc)
        lo, hi = 1e-6, yc * 0.9999999

    f_lo, f_hi = f(lo), f(hi)
    tries = 0
    while f_lo * f_hi > 0 and tries < 100:
        if y1 < yc:
            hi *= 1.5
        else:
            lo *= 0.5
        f_lo, f_hi = f(lo), f(hi)
        tries += 1
    if f_lo * f_hi > 0:
        raise ConvergenceError("Could not bracket the alternate depth.")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) < tol or (hi - lo) < tol:
            return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    raise ConvergenceError("Alternate-depth solver failed to converge.")


def energy_curve(Q: float, g: float, section: CrossSection, yn: float, yc: float, n_points: int = 200):
    """
    Build arrays for plotting the specific-energy (E-y) diagram.

    Returns (y_curve, E_curve, e_min_point) where e_min_point is a
    2-tuple (E_min, yc) marking the critical-depth point on the curve.
    """
    y_max = max(yn, yc) * 3.0
    y_min = y_max / n_points
    y_curve = [y_min + (y_max - y_min) * i / (n_points - 1) for i in range(n_points)]
    E_curve = [specific_energy(y, Q, g, section) for y in y_curve]
    e_min = specific_energy(yc, Q, g, section)
    return y_curve, E_curve, (e_min, yc)
