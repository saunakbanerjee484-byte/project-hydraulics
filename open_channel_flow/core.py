"""
core.py
=======
OpenChannelFlow: the central OOP engine. Wraps a CrossSection + flow
parameters (Q, n, S0, g) and exposes normal_depth(), critical_depth(),
velocity(), froude(), regime(), classify_channel().
"""
from __future__ import annotations

import math

from .geometry import CrossSection, GeometryError


class ConvergenceError(Exception):
    """Raised when a root-finding solver fails to converge."""


class OpenChannelFlow:
    def __init__(self, section: CrossSection, Q: float, n: float, S0: float, g: float = 9.81):
        if Q <= 0:
            raise GeometryError(f"Discharge Q must be positive, got Q={Q!r}")
        if n <= 0:
            raise GeometryError(f"Manning's n must be positive, got n={n!r}")
        if S0 <= 0:
            raise GeometryError(f"Bed slope S0 must be positive, got S0={S0!r}")
        if g <= 0:
            raise GeometryError(f"g must be positive, got g={g!r}")

        self.section = section
        self.Q = float(Q)
        self.n = float(n)
        self.S0 = float(S0)
        self.g = float(g)

    # ------------------------------------------------------------------
    # Basic hydraulics
    # ------------------------------------------------------------------
    def velocity(self, y: float) -> float:
        a = self.section.area(y)
        return self.Q / a

    def froude(self, y: float) -> float:
        a = self.section.area(y)
        t = self.section.top_width(y)
        v = self.Q / a
        d_hyd = a / t  # hydraulic depth
        return v / math.sqrt(self.g * d_hyd)

    def regime(self, y: float) -> str:
        fr = self.froude(y)
        if abs(fr - 1.0) < 1e-6:
            return "critical"
        return "subcritical" if fr < 1.0 else "supercritical"

    # ------------------------------------------------------------------
    # Root-finding helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _bisect(f, lo, hi, tol=1e-10, max_iter=200):
        f_lo, f_hi = f(lo), f(hi)
        if f_lo == 0:
            return lo
        if f_hi == 0:
            return hi
        if f_lo * f_hi > 0:
            raise ConvergenceError("Root is not bracketed; solver failed to converge.")
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            f_mid = f(mid)
            if abs(f_mid) < tol or (hi - lo) < tol:
                return mid
            if f_lo * f_mid <= 0:
                hi, f_hi = mid, f_mid
            else:
                lo, f_lo = mid, f_mid
        raise ConvergenceError("Bisection solver failed to converge within max_iter.")

    def _bracket_and_solve(self, f, y_guess=1.0):
        """Expand an interval outward from a small depth until f changes sign."""
        lo = 1e-6
        hi = max(y_guess, 1e-3)
        f_lo = f(lo)
        f_hi = f(hi)
        tries = 0
        while f_lo * f_hi > 0 and tries < 100:
            hi *= 1.5
            f_hi = f(hi)
            tries += 1
        if f_lo * f_hi > 0:
            raise ConvergenceError(
                "Could not bracket a root for the given Q/n/S0/geometry combination."
            )
        return self._bisect(f, lo, hi)

    # ------------------------------------------------------------------
    # Normal & critical depth
    # ------------------------------------------------------------------
    def normal_depth(self) -> float:
        """Solve Manning's equation Q = (1/n) A R^(2/3) sqrt(S0) for y."""

        def f(y):
            a = self.section.area(y)
            r = self.section.hydraulic_radius(y)
            return (1.0 / self.n) * a * r ** (2.0 / 3.0) * math.sqrt(self.S0) - self.Q

        return self._bracket_and_solve(f, y_guess=1.0)

    def critical_depth(self) -> float:
        """Solve Q^2 * T / (g * A^3) = 1 for y (Froude number = 1)."""

        def f(y):
            a = self.section.area(y)
            t = self.section.top_width(y)
            return (self.Q ** 2 * t) / (self.g * a ** 3) - 1.0

        return self._bracket_and_solve(f, y_guess=1.0)

    def classify_channel(self) -> str:
        yn = self.normal_depth()
        yc = self.critical_depth()
        if abs(yn - yc) < 1e-6:
            return "critical"
        return "mild" if yn > yc else "steep"
