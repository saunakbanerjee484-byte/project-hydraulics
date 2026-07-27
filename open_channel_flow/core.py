"""
core.py
=======
OpenChannelFlow Engine — Computational Fluid Dynamics & Open Channel Mechanics

Governing Physics Implemented:
--------------------------------
1. Continuity Equation (Conservation of Mass):
      Q = A(y) * V(y) = Constant

2. Manning's Resistance Equation (Uniform Flow Force Balance):
      Q = (1/n) * A * R^(2/3) * S0^(1/2)
      Driving Gravity Force = Resisting Boundary Shear Force

3. Specific Energy Principle (Bernoulli Theorem relative to bed):
      E(y) = y + V^2 / (2*g) = y + Q^2 / (2 * g * A^2)
      Critical Flow Condition (dE/dy = 0 => Fr = 1):
      Q^2 * T / (g * A^3) = 1

4. Shallow Water Wave Propagation Mechanics:
      Gravity Surface Wave Celerity: c = sqrt(g * D_h), where D_h = A / T
      Froude Number: Fr = V / c = V / sqrt(g * D_h)
      - Fr < 1.0 : Subcritical Flow (Tranquil, downstream wave control)
      - Fr = 1.0 : Critical Flow State (Minimum specific energy)
      - Fr > 1.0 : Supercritical Flow (Rapid, upstream wave control)
"""

from __future__ import annotations
import math
from .geometry import CrossSection, GeometryError


class ConvergenceError(Exception):
    """Raised when non-linear root solver fails to bracket or converge under extreme inputs."""


class OpenChannelFlow:
    """
    Central Physics Engine for 1D Steady Open Channel Hydraulics.
    """
    def __init__(self, section: CrossSection, Q: float, n: float, S0: float, g: float = 9.81):
        if Q <= 0:
            raise GeometryError(f"Discharge Q must be positive (got Q={Q!r}).")
        if n <= 0:
            raise GeometryError(f"Manning's roughness n must be positive (got n={n!r}).")
        if S0 <= 0:
            raise GeometryError(f"Channel bed slope S0 must be positive (got S0={S0!r}).")
        if g <= 0:
            raise GeometryError(f"Gravitational acceleration g must be positive (got g={g!r}).")

        self.section = section
        self.Q = float(Q)
        self.n = float(n)
        self.S0 = float(S0)
        self.g = float(g)

    # ------------------------------------------------------------------
    # FLUID KINEMATICS & DYNAMICS
    # ------------------------------------------------------------------
    def velocity(self, y: float) -> float:
        """Continuity Principle: V = Q / A(y)"""
        a = self.section.area(y)
        return self.Q / a

    def specific_energy(self, y: float) -> float:
        """
        Bernoulli Energy relative to Channel Bed:
        E(y) = y + V^2 / (2*g)
        """
        v = self.velocity(y)
        return y + (v ** 2) / (2.0 * self.g)

    def momentum_function(self, y: float) -> float:
        """
        Specific Force / Momentum Function (Belanger Jump Principle):
        M(y) = (Q^2 / (g * A)) + (z_bar * A)
        Momentum Flux + Hydrostatic Pressure Force
        """
        a = self.section.area(y)
        ay_bar = self.section.area_moment(y)
        return (self.Q ** 2) / (self.g * a) + ay_bar

    def bed_shear_stress(self, y: float, fluid_density: float = 1000.0) -> float:
        """
        Boundary Shear Stress:
        tau_0 = rho * g * R * S0 (N/m²)
        """
        r = self.section.hydraulic_radius(y)
        return fluid_density * self.g * r * self.S0

    def froude(self, y: float) -> float:
        """
        Froude Number: Fr = V / sqrt(g * D_h)
        Ratio of Bulk Flow Velocity to Gravity Wave Celerity
        """
        a = self.section.area(y)
        t = self.section.top_width(y)
        v = self.Q / a
        d_hyd = a / t  # Hydraulic Depth D_h
        celerity = math.sqrt(self.g * d_hyd)
        return v / celerity

    def regime(self, y: float) -> str:
        """Classify Flow State based on Wave Propagation Speed."""
        fr = self.froude(y)
        if abs(fr - 1.0) < 1e-4:
            return "critical"
        return "subcritical" if fr < 1.0 else "supercritical"

    # ------------------------------------------------------------------
    # MULTI-SCALE LOGARITHMIC ROOT SOLVER FOR EXTREME DATASETS
    # ------------------------------------------------------------------
    @staticmethod
    def _bisect(f, lo: float, hi: float, tol: float = 1e-10, max_iter: int = 300) -> float:
        """Bisection solver with guaranteed physical convergence bounds."""
        f_lo, f_hi = f(lo), f(hi)
        if f_lo == 0:
            return lo
        if f_hi == 0:
            return hi
        if f_lo * f_hi > 0:
            raise ConvergenceError("Physics Root Warning: State is not bracketed in interval.")
        
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            f_mid = f(mid)
            if abs(f_mid) < tol or (hi - lo) < tol:
                return mid
            if f_lo * f_mid <= 0:
                hi, f_hi = mid, f_mid
            else:
                lo, f_lo = mid, f_mid
        raise ConvergenceError("Bisection solver exceeded maximum iterations.")

    def _bracket_and_solve(self, f, y_guess: float = 1.0) -> float:
        """
        Logarithmic multi-scale domain scanner spanning micro-depths (1e-8 m) 
        to mega-depths (1e5 m) to handle extreme discharge & slope inputs without crashing.
        """
        max_physical_y = (self.section.D * 0.9999) if hasattr(self.section, "D") else 100000.0

        # Multi-scale Logarithmic Domain Brackets
        log_bounds = [
            1e-8, 1e-6, 1e-4, 1e-2, 0.1, 0.5, 
            1.0, 5.0, 20.0, 100.0, 1000.0, 10000.0, 100000.0
        ]

        for i in range(len(log_bounds) - 1):
            lo = log_bounds[i]
            hi = min(log_bounds[i + 1], max_physical_y)

            if lo >= max_physical_y:
                break

            try:
                f_lo = f(lo)
                f_hi = f(hi)
                if f_lo * f_hi <= 0:
                    return self._bisect(f, lo, hi)
            except (GeometryError, ValueError, OverflowError):
                continue

        # Dynamic fallback scan
        lo = 1e-8
        hi = min(max(y_guess, 1e-3), max_physical_y)
        tries = 0

        try:
            f_lo = f(lo)
        except GeometryError:
            f_lo = -self.Q

        while tries < 150:
            try:
                f_hi = f(hi)
                if f_lo * f_hi <= 0:
                    return self._bisect(f, lo, hi)
            except (GeometryError, ValueError, OverflowError):
                pass

            hi *= 2.0
            if hi >= max_physical_y:
                hi = max_physical_y
                try:
                    f_hi = f(hi)
                    if f_lo * f_hi <= 0:
                        return self._bisect(f, lo, hi)
                except GeometryError:
                    pass
                break
            tries += 1

        if hasattr(self.section, "D"):
            raise GeometryError(
                f"Physical Crown Limit: Discharge Q={self.Q:.2f} m³/s exceeds open channel capacity "
                f"for Pipe Diameter D={self.section.D:.2f}m."
            )

        raise ConvergenceError(
            f"Equilibrium Failure: Could not bracket depth for Q={self.Q}m³/s, S0={self.S0}, n={self.n}."
        )

    # ------------------------------------------------------------------
    # EQUATION SOLVERS
    # ------------------------------------------------------------------
    def normal_depth(self) -> float:
        """
        Solve Manning's Equation for Uniform Normal Depth (y_n):
        Resisting Wall Friction = Driving Gravity Force
        f(y) = (1/n) * A(y) * [R(y)]^(2/3) * sqrt(S0) - Q = 0
        """
        def f(y):
            a = self.section.area(y)
            r = self.section.hydraulic_radius(y)
            return (1.0 / self.n) * a * (r ** (2.0 / 3.0)) * math.sqrt(self.S0) - self.Q

        return self._bracket_and_solve(f, y_guess=1.0)

    def critical_depth(self) -> float:
        """
        Solve Minimum Energy State / Froude Number Unity Condition (y_c):
        dE/dy = 1 - (Q^2 * T) / (g * A^3) = 0  =>  (Q^2 * T) / (g * A^3) = 1
        """
        def f(y):
            a = self.section.area(y)
            t = self.section.top_width(y)
            return (self.Q ** 2 * t) / (self.g * (a ** 3)) - 1.0

        return self._bracket_and_solve(f, y_guess=1.0)

    def classify_channel(self) -> str:
        """
        Classify Channel Slope relative to Critical Depth Condition:
        - Mild Slope (yn > yc)  : Subcritical Flow
        - Steep Slope (yn < yc) : Supercritical Flow
        - Critical Slope (yn=yc): Critical State Flow
        """
        yn = self.normal_depth()
        yc = self.critical_depth()
        if abs(yn - yc) < 1e-4:
            return "critical"
        return "mild" if yn > yc else "steep"