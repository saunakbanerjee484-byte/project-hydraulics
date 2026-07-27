"""
geometry.py
===========
Polymorphic Hydraulic Cross-Section Physics Engine.
Supports Prismatic Channels (Rectangular, Triangular, Trapezoidal, Circular, Parabolic)
and Non-Prismatic / Irregular Natural River Channels (HEC-RAS Coordinate Style).

GOVERNING PHYSICAL CONCEPTS:
-----------------------------
1. Cross-Sectional Area A(y) = Integral[ b(z) dz, {0, y} ]
2. Top Width T(y) = dA/dy  (Water surface width)
3. Wetted Perimeter P(y) = Solid boundary friction contact length
4. Hydraulic Radius R(y) = A(y) / P(y)  [Manning Wall Shear Scaling Length]
5. Hydraulic Depth D_h(y) = A(y) / T(y) [Surface Wave Celerity Scaling Length: c = sqrt(g * D_h)]
6. Hydrostatic Area Moment z_bar * A(y) = Integral[ (y - z) * b(z) dz, {0, y} ]
   [Used in Specific Force / Momentum Equation for Hydraulic Jumps]
"""

from __future__ import annotations
import math
import numpy as np


class GeometryError(Exception):
    """Custom exception raised for physically impossible channel dimensions or parameters."""
    pass


class CrossSection:
    """Base Polymorphic Class for Channel Hydraulics Geometry."""

    def area(self, y: float) -> float:
        raise NotImplementedError

    def top_width(self, y: float) -> float:
        raise NotImplementedError

    def wetted_perimeter(self, y: float) -> float:
        raise NotImplementedError

    def hydraulic_radius(self, y: float) -> float:
        """Hydraulic Radius R = A / P (Boundary Friction Scaling)"""
        A = self.area(y)
        P = self.wetted_perimeter(y)
        if P <= 0:
            raise GeometryError("Wetted perimeter must be strictly positive (P > 0).")
        return A / P

    def hydraulic_depth(self, y: float) -> float:
        """Hydraulic Depth D_h = A / T (Wave Speed Scaling: c = sqrt(g * D_h))"""
        T = self.top_width(y)
        if T <= 0:
            raise GeometryError("Top width must be strictly positive (T > 0).")
        return self.area(y) / T

    def area_moment(self, y: float) -> float:
        """First moment of area about water surface (z_bar * A) for Hydrostatic Force"""
        raise NotImplementedError

    def validate_depth(self, y: float) -> float:
        if y <= 0:
            raise GeometryError(f"Water depth y must be strictly positive (got y={y}).")
        return y


# =============================================================================
# PRISMATIC CHANNEL SECTIONS (Constant Cross-Section along Reach)
# =============================================================================

class Rectangular(CrossSection):
    """
    Prismatic Rectangular Channel:
    Physics:
    - Area A = B * y
    - Top Width T = B
    - Wetted Perimeter P = B + 2y
    - Area Moment (z_bar * A) = B * y^2 / 2  [Linear Hydrostatic Pressure]
    """
    def __init__(self, B: float):
        if B <= 0:
            raise GeometryError(f"Bottom width B must be positive (got {B}).")
        self.B = float(B)

    def area(self, y: float) -> float:
        self.validate_depth(y)
        return self.B * y

    def top_width(self, y: float) -> float:
        self.validate_depth(y)
        return self.B

    def wetted_perimeter(self, y: float) -> float:
        self.validate_depth(y)
        return self.B + 2.0 * y

    def area_moment(self, y: float) -> float:
        self.validate_depth(y)
        return self.B * (y ** 2) / 2.0


class Triangular(CrossSection):
    """
    Prismatic Triangular Channel (Side slope z H:V):
    Physics:
    - Area A = z * y^2
    - Top Width T = 2 * z * y
    - Wetted Perimeter P = 2 * y * sqrt(1 + z^2)
    - Area Moment (z_bar * A) = z * y^3 / 3  [Triangular Hydrostatic Centroid at y/3]
    """
    def __init__(self, z: float):
        if z <= 0:
            raise GeometryError(f"Side slope z (H:V) must be positive (got {z}).")
        self.z = float(z)

    def area(self, y: float) -> float:
        self.validate_depth(y)
        return self.z * (y ** 2)

    def top_width(self, y: float) -> float:
        self.validate_depth(y)
        return 2.0 * self.z * y

    def wetted_perimeter(self, y: float) -> float:
        self.validate_depth(y)
        return 2.0 * y * math.sqrt(1.0 + self.z ** 2)

    def area_moment(self, y: float) -> float:
        self.validate_depth(y)
        return (self.z * (y ** 3)) / 3.0


class Trapezoidal(CrossSection):
    """
    Prismatic Trapezoidal Channel:
    Physics:
    - Area A = (B + z*y) * y
    - Top Width T = B + 2*z*y
    - Wetted Perimeter P = B + 2*y*sqrt(1 + z^2)
    - Area Moment (z_bar * A) = (B*y^2 / 2) + (z*y^3 / 3)
    """
    def __init__(self, B: float, z: float):
        if B <= 0 or z < 0:
            raise GeometryError(f"Invalid Trapezoid dimensions (B={B}, z={z}).")
        self.B = float(B)
        self.z = float(z)

    def area(self, y: float) -> float:
        self.validate_depth(y)
        return (self.B + self.z * y) * y

    def top_width(self, y: float) -> float:
        self.validate_depth(y)
        return self.B + 2.0 * self.z * y

    def wetted_perimeter(self, y: float) -> float:
        self.validate_depth(y)
        return self.B + 2.0 * y * math.sqrt(1.0 + self.z ** 2)

    def area_moment(self, y: float) -> float:
        self.validate_depth(y)
        return (self.B * y ** 2 / 2.0) + (self.z * y ** 3 / 3.0)


class Circular(CrossSection):
    """
    Prismatic Circular Conduit (Partially Filled Open Channel Conduit):
    Physics:
    - Submerged Angle theta = 2 * acos(1 - 2y/D)
    - Area A = (D^2 / 8) * (theta - sin(theta))
    - Top Width T = D * sin(theta / 2)
    - Wetted Perimeter P = (D / 2) * theta
    - Closed-Form Analytical Hydrostatic Area Moment:
      (z_bar * A) = (D^3 / 12) * sin^3(theta/2) - A * (D/2 - y)
    """
    def __init__(self, D: float):
        if D <= 0:
            raise GeometryError(f"Pipe Diameter D must be positive (got {D}).")
        self.D = float(D)

    def validate_depth(self, y: float) -> float:
        if y <= 0:
            raise GeometryError(f"Water depth y must be strictly positive (got y={y}).")
        # Self-Healing Physical Crown Guard (Open Channel Boundary: y < D)
        if y >= self.D:
            return self.D * 0.9999
        return y

    def _theta(self, y: float) -> float:
        y_safe = self.validate_depth(y)
        ratio = max(-1.0, min(1.0, 1.0 - (2.0 * y_safe / self.D)))
        return 2.0 * math.acos(ratio)

    def area(self, y: float) -> float:
        y_safe = self.validate_depth(y)
        th = self._theta(y_safe)
        return (self.D ** 2 / 8.0) * (th - math.sin(th))

    def top_width(self, y: float) -> float:
        y_safe = self.validate_depth(y)
        th = self._theta(y_safe)
        return max(1e-6, self.D * math.sin(th / 2.0))

    def wetted_perimeter(self, y: float) -> float:
        y_safe = self.validate_depth(y)
        th = self._theta(y_safe)
        return (self.D / 2.0) * th

    def area_moment(self, y: float) -> float:
        y_safe = self.validate_depth(y)
        th = self._theta(y_safe)
        A = self.area(y_safe)
        return (self.D ** 3 / 12.0) * (math.sin(th / 2.0) ** 3) - A * (self.D / 2.0 - y_safe)


class Parabolic(CrossSection):
    """
    Prismatic Parabolic Channel (Natural Stream & Irrigation Channels):
    Governing Equation: Water Surface Top Width T = C * sqrt(y)
    Physics Integration:
    - Area A = Integrate[ C*sqrt(z) dz, {0, y} ] = (2/3) * C * y^(1.5) = (2/3) * T * y
    - Wetted Perimeter P = Series Arc Length Expansion over Parabolic Curve
    - Hydrostatic Area Moment (z_bar * A) = (2/5) * C * y^(2.5)
    """
    def __init__(self, C: float):
        if C <= 0:
            raise GeometryError(f"Top width coefficient C must be positive (got {C}).")
        self.C = float(C)

    def area(self, y: float) -> float:
        self.validate_depth(y)
        return (2.0 / 3.0) * self.C * (y ** 1.5)

    def top_width(self, y: float) -> float:
        self.validate_depth(y)
        return self.C * math.sqrt(y)

    def wetted_perimeter(self, y: float) -> float:
        self.validate_depth(y)
        T = self.top_width(y)
        x = 4.0 * y / T
        if x < 0.001:
            return T
        # Series approximation for arc length of a parabola: P = T * (1 + 2/3*x^2 - 2/5*x^4 + ...)
        return T * (1.0 + (2.0 / 3.0) * (x ** 2) - (2.0 / 5.0) * (x ** 4))

    def area_moment(self, y: float) -> float:
        self.validate_depth(y)
        return (2.0 / 5.0) * self.C * (y ** 2.5)


# =============================================================================
# NON-PRISMATIC / IRREGULAR NATURAL RIVER CHANNEL SECTIONS
# =============================================================================

class IrregularSection(CrossSection):
    """
    Natural River Cross-Section via Station-Elevation Coordinates (HEC-RAS Standard).
    Solves Area, Perimeter, and Centroid Moments via Discrete Trapezoidal Numerical Quadrature.
    """
    def __init__(self, stations: list[float], elevations: list[float]):
        if len(stations) != len(elevations) or len(stations) < 3:
            raise GeometryError("Irregular section requires at least 3 (x, z) coordinate points.")
        self.x = np.array(stations, dtype=float)
        self.z = np.array(elevations, dtype=float)
        self.z_min = float(np.min(self.z))

    def _get_submerged_segments(self, y: float):
        self.validate_depth(y)
        z_water = self.z_min + y
        sub_x, sub_z = [], []

        for i in range(len(self.x) - 1):
            x1, z1 = self.x[i], self.z[i]
            x2, z2 = self.x[i + 1], self.z[i + 1]

            if z1 <= z_water or z2 <= z_water:
                if (z1 - z_water) * (z2 - z_water) < 0:
                    frac = (z_water - z1) / (z2 - z1)
                    x_int = x1 + frac * (x2 - x1)
                    if z1 > z_water:
                        sub_x.extend([x_int, x2])
                        sub_z.extend([z_water, z2])
                    else:
                        sub_x.extend([x1, x_int])
                        sub_z.extend([z1, z_water])
                else:
                    sub_x.extend([x1, x2])
                    sub_z.extend([z1, z_water])

        return np.array(sub_x), np.array(sub_z), z_water

    def area(self, y: float) -> float:
        sub_x, sub_z, z_w = self._get_submerged_segments(y)
        if len(sub_x) < 2:
            return 1e-6
        return float(np.trapz(z_w - sub_z, sub_x))

    def top_width(self, y: float) -> float:
        sub_x, _, _ = self._get_submerged_segments(y)
        if len(sub_x) < 2:
            return 1e-6
        return float(np.max(sub_x) - np.min(sub_x))

    def wetted_perimeter(self, y: float) -> float:
        sub_x, sub_z, _ = self._get_submerged_segments(y)
        if len(sub_x) < 2:
            return 1e-6
        dx = np.diff(sub_x)
        dz = np.diff(sub_z)
        return float(np.sum(np.sqrt(dx ** 2 + dz ** 2)))

    def area_moment(self, y: float) -> float:
        sub_x, sub_z, z_w = self._get_submerged_segments(y)
        moment = 0.0
        for i in range(len(sub_x) - 1):
            dx = sub_x[i + 1] - sub_x[i]
            d1 = z_w - sub_z[i]
            d2 = z_w - sub_z[i + 1]
            if dx > 0 and (d1 > 0 or d2 > 0):
                strip_area = 0.5 * (d1 + d2) * dx
                strip_ybar = (d1 ** 2 + d1 * d2 + d2 ** 2) / (3.0 * (d1 + d2) + 1e-9)
                moment += strip_area * strip_ybar
        return float(moment)