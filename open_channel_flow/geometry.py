"""
geometry.py
===========
Cross-section geometry for open-channel hydraulics.

Every shape implements the same interface:
    area(y), top_width(y), wetted_perimeter(y), hydraulic_radius(y),
    area_moment(y)

so the rest of the engine (Manning solver, critical depth, GVF, jump
momentum function) works identically regardless of which CrossSection
instance is plugged in.

area_moment(y) returns the static moment of the flow area about the
free surface, i.e.  A(y) * ybar(y) = integral_0^y (y - h) * T(h) dh,
where T(h) is the top width at height h above the invert. This is the
quantity needed by the momentum function M(y) = Q^2/(g*A) + A*ybar.

Every shape (including Circular) now has a closed-form area_moment;
the base class's numerical (Simpson's rule) integration remains as a
fallback/cross-check for any future shape that doesn't implement one.

Physical validity is enforced here:
    - bad construction parameters (negative/zero widths, slopes, etc.)
      raise GeometryError immediately at construction time.
    - non-positive depth raises GeometryError the first time it is used.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod


class GeometryError(Exception):
    """Raised for invalid channel geometry or invalid depth input."""


class CrossSection(ABC):
    """Abstract base class for a channel cross-section shape."""

    def _check_depth(self, y: float) -> float:
        y = float(y)
        if y <= 0:
            raise GeometryError(f"Depth must be positive, got y={y!r}")
        return y

    @abstractmethod
    def area(self, y: float) -> float:
        """Flow cross-sectional area at depth y (m^2)."""

    @abstractmethod
    def top_width(self, y: float) -> float:
        """Free-surface width at depth y (m)."""

    @abstractmethod
    def wetted_perimeter(self, y: float) -> float:
        """Wetted perimeter at depth y (m)."""

    def hydraulic_radius(self, y: float) -> float:
        """R = A / P."""
        a = self.area(y)
        p = self.wetted_perimeter(y)
        if p <= 0:
            raise GeometryError("Wetted perimeter is non-positive; cannot compute R.")
        return a / p

    def area_moment(self, y: float) -> float:
        """
        Static moment of area about the free surface, A*ybar.
        Default: numerical integration (Simpson's rule) of
        integral_0^y (y - h) * T(h) dh. Subclasses may override with a
        closed form for speed/accuracy.
        """
        y = self._check_depth(y)
        n = 200  # even number of intervals for Simpson's rule
        h_vals = [y * i / n for i in range(n + 1)]

        def integrand(h):
            if h >= y:
                return 0.0
            return (y - h) * self.top_width(h if h > 0 else 1e-12)

        vals = [integrand(h) for h in h_vals]
        step = y / n
        s = vals[0] + vals[-1]
        s += 4 * sum(vals[1:-1:2])
        s += 2 * sum(vals[2:-1:2])
        return s * step / 3.0


class Rectangular(CrossSection):
    """Rectangular channel of bottom width B."""

    def __init__(self, B: float):
        if B <= 0:
            raise GeometryError(f"Bottom width B must be positive, got B={B!r}")
        self.B = float(B)

    def area(self, y: float) -> float:
        y = self._check_depth(y)
        return self.B * y

    def top_width(self, y: float) -> float:
        y = self._check_depth(y)
        return self.B

    def wetted_perimeter(self, y: float) -> float:
        y = self._check_depth(y)
        return self.B + 2 * y

    def area_moment(self, y: float) -> float:
        y = self._check_depth(y)
        return self.B * y ** 2 / 2.0

    def __repr__(self):
        return f"Rectangular(B={self.B})"


class Triangular(CrossSection):
    """Symmetric triangular (V-shaped) channel with side slope z (H:V = z:1)."""

    def __init__(self, z: float):
        if z <= 0:
            raise GeometryError(f"Side slope z must be positive, got z={z!r}")
        self.z = float(z)

    def area(self, y: float) -> float:
        y = self._check_depth(y)
        return self.z * y ** 2

    def top_width(self, y: float) -> float:
        y = self._check_depth(y)
        return 2 * self.z * y

    def wetted_perimeter(self, y: float) -> float:
        y = self._check_depth(y)
        return 2 * y * math.sqrt(1 + self.z ** 2)

    def area_moment(self, y: float) -> float:
        y = self._check_depth(y)
        return self.z * y ** 3 / 3.0

    def __repr__(self):
        return f"Triangular(z={self.z})"


class Trapezoidal(CrossSection):
    """Trapezoidal channel: bottom width B, side slope z (H:V = z:1)."""

    def __init__(self, B: float, z: float):
        if B <= 0:
            raise GeometryError(f"Bottom width B must be positive, got B={B!r}")
        if z < 0:
            raise GeometryError(f"Side slope z must be non-negative, got z={z!r}")
        self.B = float(B)
        self.z = float(z)

    def area(self, y: float) -> float:
        y = self._check_depth(y)
        return self.B * y + self.z * y ** 2

    def top_width(self, y: float) -> float:
        y = self._check_depth(y)
        return self.B + 2 * self.z * y

    def wetted_perimeter(self, y: float) -> float:
        y = self._check_depth(y)
        return self.B + 2 * y * math.sqrt(1 + self.z ** 2)

    def area_moment(self, y: float) -> float:
        y = self._check_depth(y)
        return self.B * y ** 2 / 2.0 + self.z * y ** 3 / 3.0

    def __repr__(self):
        return f"Trapezoidal(B={self.B}, z={self.z})"


class Circular(CrossSection):
    """Circular (pipe) channel of diameter D, partially full."""

    def __init__(self, D: float):
        if D <= 0:
            raise GeometryError(f"Diameter D must be positive, got D={D!r}")
        self.D = float(D)

    def _theta(self, y: float) -> float:
        R = self.D / 2.0
        ratio = max(-1.0, min(1.0, (y - R) / R))
        return 2 * math.acos(-ratio)  # 0..2*pi, using (1 - 2y/D) form below instead

    def area(self, y: float) -> float:
        y = self._check_depth(y)
        if y >= self.D:
            raise GeometryError("Depth exceeds pipe diameter; pipe would be surcharged/full.")
        R = self.D / 2.0
        arg = max(-1.0, min(1.0, 1 - 2 * y / self.D))
        theta = 2 * math.acos(arg)
        return (R ** 2 / 2.0) * (theta - math.sin(theta))

    def top_width(self, y: float) -> float:
        y = self._check_depth(y)
        if y >= self.D:
            raise GeometryError("Depth exceeds pipe diameter; pipe would be surcharged/full.")
        return 2 * math.sqrt(max(0.0, y * (self.D - y)))

    def wetted_perimeter(self, y: float) -> float:
        y = self._check_depth(y)
        if y >= self.D:
            raise GeometryError("Depth exceeds pipe diameter; pipe would be surcharged/full.")
        R = self.D / 2.0
        arg = max(-1.0, min(1.0, 1 - 2 * y / self.D))
        theta = 2 * math.acos(arg)
        return R * theta

    def area_moment(self, y: float) -> float:
        """
        Closed-form static moment of area about the free surface for a
        circular segment:
            A*ybar = R^3 * (sin(phi) - phi*cos(phi) - sin(phi)^3 / 3)
        where phi = arccos(1 - y/R) is the half central angle subtended
        by the water surface. Derived by substituting h = R(1 - cos a)
        into the general integral definition and verified numerically
        against Simpson's-rule integration to double precision.
        """
        y = self._check_depth(y)
        if y >= self.D:
            raise GeometryError("Depth exceeds pipe diameter; pipe would be surcharged/full.")
        R = self.D / 2.0
        arg = max(-1.0, min(1.0, 1 - y / R))
        phi = math.acos(arg)
        return R ** 3 * (math.sin(phi) - phi * math.cos(phi) - math.sin(phi) ** 3 / 3.0)

    def __repr__(self):
        return f"Circular(D={self.D})"
