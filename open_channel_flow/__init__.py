"""
open_channel_flow
==================
Object-oriented core engine for open-channel hydraulics: uniform flow,
critical flow, specific energy, gradually varied flow (GVF), and
hydraulic jumps.
"""

from .geometry import Rectangular, Triangular, Trapezoidal, Circular, GeometryError
from .core import OpenChannelFlow, ConvergenceError

__all__ = [
    "Rectangular", "Triangular", "Trapezoidal", "Circular", "GeometryError",
    "OpenChannelFlow", "ConvergenceError",
]

__version__ = "1.0.0"
