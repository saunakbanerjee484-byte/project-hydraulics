"""
optimize.py
===========
Extension-lab features: most economical trapezoidal section design, and
Manning's-n roughness sensitivity sweeps.
"""
from __future__ import annotations

import math

from .core import OpenChannelFlow
from .geometry import Trapezoidal


def most_economical_trapezoidal(Q: float, n: float, S0: float, z: float, g: float = 9.81):
    """
    Find the least-wetted-perimeter (B, y) pair for a trapezoidal
    channel carrying discharge Q at slope S0, roughness n, and fixed
    side slope z, using scipy.optimize.minimize (SLSQP). Cross-checked
    against the closed-form best-hydraulic-section formula:
        B = 2*y*(sqrt(1+z^2) - z)
    """
    from scipy.optimize import minimize

    def perimeter(params):
        B, y = params
        if B <= 0 or y <= 0:
            return 1e6
        return B + 2 * y * math.sqrt(1 + z ** 2)

    def manning_constraint(params):
        B, y = params
        if B <= 0 or y <= 0:
            return -1e6
        section = Trapezoidal(B=B, z=z)
        a = section.area(y)
        p = section.wetted_perimeter(y)
        r = a / p
        q_calc = (1.0 / n) * a * r ** (2.0 / 3.0) * math.sqrt(S0)
        return q_calc - Q  # must equal zero

    x0 = [4.0, 2.0]
    result = minimize(
        perimeter, x0, method="SLSQP",
        constraints=[{"type": "eq", "fun": manning_constraint}],
        bounds=[(1e-3, None), (1e-3, None)],
    )
    B_opt, y_opt = result.x

    # closed-form best-hydraulic-section check: B = 2y(sqrt(1+z^2) - z)
    B_closed_form = 2 * y_opt * (math.sqrt(1 + z ** 2) - z)

    return {
        "B": B_opt,
        "y": y_opt,
        "perimeter": perimeter([B_opt, y_opt]),
        "B_closed_form_check": B_closed_form,
        "converged": bool(result.success),
    }


def roughness_sensitivity(section_factory, Q: float, S0: float, n_values, g: float = 9.81):
    """
    Sweep a range of Manning's n and return the resulting normal depth
    for each, using a fresh section (from section_factory()) for every
    solve. section_factory: zero-arg callable returning a CrossSection.
    """
    results = []
    for n in n_values:
        section = section_factory()
        ocf = OpenChannelFlow(section=section, Q=Q, n=n, S0=S0, g=g)
        yn = ocf.normal_depth()
        results.append({"n": n, "yn": yn})
    return results
