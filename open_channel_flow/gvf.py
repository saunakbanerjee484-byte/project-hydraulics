"""
gvf.py
======
Gradually Varied Flow (GVF) profile integration.

dy/dx = (S0 - Sf) / (1 - Fr^2)

The integrator is direction-aware: subcritical flow is governed by a
DOWNSTREAM control, so the boundary depth is taken to apply at x = L
and the profile is marched backward (x = L -> 0); supercritical flow
is governed by an UPSTREAM control, so the boundary depth applies at
x = 0 and the profile is marched forward (x = 0 -> L).

A real GVF curve cannot pass smoothly through y = yc (the (1 - Fr^2)
denominator vanishes there) -- that transition is physically a
hydraulic jump, not a continuous curve. solve_profile() detects the
approach to critical depth and stops rather than grinding through to a
numerically meaningless blow-up; locate_jump() + stitch_uniform_flow()
(see jump.py / this module) fill in the rest of the reach with the
correct uniform-flow segment.

The RK4 step size is fixed, not adaptive -- very steep gradients near
yc can need a smaller dx for full accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .core import ConvergenceError


@dataclass
class GVFResult:
    x: List[float]
    y: List[float]
    curve_type: str
    regime: str
    channel_type: str = ""


def friction_slope(y: float, ocf) -> float:
    """Manning friction slope Sf = (n^2 * Q^2) / (A^2 * R^(4/3))."""
    a = ocf.section.area(y)
    r = ocf.section.hydraulic_radius(y)
    return (ocf.n ** 2 * ocf.Q ** 2) / (a ** 2 * r ** (4.0 / 3.0))


def _dydx(y: float, ocf) -> float:
    # Guard against non-physical depths reached by an RK4 *intermediate*
    # stage (k2/k3/k4 can overshoot past y=0 on a large trial step even
    # when the final accepted result would have been fine). Raising a
    # ConvergenceError here -- instead of letting geometry.py's
    # GeometryError propagate uncaught -- lets solve_profile()'s
    # step-halving refinement loop catch it and retry with a smaller
    # step, rather than the whole solve silently aborting.
    if y <= 1e-9:
        raise ConvergenceError("Depth went non-positive during an integration substep.")
    sf = friction_slope(y, ocf)
    fr = ocf.froude(y)
    denom = 1.0 - fr ** 2
    if abs(denom) < 1e-8:
        raise ZeroDivisionError("Profile is at critical depth; dy/dx is singular.")
    return (ocf.S0 - sf) / denom


def classify_curve(channel_type: str, y_start: float, yn: float, yc: float) -> str:
    """
    Classify the GVF curve type (M1-M3 for mild channels, S1-S3 for
    steep channels) based on where the boundary depth sits relative to
    yn and yc.
    """
    if channel_type == "mild":
        if y_start > yn and y_start > yc:
            return "M1 (backwater, y > yn > yc)"
        elif yc < y_start < yn:
            return "M2 (drawdown, yc < y < yn)"
        elif y_start < yc < yn:
            return "M3 (rapid rise, y < yc < yn)"
        else:
            return "M (mild channel, boundary case)"
    elif channel_type == "steep":
        if y_start > yc and yc > yn:
            return "S1 (backwater, y > yc > yn)"
        elif yn < y_start < yc:
            return "S2 (drawdown, yn < y < yc)"
        elif y_start < yn and yn < yc:
            return "S3 (rapid rise, y < yn < yc)"
        else:
            return "S (steep channel, boundary case)"
    else:
        return "C (critical channel, yn = yc)"


def _rk4_step(y, dx, ocf):
    k1 = _dydx(y, ocf)
    k2 = _dydx(y + 0.5 * dx * k1, ocf)
    k3 = _dydx(y + 0.5 * dx * k2, ocf)
    k4 = _dydx(y + dx * k3, ocf)
    return y + (dx / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def solve_profile(ocf, y_start: float, L: float, dx: float, yn: float, yc: float) -> GVFResult:
    """
    Integrate the GVF equation from the boundary depth y_start over a
    reach of length L, using fixed-step RK4. Direction is chosen
    automatically based on the Froude number of the boundary depth.
    """
    if L <= 0:
        raise ValueError("Reach length L must be positive.")
    if dx <= 0:
        raise ValueError("Step size dx must be positive.")
    if y_start <= 0:
        raise ValueError("Boundary depth y_start must be positive.")

    channel_type = "mild" if yn > yc else ("steep" if yn < yc else "critical")
    curve_type = classify_curve(channel_type, y_start, yn, yc)

    fr_start = ocf.froude(y_start)
    if abs(fr_start - 1.0) < 1e-6:
        raise ConvergenceError(
            "Boundary depth is at (or extremely near) critical depth; "
            "the GVF equation is singular there."
        )

    supercritical_control = fr_start > 1.0
    if supercritical_control:
        regime = "Supercritical control (upstream) -> solved upstream to downstream"
        step = dx
        x0 = 0.0
    else:
        regime = "Subcritical control (downstream) -> solved downstream to upstream"
        step = -dx
        x0 = L

    xs = [x0]
    ys = [y_start]
    x, y = x0, y_start
    x_end = x0 + L if supercritical_control else x0 - L
    min_h = abs(step) * 1e-4  # floor step size when refining near critical depth
    max_refine_tries = 40

    while (supercritical_control and x < x_end - 1e-9) or (not supercritical_control and x > x_end + 1e-9):
        fr_now = ocf.froude(y)
        if abs(fr_now - 1.0) < 1e-4:
            break

        cur_h = step
        # clip so the last step lands exactly on the boundary of the reach
        if supercritical_control and x + cur_h > x_end:
            cur_h = x_end - x
        elif not supercritical_control and x + cur_h < x_end:
            cur_h = x_end - x

        y_next = None
        y_trial = y
        hit_critical_exactly = False
        for _ in range(max_refine_tries):
            try:
                y_trial = _rk4_step(y, cur_h, ocf)
            except (ZeroDivisionError, ConvergenceError):
                cur_h /= 2.0
                continue
            if y_trial <= 0:
                cur_h /= 2.0
                continue
            try:
                fr_trial = ocf.froude(y_trial)
            except Exception:
                cur_h /= 2.0
                continue
            if (fr_now - 1.0) * (fr_trial - 1.0) < 0:
                # this step would cross critical depth -- shrink and retry
                # to properly resolve the curve as it approaches yc, rather
                # than snapping straight past it in one coarse jump.
                if abs(cur_h) <= min_h:
                    y_trial = yc
                    hit_critical_exactly = True
                    y_next = y_trial
                    break
                cur_h /= 2.0
                continue
            y_next = y_trial
            break
        else:
            # refinement budget exhausted -- accept the last trial value
            y_next = y_trial

        x = x + cur_h
        y = y_next
        xs.append(x)
        ys.append(y)
        if hit_critical_exactly:
            break

    if not supercritical_control:
        # We marched backward from x=L to x=0; reverse for ascending x order.
        xs = xs[::-1]
        ys = ys[::-1]

    return GVFResult(x=xs, y=ys, curve_type=curve_type, regime=regime, channel_type=channel_type)


def stitch_uniform_flow(gvf_result: GVFResult, yn: float, jump: Optional[dict]):
    """
    Fill in the uniform-flow (y = yn) segment on the far side of a
    detected jump (or, if no jump, on the far side of wherever the GVF
    integration stopped) so the reported/exported/plotted reach covers
    the full profile.
    """
    xs = list(gvf_result.x)
    ys = list(gvf_result.y)
    if not xs:
        return xs, ys

    if jump is not None:
        x_jump = jump["x_jump"]
        y2 = jump["y2"]
        # Insert the jump point, then continue at uniform depth to the
        # far end of whichever direction the curve was travelling.
        if xs[-1] >= xs[0]:
            # ascending x: jump likely near the end -> continue at yn beyond it
            if x_jump > xs[-1]:
                # extend a short uniform tail
                xs.append(x_jump)
                ys.append(y2)
            xs.append(xs[-1] + max(1.0, (xs[-1] - xs[0]) * 0.05 if len(xs) > 1 else 10.0))
            ys.append(yn)
        else:
            xs.append(xs[-1] - max(1.0, (xs[0] - xs[-1]) * 0.05 if len(xs) > 1 else 10.0))
            ys.append(yn)
    return xs, ys
