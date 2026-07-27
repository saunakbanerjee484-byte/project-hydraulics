"""
jump.py
=======
Hydraulic jump analysis: momentum (specific force) function, sequent
depth, energy loss, and mixed-regime jump location.

locate_jump() handles the two standard "mixed regime" textbook cases:

  Case A (steep channel, downstream control): a dam/weir forces a
  subcritical depth downstream of a channel whose normal flow is
  naturally supercritical. The jump sits where the backwater (M/S1)
  curve's depth matches the sequent depth of the uniform upstream flow.

  Case B (mild channel, upstream control): a sluice gate releases
  supercritical flow into a channel whose normal flow is subcritical.
  The jump sits where the sequent depth of the S-curve first matches
  the downstream normal depth.

It does NOT currently handle multiple jumps in one reach, jumps forced
by a mid-channel slope break (compound-slope channels), or submerged /
drowned jumps.

sequent_depth() and momentum_function() are general (work for any
CrossSection) and are cross-checked in the test suite against the
closed-form Belanger equation for rectangular channels.
"""
from __future__ import annotations

from .core import ConvergenceError
from .geometry import CrossSection


def momentum_function(y: float, Q: float, g: float, section: CrossSection) -> float:
    """Specific force / momentum function M(y) = Q^2/(g*A) + A*ybar."""
    a = section.area(y)
    a_ybar = section.area_moment(y)
    return (Q ** 2) / (g * a) + a_ybar


def sequent_depth(y1: float, Q: float, g: float, section: CrossSection,
                   tol: float = 1e-9, max_iter: int = 200) -> float:
    """
    Given an initial depth y1, find the sequent (conjugate) depth y2 on
    the other side of the jump such that M(y1) == M(y2), y2 != y1.
    General solver -- works for any CrossSection via momentum_function().
    """
    m_target = momentum_function(y1, Q, g, section)

    def f(y):
        return momentum_function(y, Q, g, section) - m_target

    # M(y) -> inf as y -> 0 and as y -> inf, with a single minimum at
    # critical depth, so there is exactly one other root besides y1.
    # Determine which side of the minimum y1 sits on via a local slope
    # probe, then expand a bracket strictly on the *other* side.
    step = max(y1 * 1e-4, 1e-6)
    slope = (momentum_function(y1 + step, Q, g, section) - momentum_function(y1 - step, Q, g, section))

    if slope < 0:
        # M is decreasing with y at y1 -> y1 is on the small-depth
        # (left / supercritical-ish) branch -> the other root is larger.
        lo = y1 * 1.001
        f_lo = f(lo)
        hi = y1 * 2.0
        f_hi = f(hi)
        tries = 0
        while f_hi <= 0 and tries < 200:
            hi *= 1.5
            f_hi = f(hi)
            tries += 1
    else:
        # M is increasing with y at y1 -> y1 is on the large-depth
        # (right / subcritical-ish) branch -> the other root is smaller.
        hi = y1 * 0.999
        f_hi = f(hi)
        lo = y1 * 0.5
        f_lo = f(lo)
        tries = 0
        while f_lo <= 0 and tries < 200:
            lo *= 0.5
            f_lo = f(lo)
            tries += 1

    if f_lo * f_hi > 0:
        raise ConvergenceError("Could not bracket the sequent depth.")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) < tol or (hi - lo) < tol:
            return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    raise ConvergenceError("Sequent-depth solver failed to converge.")


def jump_energy_loss(y1: float, y2: float, Q: float, g: float, section: CrossSection) -> float:
    """Energy dissipated across a hydraulic jump: dE = E(y1) - E(y2)."""
    from .energy import specific_energy
    e1 = specific_energy(y1, Q, g, section)
    e2 = specific_energy(y2, Q, g, section)
    return e1 - e2


def belanger_rectangular(y1: float, Q: float, g: float, B: float) -> float:
    """
    Closed-form Belanger equation for a RECTANGULAR channel, used only
    to cross-check the general sequent_depth() solver in tests.
        y2/y1 = 0.5 * (sqrt(1 + 8*Fr1^2) - 1)
    """
    q = Q / B  # discharge per unit width
    v1 = q / y1
    fr1 = v1 / (g * y1) ** 0.5
    return 0.5 * y1 * ((1 + 8 * fr1 ** 2) ** 0.5 - 1)


def locate_jump(gvf_result, ocf, yn: float, yc: float):
    """
    Scan a solved GVF profile for a mixed-regime hydraulic jump.

    Returns a dict {x_jump, y1, y2, delta_E} if a jump is located, or
    None if this profile/channel combination doesn't imply one.
    """
    Q, g, section = ocf.Q, ocf.g, ocf.section
    xs, ys = gvf_result.x, gvf_result.y
    if len(xs) < 2:
        return None

    channel_type = "mild" if yn > yc else ("steep" if yn < yc else "critical")
    if channel_type == "critical":
        return None

    if channel_type == "steep":
        # Case A: uniform upstream flow is supercritical (yn < yc). A
        # downstream-control subcritical backwater curve is what we
        # solved (gvf profile with y > yc somewhere). The jump occurs
        # where the backwater depth equals the sequent depth of yn.
        try:
            y2_target = sequent_depth(yn, Q, g, section)
        except ConvergenceError:
            return None
        diffs = [y - y2_target for y in ys]
    else:
        # Case B: mild channel, uniform downstream flow is subcritical
        # (yn > yc). An upstream sluice gate released supercritical flow
        # (gvf profile with y < yc somewhere, decelerating downstream).
        # The jump occurs where the *sequent depth of the curve* first
        # reaches yn.
        seq_depths = []
        for y in ys:
            try:
                seq_depths.append(sequent_depth(y, Q, g, section))
            except ConvergenceError:
                seq_depths.append(float("nan"))
        diffs = [sd - yn for sd in seq_depths]

    # Look for a sign change along the scan direction.
    for i in range(len(diffs) - 1):
        d0, d1 = diffs[i], diffs[i + 1]
        if d0 != d0 or d1 != d1:  # NaN guard
            continue
        if d0 == 0:
            x_jump = xs[i]
            break
        if d0 * d1 < 0:
            # linear interpolation for the crossing point
            frac = d0 / (d0 - d1)
            x_jump = xs[i] + frac * (xs[i + 1] - xs[i])
            break
    else:
        return None

    if channel_type == "steep":
        y1 = yn
        y2 = y2_target
    else:
        # interpolate the curve depth at x_jump to get y1 (supercritical side)
        y1 = ys[i] + (ys[i + 1] - ys[i]) * (
            0 if xs[i + 1] == xs[i] else (x_jump - xs[i]) / (xs[i + 1] - xs[i])
        )
        y2 = yn

    delta_e = jump_energy_loss(y1, y2, Q, g, section)
    return {"x_jump": x_jump, "y1": y1, "y2": y2, "delta_E": delta_e}
