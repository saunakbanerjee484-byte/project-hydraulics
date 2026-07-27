"""
tests/test_hydraulics.py
=========================
Unit tests for the open_channel_flow engine: geometry, core hydraulics,
specific energy, GVF integration, and hydraulic jump detection.
"""
import math

import pytest

from open_channel_flow.geometry import (
    Rectangular, Triangular, Trapezoidal, Circular, GeometryError,
)
from open_channel_flow.core import OpenChannelFlow, ConvergenceError
from open_channel_flow.energy import specific_energy, alternate_depth
from open_channel_flow.gvf import friction_slope, solve_profile, classify_curve
from open_channel_flow.jump import (
    momentum_function, sequent_depth, jump_energy_loss, belanger_rectangular, locate_jump,
)

G = 9.81


# ---------------------------------------------------------------------
# geometry.py
# ---------------------------------------------------------------------

def test_rectangular_geometry():
    sec = Rectangular(B=4.0)
    y = 2.0
    assert sec.area(y) == pytest.approx(8.0)
    assert sec.top_width(y) == pytest.approx(4.0)
    assert sec.wetted_perimeter(y) == pytest.approx(8.0)
    assert sec.hydraulic_radius(y) == pytest.approx(1.0)


def test_triangular_geometry():
    sec = Triangular(z=1.5)
    y = 2.0
    assert sec.area(y) == pytest.approx(1.5 * y ** 2)
    assert sec.top_width(y) == pytest.approx(2 * 1.5 * y)
    expected_p = 2 * y * math.sqrt(1 + 1.5 ** 2)
    assert sec.wetted_perimeter(y) == pytest.approx(expected_p)


def test_trapezoidal_geometry_matches_rect_plus_triangle():
    B, z, y = 4.0, 1.5, 2.0
    trap = Trapezoidal(B=B, z=z)
    rect_area = B * y
    tri_area = z * y ** 2
    assert trap.area(y) == pytest.approx(rect_area + tri_area)
    assert trap.area_moment(y) == pytest.approx(B * y ** 2 / 2.0 + z * y ** 3 / 3.0)


def test_circular_geometry_half_full():
    D = 2.0
    sec = Circular(D=D)
    y = D / 2.0  # half full
    # half-full pipe: area = pi*R^2/2, top width = D
    R = D / 2.0
    assert sec.area(y) == pytest.approx(math.pi * R ** 2 / 2.0, rel=1e-6)
    assert sec.top_width(y) == pytest.approx(D, rel=1e-6)


def test_circular_closed_form_area_moment_matches_numerical_integration():
    from open_channel_flow.geometry import CrossSection

    sec = Circular(D=2.0)
    # Force the base-class numerical (Simpson's rule) integrator for
    # comparison, bypassing Circular's closed-form override.
    numeric_moment = CrossSection.area_moment(sec, 1.2)
    closed_form_moment = sec.area_moment(1.2)
    assert closed_form_moment == pytest.approx(numeric_moment, rel=2e-3)
    assert closed_form_moment > 0


def test_negative_width_raises_at_construction():
    with pytest.raises(GeometryError):
        Rectangular(B=-1.0)
    with pytest.raises(GeometryError):
        Triangular(z=0)
    with pytest.raises(GeometryError):
        Trapezoidal(B=4.0, z=-0.5)


def test_nonpositive_depth_raises_on_use():
    sec = Rectangular(B=4.0)
    with pytest.raises(GeometryError):
        sec.area(0.0)
    with pytest.raises(GeometryError):
        sec.area(-1.0)


# ---------------------------------------------------------------------
# core.py
# ---------------------------------------------------------------------

def test_normal_depth_satisfies_manning_equation():
    sec = Rectangular(B=4.0)
    ocf = OpenChannelFlow(section=sec, Q=20.0, n=0.025, S0=0.001)
    yn = ocf.normal_depth()
    a = sec.area(yn)
    r = sec.hydraulic_radius(yn)
    q_check = (1.0 / ocf.n) * a * r ** (2.0 / 3.0) * math.sqrt(ocf.S0)
    assert q_check == pytest.approx(ocf.Q, rel=1e-4)


def test_critical_depth_matches_rectangular_closed_form():
    B, Q = 4.0, 20.0
    sec = Rectangular(B=B)
    ocf = OpenChannelFlow(section=sec, Q=Q, n=0.025, S0=0.001)
    yc = ocf.critical_depth()
    q = Q / B
    yc_closed_form = (q ** 2 / G) ** (1.0 / 3.0)
    assert yc == pytest.approx(yc_closed_form, rel=1e-4)


def test_froude_and_regime_consistency():
    sec = Rectangular(B=4.0)
    ocf = OpenChannelFlow(section=sec, Q=20.0, n=0.025, S0=0.001)
    yc = ocf.critical_depth()
    assert ocf.regime(yc * 1.5) == "subcritical"
    assert ocf.regime(yc * 0.5) == "supercritical"
    assert ocf.froude(yc) == pytest.approx(1.0, rel=1e-3)


def test_classify_channel_mild_vs_steep():
    sec = Rectangular(B=4.0)
    mild = OpenChannelFlow(section=sec, Q=20.0, n=0.025, S0=0.0005)
    steep = OpenChannelFlow(section=sec, Q=20.0, n=0.015, S0=0.02)
    assert mild.classify_channel() == "mild"
    assert steep.classify_channel() == "steep"


def test_convergence_error_on_bad_construction():
    sec = Rectangular(B=4.0)
    with pytest.raises(GeometryError):
        OpenChannelFlow(section=sec, Q=-5.0, n=0.025, S0=0.001)


# ---------------------------------------------------------------------
# energy.py
# ---------------------------------------------------------------------

def test_specific_energy_minimum_near_critical_depth():
    sec = Rectangular(B=4.0)
    ocf = OpenChannelFlow(section=sec, Q=20.0, n=0.025, S0=0.001)
    yc = ocf.critical_depth()
    e_at_c = specific_energy(yc, ocf.Q, ocf.g, sec)
    e_below = specific_energy(yc * 0.9, ocf.Q, ocf.g, sec)
    e_above = specific_energy(yc * 1.1, ocf.Q, ocf.g, sec)
    assert e_at_c < e_below
    assert e_at_c < e_above


def test_alternate_depth_has_same_energy():
    sec = Rectangular(B=4.0)
    ocf = OpenChannelFlow(section=sec, Q=20.0, n=0.025, S0=0.001)
    yc = ocf.critical_depth()
    y1 = yc * 1.5  # subcritical
    y2 = alternate_depth(y1, ocf.Q, ocf.g, sec, yc)
    e1 = specific_energy(y1, ocf.Q, ocf.g, sec)
    e2 = specific_energy(y2, ocf.Q, ocf.g, sec)
    assert e1 == pytest.approx(e2, rel=1e-4)
    assert y2 < yc  # alternate depth is on the supercritical branch


# ---------------------------------------------------------------------
# gvf.py
# ---------------------------------------------------------------------

def test_friction_slope_equals_bed_slope_at_normal_depth():
    sec = Rectangular(B=4.0)
    ocf = OpenChannelFlow(section=sec, Q=20.0, n=0.025, S0=0.001)
    yn = ocf.normal_depth()
    sf = friction_slope(yn, ocf)
    assert sf == pytest.approx(ocf.S0, rel=1e-3)


def test_classify_curve_m1_backwater():
    ctype = classify_curve("mild", y_start=5.0, yn=2.0, yc=1.0)
    assert ctype.startswith("M1")


def test_classify_curve_s2_drawdown():
    ctype = classify_curve("steep", y_start=1.0, yn=0.5, yc=1.5)
    assert ctype.startswith("S2")


def test_solve_profile_mild_m1_backwater():
    sec = Trapezoidal(B=4.0, z=1.5)
    ocf = OpenChannelFlow(section=sec, Q=20.0, n=0.025, S0=0.001)
    yn, yc = ocf.normal_depth(), ocf.critical_depth()
    assert yn > yc  # mild channel
    y_start = yn * 1.3  # above normal depth -> M1 backwater
    result = solve_profile(ocf, y_start, L=500.0, dx=10.0, yn=yn, yc=yc)
    assert result.curve_type.startswith("M1")
    assert len(result.x) == len(result.y)
    assert len(result.x) >= 2


def test_solve_profile_does_not_cross_critical_depth():
    sec = Trapezoidal(B=4.0, z=1.5)
    ocf = OpenChannelFlow(section=sec, Q=20.0, n=0.025, S0=0.02)  # steep
    yn, yc = ocf.normal_depth(), ocf.critical_depth()
    assert yn < yc  # steep channel
    y_start = yc * 1.5  # S1 backwater, approaches yc from above
    result = solve_profile(ocf, y_start, L=500.0, dx=5.0, yn=yn, yc=yc)
    # every station depth should stay >= yc (curve should not overshoot through it)
    assert all(y >= yc - 1e-3 for y in result.y)


# ---------------------------------------------------------------------
# jump.py
# ---------------------------------------------------------------------

def test_sequent_depth_matches_belanger_rectangular():
    B, Q = 4.0, 20.0
    sec = Rectangular(B=B)
    y1 = 0.4  # a shallow, fast (supercritical) depth
    y2_general = sequent_depth(y1, Q, G, sec)
    y2_closed_form = belanger_rectangular(y1, Q, G, B)
    assert y2_general == pytest.approx(y2_closed_form, rel=1e-3)


def test_momentum_function_minimum_at_critical_depth():
    sec = Rectangular(B=4.0)
    ocf = OpenChannelFlow(section=sec, Q=20.0, n=0.025, S0=0.001)
    yc = ocf.critical_depth()
    m_at_c = momentum_function(yc, ocf.Q, ocf.g, sec)
    m_below = momentum_function(yc * 0.8, ocf.Q, ocf.g, sec)
    m_above = momentum_function(yc * 1.2, ocf.Q, ocf.g, sec)
    assert m_at_c < m_below
    assert m_at_c < m_above


def test_jump_energy_loss_is_positive():
    B, Q = 4.0, 20.0
    sec = Rectangular(B=B)
    y1 = 0.4
    y2 = sequent_depth(y1, Q, G, sec)
    dE = jump_energy_loss(y1, y2, Q, G, sec)
    assert dE > 0  # a real jump always dissipates energy


def test_locate_jump_case_a_steep_channel_with_downstream_control():
    sec = Trapezoidal(B=4.0, z=1.5)
    ocf = OpenChannelFlow(section=sec, Q=20.0, n=0.015, S0=0.02)  # steep
    yn, yc = ocf.normal_depth(), ocf.critical_depth()
    assert yn < yc

    # The downstream boundary must be at (or above) the sequent depth of
    # yn for a jump to be reachable within the reach -- pick a boundary
    # depth comfortably above that threshold so this is a well-posed,
    # guaranteed-jump scenario (this is what "pen and paper" checks
    # first before asking the solver to find it).
    y2_required = sequent_depth(yn, ocf.Q, ocf.g, sec)
    y_start = y2_required * 1.15

    result = solve_profile(ocf, y_start, L=800.0, dx=5.0, yn=yn, yc=yc)
    jump = locate_jump(result, ocf, yn, yc)

    assert jump is not None, "Jump should be found: boundary depth exceeds the required sequent depth"
    assert jump["y1"] == pytest.approx(yn, rel=1e-3)
    assert jump["y2"] == pytest.approx(y2_required, rel=1e-3)
    assert jump["y2"] > jump["y1"]
    assert jump["delta_E"] > 0
    assert 0 < jump["x_jump"] < 800.0


def test_locate_jump_case_b_mild_channel_with_sluice_gate():
    sec = Trapezoidal(B=4.0, z=1.5)
    ocf = OpenChannelFlow(section=sec, Q=20.0, n=0.025, S0=0.001)  # mild
    yn, yc = ocf.normal_depth(), ocf.critical_depth()
    assert yn > yc

    # A sluice gate releasing a shallow, fast (supercritical) depth into
    # a mild channel must produce an M3 curve rising toward a jump.
    y_start = yc * 0.3
    result = solve_profile(ocf, y_start, L=1000.0, dx=5.0, yn=yn, yc=yc)
    jump = locate_jump(result, ocf, yn, yc)

    assert jump is not None, "Jump should be found: M3 curve rises to meet the downstream normal depth"
    assert jump["y2"] == pytest.approx(yn, rel=1e-3)
    assert jump["y2"] > jump["y1"]
    assert jump["delta_E"] > 0


def test_locate_jump_returns_none_when_boundary_too_low():
    """If the downstream control depth is BELOW the required sequent
    depth, physics says no jump occurs within the reach -- the solver
    must not fabricate one."""
    sec = Trapezoidal(B=4.0, z=1.5)
    ocf = OpenChannelFlow(section=sec, Q=20.0, n=0.015, S0=0.02)  # steep
    yn, yc = ocf.normal_depth(), ocf.critical_depth()
    y2_required = sequent_depth(yn, ocf.Q, ocf.g, sec)
    # pick a boundary strictly between yc and the required sequent depth
    y_start = (yc + y2_required) / 2.0
    assert yc < y_start < y2_required  # a valid S1 boundary depth, just too low for a jump

    result = solve_profile(ocf, y_start, L=800.0, dx=5.0, yn=yn, yc=yc)
    jump = locate_jump(result, ocf, yn, yc)
    assert jump is None
