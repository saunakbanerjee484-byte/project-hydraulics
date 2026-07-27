"""
===============================================================================
HYDRAULIC JUMP MECHANICS & DISCONTINUOUS PROFILE SOLVER ENGINE
===============================================================================

PHYSICAL GOVERNING PRINCIPLES:
------------------------------
1. WAVE CELERITY & INFORMATION BARRIER (THE SHOCK TRIGGER):
   Small-amplitude surface gravity waves propagate at celerity c = sqrt(g * d_h).
   When bulk flow velocity V > c (Froude Fr > 1), flow is supercritical.
   In this regime, downstream boundary information cannot propagate upstream.
   When confronted by downstream subcritical flow or an obstacle, the fluid 
   undergoes a violent non-linear phase transition across a stationary hydraulic 
   shockwave -- THE HYDRAULIC JUMP.

2. COMPRESSIBLE FLOW ANALOGY (GAS DYNAMICS):
   The 1D Shallow Water Equations (Saint-Venant) are mathematically isomorphic 
   to the Euler equations for compressible gas dynamics (gamma = 2):
     - Water Depth (y)         <===>  Gas Density (rho)
     - Gravity Wave Speed (c)  <===>  Acoustic Speed of Sound (a)
     - Froude Number (Fr)      <===>  Mach Number (Ma)
     - Hydraulic Jump          <===>  Normal Shock Wave (Rankine-Hugoniot)

3. MOMENTUM CONSERVATION ACROSS DISCONTINUITY (BELANGER THEOREM):
   Due to extreme viscous shear, turbulent vortex breakdown, and air entrainment,
   mechanical energy is NOT conserved across the jump (dE > 0 loss).
   However, net linear impulse-momentum is conserved across the control volume:
   
       sum(F_ext) = Integral(rho * v * (v . n) dA)
       
   P_hydrostatic_1 + M_flux_1 = P_hydrostatic_2 + M_flux_2
   
   Specific Force / Momentum Function:
       M(y) = (Q^2 / (g * A)) + A * y_bar
       where A * y_bar is the first moment of area about the free surface.

4. TURBULENT DISSIPATION & RECIRCULATION ROLLER:
   Energy dissipation (dE) is governed by the energy cascade where large 3D
   vortices break down into micro-turbulent eddies (Kolmogorov length scale),
   converting organized kinetic head into thermal energy and acoustic noise:
       dE = (y_2 - y_1)^3 / (4 * y_1 * y_2)  [For Rectangular Sections]
===============================================================================
"""

from __future__ import annotations

import math
from typing import Optional, Dict, Any
from .core import ConvergenceError
from .geometry import CrossSection


def momentum_function(y: float, Q: float, g: float, section: CrossSection) -> float:
    """
    Computes the Specific Force / Momentum Function M(y):
    
        M(y) = Q^2 / (g * A(y)) + A(y) * y_bar(y)
        
    PHYSICAL INTERPRETATION:
    - Term 1: Momentum flux (dynamic thrust) passing through section per unit weight.
    - Term 2: Hydrostatic force resultant acting on cross-sectional area per unit weight.
    
    M(y) exhibits a global minimum at Critical Depth (y = y_c, Fr = 1.0),
    representing the minimum force required to pass a given discharge Q.
    """
    a = section.area(y)
    a_ybar = section.area_moment(y)  # First moment of cross-sectional area about surface
    return (Q ** 2) / (g * a) + a_ybar


def sequent_depth(y1: float, Q: float, g: float, section: CrossSection,
                  tol: float = 1e-9, max_iter: int = 200) -> float:
    """
    Finds the conjugate/sequent depth y2 across the hydraulic jump such that:
    
        M(y1) == M(y2)   where y2 != y1
        
    MECHANICS OF THE SOLVER:
    The Specific Force curve M(y) is convex-strict with two branches:
      - Supercritical branch (y < y_c, dM/dy < 0)
      - Subcritical branch   (y > y_c, dM/dy > 0)
      
    This function uses a local gradient probe (dM/dy) to identify the starting 
    branch of y1, constructs a rigorous bracket on the opposite side of y_c, 
    and converges via Bisection search to locate the unique physical conjugate root.
    """
    m_target = momentum_function(y1, Q, g, section)

    def f(y):
        return momentum_function(y, Q, g, section) - m_target

    # Probe local gradient dM/dy to determine flow regime (Supercritical vs Subcritical)
    step = max(y1 * 1e-4, 1e-6)
    slope = (momentum_function(y1 + step, Q, g, section) - momentum_function(y1 - step, Q, g, section))

    if slope < 0:
        # dM/dy < 0 ==> Supercritical regime (y1 < y_c).
        # Conjugate depth y2 lies strictly on the subcritical branch (y2 > y_c > y1).
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
        # dM/dy > 0 ==> Subcritical regime (y1 > y_c).
        # Conjugate depth y2 lies strictly on the supercritical branch (y2 < y_c < y1).
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
        raise ConvergenceError(
            f"Physical Boundary Failure: Could not bracket conjugate sequent depth for initial depth y1={y1:.4f}m."
        )

    # Rigorous Bisection Method over the bracketed domain
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) < tol or (hi - lo) < tol:
            return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
            
    raise ConvergenceError("Sequent-depth non-linear solver failed to converge within max iterations.")


def jump_energy_loss(y1: float, y2: float, Q: float, g: float, section: CrossSection) -> float:
    """
    Computes total energy head loss dE across the hydraulic jump:
    
        dE = E(y1) - E(y2) = (y1 + V1^2 / 2g) - (y2 + V2^2 / 2g) > 0
        
    PHYSICS OF ENERGY LOSS:
    In accordance with the 2nd Law of Thermodynamics, energy loss dE MUST be positive.
    The lost mechanical head is transformed into turbulence, kinetic energy of entrained
    air bubbles, noise, and micro-thermal energy.
    """
    from .energy import specific_energy
    e1 = specific_energy(y1, Q, g, section)
    e2 = specific_energy(y2, Q, g, section)
    return e1 - e2


def belanger_rectangular(y1: float, Q: float, g: float, B: float) -> float:
    """
    Closed-form Belanger Equation for RECTANGULAR CHANNELS:
    
        y2 / y1 = 0.5 * ( sqrt(1 + 8 * Fr1^2) - 1 )
        
    DERIVATION:
    Derived directly from integrating 1D hydrostatic pressure & momentum flux
    balance over a rectangular control volume without friction losses.
    Used for analytical benchmark verification against numerical root solver.
    """
    q = Q / B  # Discharge per unit width
    v1 = q / y1
    fr1 = v1 / math.sqrt(g * y1)
    return 0.5 * y1 * (math.sqrt(1.0 + 8.0 * (fr1 ** 2)) - 1.0)


def locate_jump(gvf_result, ocf, yn: float, yc: float) -> Optional[Dict[str, Any]]:
    """
    SCANS GRADUALLY VARIED FLOW (GVF) PROFILE FOR DISCONTINUOUS HYDRAULIC JUMPS.
    
    HYDRODYNAMICS & REGIME ANALYSIS:
    --------------------------------
    Case A: STEEP CHANNEL (yn < yc) -- Downstream Control
        - Flow naturally enters reach as supercritical uniform flow (yn).
        - Downstream obstacle/weir forces subcritical depth profile (S1 backwater).
        - Jump occurs where the S1 backwater depth equals the conjugate depth of yn.

    Case B: MILD CHANNEL (yn > yc) -- Upstream Control
        - Sluice gate or spillway releases supercritical flow into mild slope (S3/M3 curve).
        - Flow decelerates downstream as water depth increases.
        - Jump occurs where the local conjugate depth of the curve matches normal depth (yn).
    """
    Q, g, section = ocf.Q, ocf.g, ocf.section
    xs, ys = gvf_result.x, gvf_result.y
    if len(xs) < 2:
        return None

    channel_type = "mild" if yn > yc else ("steep" if yn < yc else "critical")
    if channel_type == "critical":
        return None

    if channel_type == "steep":
        try:
            # Target depth is the subcritical conjugate depth required to balance upstream yn momentum
            y2_target = sequent_depth(yn, Q, g, section)
        except ConvergenceError:
            return None
        diffs = [y - y2_target for y in ys]
    else:
        # Compute conjugate depth profile along the supercritical GVF curve
        seq_depths = []
        for y in ys:
            try:
                seq_depths.append(sequent_depth(y, Q, g, section))
            except ConvergenceError:
                seq_depths.append(float("nan"))
        diffs = [sd - yn for sd in seq_depths]

    # Detect hydrodynamic boundary crossing (Momentum balance point)
    for i in range(len(diffs) - 1):
        d0, d1 = diffs[i], diffs[i + 1]
        if d0 != d0 or d1 != d1:  # NaN Guard
            continue
        if d0 == 0:
            x_jump = xs[i]
            break
        if d0 * d1 < 0:
            # Linear spatial interpolation of jump shockwave position
            frac = d0 / (d0 - d1)
            x_jump = xs[i] + frac * (xs[i + 1] - xs[i])
            break
    else:
        return None

    if channel_type == "steep":
        y1 = yn
        y2 = y2_target
    else:
        # Interpolate supercritical depth y1 at exact spatial shock location x_jump
        y1 = ys[i] + (ys[i + 1] - ys[i]) * (
            0 if xs[i + 1] == xs[i] else (x_jump - xs[i]) / (xs[i + 1] - xs[i])
        )
        y2 = yn

    delta_e = jump_energy_loss(y1, y2, Q, g, section)
    
    # Calculate Upstream Froude Number & Hydrodynamic Jump Classification
    v1 = Q / section.area(y1)
    
    # SAFE FIX: Calculate Hydraulic Depth D_h = Area / Top_Width directly
    d_h = section.area(y1) / section.top_width(y1)
    fr1 = v1 / math.sqrt(g * d_h)
    
    if fr1 < 1.7:
        jump_type = "Undular Jump (Low energy dissipation, surface waves)"
    elif fr1 < 2.5:
        jump_type = "Weak Jump (Roller formation, uniform velocity downstream)"
    elif fr1 < 4.5:
        jump_type = "Oscillating Jump (Pulsating jet, wave action downstream)"
    elif fr1 < 9.0:
        jump_type = "Steady / Stable Jump (Extremely efficient energy dissipator, 45-70% loss)"
    else:
        jump_type = "Strong Jump (Violent shockwave, high air entrainment, >70% dissipation)"

    return {
        "x_jump": float(x_jump),
        "y1": float(y1),
        "y2": float(y2),
        "delta_E": float(delta_e),
        "Fr1": float(fr1),
        "jump_type": jump_type
    }