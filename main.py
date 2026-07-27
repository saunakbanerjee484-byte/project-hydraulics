"""
main.py
=======
CLI orchestrator: ties geometry, core hydraulics, specific energy, GVF,
hydraulic-jump, control-structure, and I/O/visualization modules into one
interactive run (class OpenChannelFlow is the OOP engine at the center).
"""
import os
import sys
import dataclasses

from open_channel_flow.geometry import Rectangular, Triangular, Trapezoidal, Circular, GeometryError
from open_channel_flow.core import OpenChannelFlow, ConvergenceError
from open_channel_flow.energy import energy_curve
from open_channel_flow.gvf import solve_profile, stitch_uniform_flow
from open_channel_flow.jump import locate_jump
from open_channel_flow.control_structures import sluice_gate_depth, weir_downstream_depth
from open_channel_flow.io_utils import build_station_table, tabular_report, export_csv, export_json, export_excel
from open_channel_flow.visualization import build_dashboard

import matplotlib
matplotlib.use("Agg")  # safe default for headless environments; main() switches to interactive show if a display exists
import matplotlib.pyplot as plt


def ask(prompt, default, cast=float):
    raw = input(f"{prompt} [default = {default}]: ").strip()
    return cast(raw) if raw else default


def build_section():
    print("\nChannel shape: 1) Rectangular  2) Triangular  3) Trapezoidal  4) Circular")
    choice = ask("Choose shape number", 3, int)
    if choice == 1:
        B = ask("Bottom width B (m)", 4.0)
        return Rectangular(B=B)
    elif choice == 2:
        z = ask("Side slope z (H:V = z:1)", 1.0)
        return Triangular(z=z)
    elif choice == 4:
        D = ask("Pipe diameter D (m)", 1.5)
        return Circular(D=D)
    else:
        B = ask("Bottom width B (m)", 4.0)
        z = ask("Side slope z (H:V = z:1)", 1.5)
        return Trapezoidal(B=B, z=z)


def get_boundary_depth(ocf):
    print("\nGVF boundary condition source:")
    print("  1) Direct depth entry  2) Sluice gate  3) Sharp-crested weir")
    choice = ask("Choose option", 1, int)
    if choice == 2:
        a = ask("Gate opening (m)", 0.3)
        Cc = ask("Contraction coefficient Cc", 0.61)
        return sluice_gate_depth(a, Cc)
    elif choice == 3:
        default_L = ocf.section.B if hasattr(ocf.section, "B") else 4.0
        L = ask("Weir length (m)", default_L)
        P = ask("Weir crest height above bed (m)", 1.0)
        Cw = ask("Weir coefficient Cw", 1.84)
        return weir_downstream_depth(ocf.Q, L, P, Cw)
    else:
        return ask("Known depth y_start (m)", 3.0)


def main():
    print("=" * 70)
    print(" OPEN CHANNEL FLOW ANALYSIS SUITE (OOP Edition)")
    print("=" * 70)

    section = build_section()
    Q = ask("Discharge Q (m^3/s)", 20.0)
    n = ask("Manning's n", 0.025)
    S0 = ask("Bed slope S0 (m/m)", 0.001)
    g = ask("g (m/s^2)", 9.81)

    try:
        ocf = OpenChannelFlow(section=section, Q=Q, n=n, S0=S0, g=g)
        yn = ocf.normal_depth()
        yc = ocf.critical_depth()
    except (GeometryError, ConvergenceError) as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    Fr_n = ocf.froude(yn)
    print("\n" + "=" * 70)
    print(" UNIFORM / CRITICAL FLOW RESULTS")
    print("=" * 70)
    print(f" Normal depth   yn = {yn:.4f} m")
    print(f" Critical depth yc = {yc:.4f} m")
    print(f" Froude @ yn        = {Fr_n:.4f}  ({ocf.regime(yn)})")
    print(f" Channel classification: {ocf.classify_channel()}")

    y_start = get_boundary_depth(ocf)
    L = ask("Channel stretch length L (m)", 1000.0)
    dx = ask("Step size dx (m)", 10.0)

    gvf_result = solve_profile(ocf, y_start, L, dx, yn=yn, yc=yc)
    print("\n" + "=" * 70)
    print(" GVF PROFILE RESULTS")
    print("=" * 70)
    print(f" {gvf_result.regime}")
    print(f" Curve type: {gvf_result.curve_type}")

    jump = locate_jump(gvf_result, ocf, yn, yc)
    if jump:
        print("\n" + "-" * 70)
        print(" HYDRAULIC JUMP DETECTED")
        print("-" * 70)
        print(f" Location x         = {jump['x_jump']:.2f} m")
        print(f" Pre-jump depth  y1 = {jump['y1']:.4f} m")
        print(f" Post-jump depth y2 = {jump['y2']:.4f} m")
        print(f" Energy dissipated  = {jump['delta_E']:.4f} m")
    else:
        print("\n No hydraulic jump implied by this profile/channel combination.")

    # Stitch in the uniform normal-depth segment on the far side of a jump
    # (if any) so reporting/export/plotting cover the full 0 -> L reach.
    x_full, y_full = stitch_uniform_flow(gvf_result, yn, jump)
    gvf_result = dataclasses.replace(gvf_result, x=x_full, y=y_full)

    rows = build_station_table(ocf, gvf_result)
    print("\nStation table (first 10 rows):")
    tabular_report(rows[:10], headers=("x_m", "y_m", "V_mps", "Fr", "Sf", "E_m"))

    out_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = export_csv(rows, os.path.join(out_dir, "gvf_results.csv"))
    json_path = export_json({
        "yn": yn, "yc": yc, "regime": gvf_result.regime, "curve_type": gvf_result.curve_type,
        "jump": jump
    }, os.path.join(out_dir, "gvf_summary.json"))
    try:
        xlsx_path = export_excel(rows, os.path.join(out_dir, "gvf_results.xlsx"))
        print(f"\n Exported: {csv_path}\n Exported: {json_path}\n Exported: {xlsx_path}")
    except ImportError:
        print(f"\n Exported: {csv_path}\n Exported: {json_path}\n (Excel export needs pandas+openpyxl)")

    y_curve, E_curve, e_min_point = energy_curve(Q, g, section, yn, yc)
    fig = build_dashboard(ocf, yn, yc, y_curve, E_curve, e_min_point, gvf_result, jump)
    png_path = os.path.join(out_dir, "gvf_dashboard.png")
    fig.savefig(png_path, dpi=150)
    print(f" Dashboard saved to: {png_path}")


if __name__ == "__main__":
    main()
