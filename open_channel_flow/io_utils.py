"""
io_utils.py
===========
Station-table construction, plain-text tabular reports, and CSV/JSON/
Excel export helpers.
"""
from __future__ import annotations

import csv
import json


def build_station_table(ocf, gvf_result):
    """
    Build a list of row-dicts (x, y, V, Fr, Sf, E) for every station in
    a solved GVF profile.
    """
    from .gvf import friction_slope
    from .energy import specific_energy

    rows = []
    for x, y in zip(gvf_result.x, gvf_result.y):
        v = ocf.velocity(y)
        fr = ocf.froude(y)
        sf = friction_slope(y, ocf)
        e = specific_energy(y, ocf.Q, ocf.g, ocf.section)
        rows.append({"x_m": x, "y_m": y, "V_mps": v, "Fr": fr, "Sf": sf, "E_m": e})
    return rows


def tabular_report(rows, headers=("x_m", "y_m", "V_mps", "Fr", "Sf", "E_m")):
    """Print a simple fixed-width text table of station rows."""
    widths = {h: max(len(h), 10) for h in headers}
    header_line = " | ".join(h.rjust(widths[h]) for h in headers)
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        line = " | ".join(f"{row[h]:.4f}".rjust(widths[h]) if isinstance(row[h], float)
                           else str(row[h]).rjust(widths[h]) for h in headers)
        print(line)


def export_csv(rows, path):
    if not rows:
        raise ValueError("No rows to export.")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def export_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=lambda o: None)
    return path


def export_excel(rows, path):
    """Export station rows to an .xlsx file. Requires pandas + openpyxl."""
    import pandas as pd  # noqa: import here so the module works without pandas installed
    df = pd.DataFrame(rows)
    df.to_excel(path, index=False)
    return path
