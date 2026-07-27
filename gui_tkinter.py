"""
gui_tkinter.py
==============
Tkinter GUI Control Dashboard.

A lightweight desktop front-end for the OpenChannelFlow engine: enter
geometry/flow parameters, click "Run Analysis", and see results plus an
embedded Matplotlib dashboard in the same window.

Run with:  python gui_tkinter.py
(Requires a display -- this will not run in a headless environment.)
"""
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from open_channel_flow.geometry import Trapezoidal, GeometryError
from open_channel_flow.core import OpenChannelFlow, ConvergenceError
from open_channel_flow.energy import energy_curve
from open_channel_flow.gvf import solve_profile
from open_channel_flow.jump import locate_jump
from open_channel_flow.visualization import build_dashboard


DEFAULTS = dict(B=4.0, z=1.5, S0=0.001, Q=20.0, n=0.025, g=9.81,
                y_start=3.0, L=1000.0, dx=10.0)


class OCFApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Open Channel Flow -- GVF Analysis")
        self.geometry("1200x700")
        self.entries = {}
        self.canvas = None
        self._build_form()
        self._build_output()

    def _build_form(self):
        frame = ttk.Frame(self, padding=10)
        frame.grid(row=0, column=0, sticky="ns")
        ttk.Label(frame, text="Trapezoidal Channel Inputs", font=("Segoe UI", 11, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(0, 8))
        for i, (key, val) in enumerate(DEFAULTS.items(), start=1):
            ttk.Label(frame, text=key).grid(row=i, column=0, sticky="w", pady=2)
            e = ttk.Entry(frame, width=12)
            e.insert(0, str(val))
            e.grid(row=i, column=1, pady=2)
            self.entries[key] = e
        ttk.Button(frame, text="Run Analysis", command=self.run).grid(
            row=len(DEFAULTS) + 1, column=0, columnspan=2, pady=10)
        self.result_text = tk.Text(frame, width=34, height=18, font=("Consolas", 9))
        self.result_text.grid(row=len(DEFAULTS) + 2, column=0, columnspan=2)

    def _build_output(self):
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.grid(row=0, column=1, sticky="nsew")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

    def run(self):
        try:
            v = {k: float(self.entries[k].get()) for k in DEFAULTS}
            section = Trapezoidal(B=v["B"], z=v["z"])
            ocf = OpenChannelFlow(section=section, Q=v["Q"], n=v["n"], S0=v["S0"], g=v["g"])
            yn, yc = ocf.normal_depth(), ocf.critical_depth()
            gvf_result = solve_profile(ocf, v["y_start"], v["L"], v["dx"], yn=yn, yc=yc)
            jump = locate_jump(gvf_result, ocf, yn, yc)
            y_curve, E_curve, e_min_point = energy_curve(v["Q"], v["g"], section, yn, yc)
            fig = build_dashboard(ocf, yn, yc, y_curve, E_curve, e_min_point, gvf_result, jump)

            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, f"yn = {yn:.4f} m\nyc = {yc:.4f} m\n")
            self.result_text.insert(tk.END, f"Classification: {ocf.classify_channel()}\n")
            self.result_text.insert(tk.END, f"Curve type: {gvf_result.curve_type}\n")
            if jump:
                self.result_text.insert(tk.END, f"\nJump at x={jump['x_jump']:.1f} m\n"
                                                  f"y1={jump['y1']:.3f} -> y2={jump['y2']:.3f}\n"
                                                  f"dE = {jump['delta_E']:.3f} m\n")
            else:
                self.result_text.insert(tk.END, "\nNo hydraulic jump.\n")

            if self.canvas:
                self.canvas.get_tk_widget().destroy()
            self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
        except (GeometryError, ConvergenceError, ValueError) as e:
            messagebox.showerror("Input error", str(e))


if __name__ == "__main__":
    OCFApp().mainloop()
