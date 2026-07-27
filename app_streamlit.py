"""
app_streamlit.py
=================
Interactive web dashboard built on Streamlit + Plotly, sharing the exact
same physics engine as main.py / gui_tkinter.py.

Run with:  streamlit run app_streamlit.py
"""
import dataclasses

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from open_channel_flow.geometry import Rectangular, Triangular, Trapezoidal, Circular, GeometryError
from open_channel_flow.core import OpenChannelFlow, ConvergenceError
from open_channel_flow.energy import energy_curve, specific_energy
from open_channel_flow.gvf import solve_profile, stitch_uniform_flow
from open_channel_flow.jump import locate_jump
from open_channel_flow.control_structures import sluice_gate_depth, weir_downstream_depth
from open_channel_flow.io_utils import build_station_table

# ---------------------------------------------------------------------------
# Page config + theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Open Channel Flow Analysis",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#0EA5E9"      # sky blue
ACCENT = "#F97316"       # amber
DEEP = "#0B1F33"         # deep navy (bed / dark text)
MILD_COLOR = "#22C55E"   # green
STEEP_COLOR = "#EF4444"  # red
SUB_COLOR = "#0EA5E9"    # blue
SUPER_COLOR = "#F97316"  # amber

st.markdown(f"""
<style>
    .stApp {{
        background: radial-gradient(circle at 10% 0%, #eef6ff 0%, #f7fafc 40%, #ffffff 100%);
    }}
    .ocf-hero {{
        padding: 1.6rem 2rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #0B1F33 0%, #0F3D5C 45%, #0EA5E9 100%);
        color: white;
        margin-bottom: 1.4rem;
        box-shadow: 0 10px 30px -12px rgba(14,165,233,0.55);
    }}
    .ocf-hero h1 {{
        font-size: 1.9rem;
        margin: 0 0 0.2rem 0;
        letter-spacing: -0.02em;
    }}
    .ocf-hero p {{
        margin: 0;
        opacity: 0.85;
        font-size: 0.95rem;
    }}
    .ocf-badge {{
        display: inline-block;
        padding: 0.28rem 0.85rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.8rem;
        margin-right: 0.4rem;
        color: white;
    }}
    div[data-testid="stMetric"] {{
        background: white;
        border: 1px solid #e6edf3;
        border-radius: 14px;
        padding: 0.9rem 1rem 0.6rem 1rem;
        box-shadow: 0 4px 14px -6px rgba(15,61,92,0.12);
    }}
    div[data-testid="stMetricLabel"] {{
        font-weight: 600;
        color: #64748b;
    }}
    /* Light, high-contrast sidebar -- forcing white text on every native
       widget (old dark-navy theme) made slider values, number inputs,
       and open dropdown text unreadable. This keeps Streamlit's normal
       widget contrast intact and only themes the surrounding chrome. */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #EAF4FB 0%, #F7FBFE 55%, #FFFFFF 100%);
        border-right: 1px solid #d7e6f0;
    }}
    section[data-testid="stSidebar"] h3 {{
        color: #0B1F33;
        font-weight: 800;
        border-bottom: 2px solid #0EA5E9;
        padding-bottom: 0.3rem;
        margin-top: 1.1rem;
    }}
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {{
        color: #0F3D5C !important;
        font-weight: 600;
    }}
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
        background-color: #ffffff;
        border: 1px solid #b9dcf0;
    }}
    section[data-testid="stSidebar"] [data-testid="stNumberInput"] input {{
        background-color: #ffffff;
        color: #0B1F33;
    }}
    section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {{
        padding-top: 0.2rem;
    }}
    .ocf-jump-card {{
        background: linear-gradient(120deg, #FEF3C7 0%, #FEE2E2 100%);
        border-left: 5px solid #EF4444;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-top: 0.5rem;
    }}
    .ocf-nojump-card {{
        background: linear-gradient(120deg, #ECFDF5 0%, #EFF6FF 100%);
        border-left: 5px solid #22C55E;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-top: 0.5rem;
    }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar: inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌊 Channel Geometry")
    shape = st.selectbox("Cross-section shape", ["Trapezoidal", "Rectangular", "Triangular", "Circular"])

    if shape == "Rectangular":
        B = st.slider("Bottom width B (m)", 0.5, 20.0, 4.0, 0.1)
        section_factory = lambda: Rectangular(B=B)
    elif shape == "Triangular":
        z = st.slider("Side slope z (H:V)", 0.1, 4.0, 1.5, 0.1)
        section_factory = lambda: Triangular(z=z)
    elif shape == "Circular":
        D = st.slider("Pipe diameter D (m)", 0.3, 5.0, 1.5, 0.1)
        section_factory = lambda: Circular(D=D)
    else:
        B = st.slider("Bottom width B (m)", 0.5, 20.0, 4.0, 0.1)
        z = st.slider("Side slope z (H:V)", 0.0, 4.0, 1.5, 0.1)
        section_factory = lambda: Trapezoidal(B=B, z=z)

    st.markdown("### 💧 Flow Parameters")
    S0 = st.number_input("Bed slope S0 (m/m)", 0.0001, 0.05, 0.001, format="%.4f")
    Q = st.slider("Discharge Q (m³/s)", 1.0, 200.0, 20.0, 1.0)
    n = st.slider("Manning's n", 0.010, 0.060, 0.025, 0.001)

    st.markdown("### 🚧 GVF Boundary")
    boundary_mode = st.radio("Boundary depth source", ["Direct entry", "Sluice gate", "Sharp-crested weir"])
    if boundary_mode == "Sluice gate":
        a = st.slider("Gate opening a (m)", 0.05, 2.0, 0.3, 0.05)
        Cc = st.slider("Contraction coefficient Cc", 0.4, 1.0, 0.61, 0.01)
    elif boundary_mode == "Sharp-crested weir":
        L_weir = st.slider("Weir length (m)", 0.5, 20.0, 4.0, 0.5)
        P_weir = st.slider("Weir crest height P (m)", 0.1, 5.0, 1.0, 0.1)
        Cw = st.slider("Weir coefficient Cw", 1.4, 2.2, 1.84, 0.01)
    else:
        y_start_direct = st.slider("Boundary depth y_start (m)", 0.1, 10.0, 3.0, 0.1)

    st.markdown("### 📏 Reach")
    L = st.number_input("Reach length L (m)", 10.0, 20000.0, 1000.0)
    dx = st.number_input("Step size dx (m)", 1.0, 100.0, 10.0)

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="ocf-hero">
    <h1>🌊 Open Channel Flow — Interactive GVF Analysis</h1>
    <p>Uniform flow · Critical flow · Specific energy · Gradually varied flow · Hydraulic jumps</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Core solve
# ---------------------------------------------------------------------------
try:
    section = section_factory()
    ocf = OpenChannelFlow(section=section, Q=Q, n=n, S0=S0)
    yn, yc = ocf.normal_depth(), ocf.critical_depth()
except (GeometryError, ConvergenceError) as e:
    st.error(f"⚠️ {e}")
    st.stop()

try:
    if boundary_mode == "Sluice gate":
        y_start = sluice_gate_depth(a, Cc)
    elif boundary_mode == "Sharp-crested weir":
        y_start = weir_downstream_depth(Q, L_weir, P_weir, Cw)
    else:
        y_start = y_start_direct
except GeometryError as e:
    st.error(f"⚠️ {e}")
    st.stop()

classification = ocf.classify_channel()
class_color = MILD_COLOR if classification == "mild" else (STEEP_COLOR if classification == "steep" else "#94A3B8")
fr_n = ocf.froude(yn)
regime_n = ocf.regime(yn)
regime_color = SUB_COLOR if regime_n == "subcritical" else (SUPER_COLOR if regime_n == "supercritical" else "#94A3B8")

st.markdown(
    f'<span class="ocf-badge" style="background:{class_color}">Channel: {classification.upper()}</span>'
    f'<span class="ocf-badge" style="background:{regime_color}">Flow @ yn: {regime_n.upper()}</span>'
    f'<span class="ocf-badge" style="background:#334155">Boundary depth y₀ = {y_start:.3f} m</span>',
    unsafe_allow_html=True,
)
st.write("")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Normal depth yₙ", f"{yn:.3f} m")
col2.metric("Critical depth y꜀", f"{yc:.3f} m")
col3.metric("Froude @ yₙ", f"{fr_n:.3f}")
col4.metric("Classification", classification.capitalize())

try:
    gvf_result = solve_profile(ocf, y_start, L, dx, yn=yn, yc=yc)
except (ConvergenceError, ValueError) as e:
    st.error(f"⚠️ {e}")
    st.stop()

jump = locate_jump(gvf_result, ocf, yn, yc)
x_full, y_full = stitch_uniform_flow(gvf_result, yn, jump)
gvf_result = dataclasses.replace(gvf_result, x=x_full, y=y_full)

st.write("")
st.markdown(f"#### 📈 Curve type: `{gvf_result.curve_type}`")
st.caption(gvf_result.regime)

y_curve, E_curve, e_min_point = energy_curve(Q, ocf.g, section, yn, yc)

# ---------------------------------------------------------------------------
# Tabs: charts + data
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Energy & Profile", "🧮 Station Data", "ℹ️ Details"])

PLOTLY_TEMPLATE = "plotly_white"

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(
            x=E_curve, y=y_curve, mode="lines", name="E–y curve",
            line=dict(color=PRIMARY, width=3),
            fill="tozerox", fillcolor="rgba(14,165,233,0.08)",
        ))
        fig_e.add_trace(go.Scatter(
            x=[e_min_point[0]], y=[yc], mode="markers+text", name=f"Critical y꜀={yc:.3f} m",
            marker=dict(color=STEEP_COLOR, size=12, line=dict(color="white", width=1.5)),
            text=["y꜀"], textposition="top center",
        ))
        E_at_yn = specific_energy(yn, Q, ocf.g, section)
        fig_e.add_trace(go.Scatter(
            x=[E_at_yn], y=[yn], mode="markers+text", name=f"Normal yₙ={yn:.3f} m",
            marker=dict(color=ACCENT, size=12, line=dict(color="white", width=1.5)),
            text=["yₙ"], textposition="top center",
        ))
        fig_e.update_layout(
            title="Specific Energy Diagram", xaxis_title="E (m)", yaxis_title="y (m)",
            template=PLOTLY_TEMPLATE, hovermode="closest",
            legend=dict(orientation="h", y=-0.2), margin=dict(t=50, b=10),
        )
        st.plotly_chart(fig_e, use_container_width=True)

    with c2:
        x_line = np.linspace(0, L, 200)
        bed = S0 * x_line
        surface = [S0 * xv + yv for xv, yv in zip(gvf_result.x, gvf_result.y)]

        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=x_line, y=bed, mode="lines", name="Bed",
                                    line=dict(color="#92400E", width=2)))
        fig_p.add_trace(go.Scatter(x=x_line, y=bed + yn, mode="lines", name="Normal depth line",
                                    line=dict(color="#334155", dash="dot", width=1.5)))
        fig_p.add_trace(go.Scatter(x=x_line, y=bed + yc, mode="lines", name="Critical depth line",
                                    line=dict(color="#334155", dash="dash", width=1.5)))
        fig_p.add_trace(go.Scatter(
            x=gvf_result.x, y=surface, mode="lines", name="Water surface",
            line=dict(color=PRIMARY, width=3.5),
            fill="tonexty", fillcolor="rgba(14,165,233,0.18)",
        ))
        if jump:
            fig_p.add_vline(x=jump["x_jump"], line=dict(color=STEEP_COLOR, dash="dashdot", width=2),
                             annotation_text="⚡ Hydraulic jump", annotation_font_color=STEEP_COLOR)
        fig_p.update_layout(
            title="Water Surface Profile", xaxis_title="x (m)", yaxis_title="Elevation (m)",
            template=PLOTLY_TEMPLATE, hovermode="x unified",
            legend=dict(orientation="h", y=-0.2), margin=dict(t=50, b=10),
        )
        st.plotly_chart(fig_p, use_container_width=True)

    if jump:
        st.markdown(f"""
        <div class="ocf-jump-card">
        <b>⚡ Hydraulic jump detected</b> at x = {jump['x_jump']:.1f} m<br>
        y₁ = {jump['y1']:.3f} m &nbsp;→&nbsp; y₂ = {jump['y2']:.3f} m &nbsp;|&nbsp;
        Energy dissipated: <b>{jump['delta_E']:.3f} m</b>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="ocf-nojump-card">
        ✅ No hydraulic jump implied by this profile / channel combination.
        </div>
        """, unsafe_allow_html=True)

with tab2:
    rows = build_station_table(ocf, gvf_result)
    df = pd.DataFrame(rows, columns=["x_m", "y_m", "V_mps", "Fr", "Sf", "E_m"])
    st.dataframe(
        df.style.background_gradient(subset=["Fr"], cmap="RdYlBu_r").format(precision=4),
        use_container_width=True, height=420,
    )
    st.download_button("⬇️ Download CSV", df.to_csv(index=False), file_name="gvf_results.csv")

with tab3:
    st.markdown(f"""
    - **Section:** `{section!r}`
    - **Discharge Q:** {Q:.2f} m³/s
    - **Manning's n:** {n:.4f}
    - **Bed slope S0:** {S0:.4f}
    - **Boundary source:** {boundary_mode} → y₀ = {y_start:.3f} m
    - **Reach length L:** {L:.0f} m, step dx = {dx:.0f} m
    - **g:** {ocf.g} m/s²
    """)
