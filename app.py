# app.py
"""
Shelf Life Prediction Application

A comprehensive tool for predicting product shelf life using kinetic modeling,
Arrhenius equations, Q10 factors, and dynamic temperature simulations.
"""

import math
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st

from arrhenius_ui import arrhenius_fitting_tab
from dynamic_simulation import ensemble_dynamic_simulation_tab
from landing_page import landing_page_tab


def apply_app_style(max_width: int = 1200) -> None:
    """Inject enhanced app-wide CSS for modern design, better alignment and spacing."""
    css = f"""
    <style>
    /* ========== Layout & Spacing ========== */
    .block-container{{
        max-width:{max_width}px; 
        padding-top:2rem; 
        padding-left:2rem; 
        padding-right:2rem;
        padding-bottom:2rem;
    }}
    
    /* ========== Typography ========== */
    h1 {{
        font-family: 'Segoe UI', 'Inter', Roboto, Arial, sans-serif;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #2E7D32 !important;
        margin-bottom: 1.5rem !important;
        letter-spacing: -0.5px;
    }}
    
    h2 {{
        font-family: 'Segoe UI', 'Inter', Roboto, Arial, sans-serif;
        font-size: 2rem !important;
        font-weight: 600 !important;
        color: #1B5E20 !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }}
    
    h3 {{
        font-family: 'Segoe UI', 'Inter', Roboto, Arial, sans-serif;
        font-size: 1.5rem !important;
        font-weight: 500 !important;
        color: #2E7D32 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.75rem !important;
    }}
    
    p, .stMarkdown {{
        line-height: 1.6 !important;
    }}
    
    /* ========== Buttons ========== */
    .stButton>button {{
        padding: 0.75rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        border: none !important;
        background: linear-gradient(135deg, #2E7D32 0%, #388E3C 100%) !important;
        color: white !important;
        box-shadow: 0 2px 8px rgba(46, 125, 50, 0.3) !important;
        transition: all 0.3s ease !important;
    }}
    
    .stButton>button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.4) !important;
        background: linear-gradient(135deg, #388E3C 0%, #43A047 100%) !important;
    }}
    
    .stButton>button:active {{
        transform: translateY(0px) !important;
    }}
    
    /* ========== Metric Cards ========== */
    [data-testid="stMetric"] {{
        background: linear-gradient(135deg, #F5F7FA 0%, #FFFFFF 100%);
        padding: 1.25rem !important;
        border-radius: 12px !important;
        border: 1px solid #E0E0E0 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
        transition: all 0.3s ease !important;
    }}
    
    [data-testid="stMetric"]:hover {{
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12) !important;
        transform: translateY(-2px) !important;
    }}
    
    [data-testid="stMetricValue"] {{
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: #2E7D32 !important;
    }}
    
    [data-testid="stMetricLabel"] {{
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        color: #666 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    /* ========== Input Fields ========== */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {{
        border-radius: 8px !important;
        border: 2px solid #E0E0E0 !important;
        padding: 0.75rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }}
    
    .stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus {{
        border-color: #2E7D32 !important;
        box-shadow: 0 0 0 3px rgba(46, 125, 50, 0.1) !important;
    }}
    
    /* ========== Select Boxes ========== */
    .stSelectbox>div>div {{
        border-radius: 8px !important;
        border: 2px solid #E0E0E0 !important;
    }}
    
    /* ========== Expanders ========== */
    .streamlit-expanderHeader {{
        background-color: #F5F7FA !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        padding: 1rem !important;
        border: 1px solid #E0E0E0 !important;
    }}
    
    .streamlit-expanderHeader:hover {{
        background-color: #E8F5E9 !important;
        border-color: #2E7D32 !important;
    }}
    
    /* ========== Tabs ========== */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        background-color: #F5F7FA;
        padding: 0.5rem;
        border-radius: 10px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        height: auto;
        min-height: 45px;
        padding: 0.5rem 1rem;
        background-color: transparent;
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        white-space: normal;
        word-wrap: break-word;
        text-align: center;
        line-height: 1.3;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: #E8F5E9;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, #2E7D32 0%, #388E3C 100%) !important;
        color: white !important;
        font-weight: 600 !important;
    }}
    
    /* ========== Data Tables ========== */
    .stDataFrame {{
        border-radius: 8px !important;
        overflow: hidden !important;
        border: 1px solid #E0E0E0 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
    }}
    
    /* ========== Charts ========== */
    .element-container iframe, .stPlotlyChart, .stDataFrame {{
        width: 100% !important;
    }}
    
    .js-plotly-plot {{
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
    }}
    
    /* ========== Dividers ========== */
    hr {{
        margin: 2rem 0 !important;
        border: none !important;
        height: 2px !important;
        background: linear-gradient(90deg, transparent, #E0E0E0, transparent) !important;
    }}
    
    /* ========== Info/Success/Warning Boxes ========== */
    .stAlert {{
        border-radius: 8px !important;
        border-left: 4px solid !important;
        padding: 1rem 1.25rem !important;
    }}
    
    /* ========== Form Containers ========== */
    .stForm {{
        border: 1px solid #E0E0E0 !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        background-color: #FAFAFA !important;
    }}
    
    /* ========== Download Buttons ========== */
    .stDownloadButton>button {{
        background: linear-gradient(135deg, #FF6F00 0%, #FF8F00 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(255, 111, 0, 0.3) !important;
        transition: all 0.3s ease !important;
    }}
    
    .stDownloadButton>button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(255, 111, 0, 0.4) !important;
        background: linear-gradient(135deg, #FF8F00 0%, #FFA726 100%) !important;
    }}
    
    /* ========== Table and Data Editor Styling ========== */
    /* Data Editor Height Control */
    [data-testid="stDataFrameResizable"] {{
        min-height: 150px !important;
        max-height: 400px !important;
    }}
    
    /* Table Container */
    .stDataFrame {{
        font-size: 0.95rem !important;
    }}
    
    /* Table Headers */
    .stDataFrame thead tr th {{
        background: linear-gradient(135deg, #F5F7FA 0%, #E8F5E9 100%) !important;
        font-weight: 600 !important;
        padding: 0.75rem !important;
        color: #2E7D32 !important;
        border-bottom: 2px solid #2E7D32 !important;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }}
    
    /* Table Cells */
    .stDataFrame tbody tr td {{
        padding: 0.6rem 0.75rem !important;
        border-bottom: 1px solid #E0E0E0 !important;
    }}
    
    /* Alternating Row Colors */
    .stDataFrame tbody tr:nth-child(even) {{
        background-color: #FAFBFC !important;
    }}
    
    .stDataFrame tbody tr:hover {{
        background-color: #E8F5E9 !important;
        transition: background-color 0.2s ease;
    }}
    
    /* Data Editor Specific */
    [data-testid="data-editor"] {{
        border: 1px solid #E0E0E0 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }}
    
    /* Column Headers in Data Editor */
    [data-testid="data-editor"] [data-testid="column-header"] {{
        background-color: #F5F7FA !important;
        font-weight: 600 !important;
        color: #2E7D32 !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ============================================================
#  UTILITY — ISOTHERMAL KINETIC SHELF-LIFE (DIRECT)
# ============================================================

def compute_shelf_life_isothermal(
    A0: float,
    Ai: float,
    k: float,
    order: int,
    increasing: bool,
) -> Tuple[Optional[float], str]:
    """Compute isothermal shelf life using kinetic equations."""
    if k <= 0:
        return None, "Rate constant k must be > 0."

    try:
        if order == 0:
            # A = A0 ± k t
            if increasing:
                if Ai <= A0:
                    return None, "Ai must be > A0 for increasing 0th order."
                t = (Ai - A0) / k
            else:
                if Ai >= A0:
                    return None, "Ai must be < A0 for decreasing 0th order."
                t = (A0 - Ai) / k

        elif order == 1:
            # A = A0 · exp(±k t)
            if increasing:
                if Ai <= A0:
                    return None, "Ai must be > A0 for increasing 1st order."
                t = (1.0 / k) * math.log(Ai / A0)
            else:
                if Ai >= A0:
                    return None, "Ai must be < A0 for decreasing 1st order."
                t = (1.0 / k) * math.log(A0 / Ai)

        elif order == 2:
            # 1/A = 1/A0 ∓ k t
            if A0 <= 0 or Ai <= 0:
                return None, "A0 and Ai must be > 0 for 2nd order."
            if increasing:
                if Ai <= A0:
                    return None, "Ai must be > A0 for increasing 2nd order."
                t = (1.0 / A0 - 1.0 / Ai) / k
            else:
                if Ai >= A0:
                    return None, "Ai must be < A0 for decreasing 2nd order."
                t = (1.0 / Ai - 1.0 / A0) / k
        else:
            return None, "Unsupported reaction order."

        if t < 0:
            return None, "Calculated shelf life is negative — check inputs."

        return t, "OK"

    except Exception as e:
        return None, f"Error in calculation: {e}"


# ============================================================
#  TAB 1 — KINETIC FITTING AT CONSTANT TEMPERATURE
# ============================================================

def kinetic_analysis_constant_temp_tab():
    st.header("📊 Analyze Product Degradation Data")
    st.caption("Fit kinetic models to your experimental data and determine the reaction order")
    
    with st.expander("ℹ️ **How This Works** — Click to learn more", expanded=False):
        st.markdown(
            """
            ### Kinetic Reaction Modeling
            
            The kinetic approach predicts shelf life by modeling how a product attribute 
            (e.g., nutrient content, quality index) changes over time at constant temperature.
            
            **Governing Equation:**
            $$\\frac{d[A]}{dt} = K [A]^n$$
            
            where:
            - $K$ = kinetic rate constant
            - $t$ = time
            - $n$ = reaction order (0, 1, or 2)
            - $[A]$ = concentration or quality index
            
            **Reaction Orders:**
            
            **0th Order** (linear change):
            - Model: $A = A_0 \\pm Kt$
            - Best for: Degradation at high concentrations
            - Shelf life: $t = \\frac{A_0 - A_i}{K}$
            
            **1st Order** (exponential change):
            - Model: $\\ln A = \\ln A_0 - kt$
            - Best for: Most nutrient losses, color changes
            - Shelf life: $t = \\frac{1}{k}\\ln\\frac{A_0}{A_i}$
            
            **2nd Order** (concentration-dependent):
            - Model: $\\frac{1}{A} = \\frac{1}{A_0} + kt$
            - Best for: Enzymatic reactions
            - Shelf life: $t = \\frac{1/A_i - 1/A_0}{k}$
            
            **Tip:** Always check which transform ($A$, $\\ln A$, or $1/A$) gives the best R² value!
            """
        )

    # Example datasets selector
    st.markdown("### 📥 Load Example Data")
    example_choice = st.selectbox(
        "Choose an example dataset to get started",
        ["None", "Ascorbic acid (example)", "Browning (example)"],
        key="kin_example_choice",
    )

    # Editable data input (no CSV upload required)
    st.markdown("### 📝 Enter Your Experimental Data")
    st.caption("Enter time points and corresponding quality measurements (e.g., vitamin content, color index, freshness score)")

    if example_choice != "None":
        if example_choice == "Ascorbic acid (example)":
            days = [0, 10, 20, 30, 40, 50]
            Avals = [271, 109, 58, 30.5, 18, 10]
        else:
            days = [0, 10, 20, 30, 40, 50, 60]
            Avals = [0.05, 0.071, 0.089, 0.11, 0.128, 0.149, 0.17]
        df_default = pd.DataFrame({"time": days, "A": Avals})
    else:
        st.caption("Using example synthetic data (1st-order decay).")
        t = np.linspace(0, 10, 11)
        A = 100 * np.exp(-0.2 * t)
        df_default = pd.DataFrame({"time": t, "A": A})

    # Persist editable table in session state so edits survive reruns
    if "kin_const_df" not in st.session_state:
        st.session_state["kin_const_df"] = df_default

    # Use `st.data_editor` when available, otherwise fall back to experimental API
    data_editor_fn = getattr(st, "data_editor", None) or getattr(st, "experimental_data_editor", None)
    if data_editor_fn is not None:
        edited = data_editor_fn(
            st.session_state["kin_const_df"],
            num_rows="dynamic",
            height=350,
            width='stretch',
            key="kin_const_editor",
        )
        # ensure we have a DataFrame
        try:
            edited_df = pd.DataFrame(edited)
        except Exception:
            edited_df = st.session_state["kin_const_df"]
        st.session_state["kin_const_df"] = edited_df
    else:
        st.info("Editable table not available in this Streamlit version. Showing static table.")
        st.dataframe(st.session_state["kin_const_df"], width='stretch')

    df = st.session_state["kin_const_df"]

    # Time unit selector for input table (affects fitting and labels)
    # Hardcode time_unit to "days"
    time_unit = "days"
    st.dataframe(df.head(), width='stretch')

    if not {"time", "A"}.issubset(df.columns):
        st.error("CSV must have columns: time, A")
        return

    # Run fitting only when user requests it
    run_fit = st.button("🔬 Analyze Data & Fit Models", key="kin_const_run_fit")

    if run_fit:
        t_input = df["time"].values.astype(float)
        A = df["A"].values.astype(float)

        # Convert input times to hours for internal fitting calculations
        factor = 1.0 if time_unit == "hours" else 24.0
        t = t_input * factor

        if len(t) < 2:
            st.error("Need at least 2 data points to fit models.")
            return

        # Determine trend (increasing vs decreasing) from data
        increasing_guess = A[-1] > A[0]

        fits = {}

        # 0th order: A = b + m*t  -> slope = m, k = abs(m)
        x0 = t
        y0 = A
        m0, b0 = np.polyfit(x0, y0, 1)
        # predicted y evaluated at input (plot) times
        y0_pred = m0 * t + b0
        y0_pred_plot = m0 * t + b0
        ss_res0 = np.sum((y0 - y0_pred) ** 2)
        ss_tot0 = np.sum((y0 - np.mean(y0)) ** 2)
        r20 = 1.0 - ss_res0 / ss_tot0 if ss_tot0 > 0 else 0.0
        k0 = abs(m0)
        # store predicted x in original input units for plotting
        fits[0] = {"k": float(k0), "r2": float(r20), "pred_x": t_input, "pred_y": y0_pred_plot}

        # 1st order: ln A = lnA0 - k t
        if np.any(A <= 0):
            fits[1] = {"k": None, "r2": -1.0, "pred_x": t, "pred_y": None}
        else:
            x1 = t
            y1 = np.log(A)
            m1, b1 = np.polyfit(x1, y1, 1)
            y1_pred = m1 * x1 + b1
            # predicted (exp) evaluated at fit times; will map to plot times later
            ss_res1 = np.sum((y1 - y1_pred) ** 2)
            ss_tot1 = np.sum((y1 - np.mean(y1)) ** 2)
            r21 = 1.0 - ss_res1 / ss_tot1 if ss_tot1 > 0 else 0.0
            k1 = abs(m1)
            # store predicted values for plotting at original input times
            fits[1] = {"k": float(k1), "r2": float(r21), "pred_x": t_input, "pred_y": np.exp(y1_pred)}

        # 2nd order: 1/A = 1/A0 + k t (for decreasing)
        if np.any(A <= 0):
            fits[2] = {"k": None, "r2": -1.0, "pred_x": t, "pred_y": None}
        else:
            x2 = t
            y2 = 1.0 / A
            m2, b2 = np.polyfit(x2, y2, 1)
            y2_pred = m2 * x2 + b2
            ss_res2 = np.sum((y2 - y2_pred) ** 2)
            ss_tot2 = np.sum((y2 - np.mean(y2)) ** 2)
            r22 = 1.0 - ss_res2 / ss_tot2 if ss_tot2 > 0 else 0.0
            k2 = abs(m2)
            fits[2] = {"k": float(k2), "r2": float(r22), "pred_x": t_input, "pred_y": 1.0 / y2_pred}

        # Choose best model by R²
        best_order = max(fits.keys(), key=lambda o: fits[o].get("r2", -1.0) if fits[o].get("r2") is not None else -1.0)
        best = fits[best_order]

        # store fits in session state for downstream actions
        st.session_state["kin_const_fits"] = fits
        st.session_state["kin_const_best_order"] = int(best_order)

        st.markdown("### 📈 Fitting Summary")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("0️⃣ 0th Order k", f"{fits[0]['k']:.6f}" if fits[0]["k"] is not None else "n/a")
        with c2:
            st.metric("1️⃣ 1st Order k", f"{fits[1]['k']:.6f}" if fits[1]["k"] is not None else "n/a")
        with c3:
            st.metric("2️⃣ 2nd Order k", f"{fits[2]['k']:.6f}" if fits[2]["k"] is not None else "n/a")
        with c4:
            st.metric("🏆 Best R²", f"{best['r2']:.4f}")

        st.markdown(f"### ✅ Selected Model")
        st.info(f"**{best_order}-Order Kinetics** (R² = {best['r2']:.4f})")
        # Display k in both per-hour (internal) and per-day for clarity
        k_per_h = best["k"]
        k_per_d = k_per_h * 24.0
        col1, col2 = st.columns(2)
        # col1.metric("k (per hour)", f"{k_per_h:.6f}") # Hidden
        col2.metric("k (per day)", f"{k_per_d:.6f}")

        st.markdown("### 📊 Fitting Plots")

        # Create three static matplotlib plots with consistent, wider aspect ratios
        # ==========================
        # ==========================
        #   Publication-quality plotting (Final Polished Version)
        # ==========================

        plt.rcParams.update({
            "font.size": 8,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 5,
            "lines.linewidth": 2.4
        })

        fig, axes = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=(4, 8),  # very clean aspect ratio
            dpi=300
        )

        fig.suptitle("Kinetic Model Comparisons", fontsize=10, fontweight="bold", y=0.98)

        plt.subplots_adjust(
            hspace=0.50,  # More breathing room
            top=0.92,
            bottom=0.08
        )

        marker_style = dict(
            s=55,
            edgecolor="white",
            linewidth=1.0,
            alpha=0.9
        )

        # -------------------------
        # 0th order
        # -------------------------
        axes[0].scatter(t_input, A, label="Observed Data", **marker_style)
        axes[0].plot(fits[0]["pred_x"], fits[0]["pred_y"], label="Fit")
        axes[0].set_title(f"0th Order: A vs Time (R²={fits[0]['r2']:.4f})")
        axes[0].set_xlabel(f"Time ({time_unit})")
        axes[0].set_ylabel("Concentration (A)")
        axes[0].grid(alpha=0.25, linestyle="--")
        axes[0].legend(loc="upper right", frameon=True)

        # -------------------------
        # 1st order
        # -------------------------
        if fits[1]["pred_y"] is not None:
            axes[1].scatter(t_input, np.log(A), label="ln(Data)", **marker_style)
            axes[1].plot(fits[1]["pred_x"], np.log(fits[1]["pred_y"]), label="Fit")
            axes[1].set_title(f"1st Order: ln(A) vs Time (R²={fits[1]['r2']:.4f})")
            axes[1].set_xlabel(f"Time ({time_unit})")
            axes[1].set_ylabel("ln(Concentration)")
            axes[1].grid(alpha=0.25, linestyle="--")
            axes[1].legend(loc="upper right", frameon=True)
        else:
            axes[1].text(0.5, 0.5,
                         "1st order unavailable\n(A contains non-positive values)",
                         ha="center", va="center", fontsize=13)
            axes[1].set_axis_off()

        # -------------------------
        # 2nd order
        # -------------------------
        if fits[2]["pred_y"] is not None:
            axes[2].scatter(t_input, 1.0 / A, label="1/Data", **marker_style)
            axes[2].plot(fits[2]["pred_x"], 1.0 / fits[2]["pred_y"], label="Fit")
            axes[2].set_title(f"2nd Order: 1/A vs Time (R²={fits[2]['r2']:.4f})")
            axes[2].set_xlabel(f"Time ({time_unit})")
            axes[2].set_ylabel("1/Concentration")
            axes[2].grid(alpha=0.25, linestyle="--")
            axes[2].legend(loc="upper right", frameon=True)
        else:
            axes[2].text(0.5, 0.5,
                         "2nd order unavailable\n(A contains non-positive values)",
                         ha="center", va="center", fontsize=13)
            axes[2].set_axis_off()

        st.pyplot(fig, use_container_width=False, clear_figure=True)



    else:
        # If fits exist in session state, show summary and plots without recomputing
        if "kin_const_fits" in st.session_state:
            fits = st.session_state["kin_const_fits"]
            best_order = st.session_state.get("kin_const_best_order", 1)
            best = fits[best_order]
            st.markdown("### 📋 Previous Fitting Summary")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("0th Order Rate (k)", f"{fits[0]['k']:.6f}" if fits[0]["k"] is not None else "n/a")
            with c2:
                st.metric("1st Order Rate (k)", f"{fits[1]['k']:.6f}" if fits[1]["k"] is not None else "n/a")
            with c3:
                st.metric("2nd Order Rate (k)", f"{fits[2]['k']:.6f}" if fits[2]["k"] is not None else "n/a")
            with c4:
                st.metric("Best Fit Quality (R²)", f"{best['r2']:.4f}")
            st.markdown(f"**Selected:** {best_order}-Order (R² = {best['r2']:.4f})")
            st.metric("Selected k", f"{best['k']:.6f} (1/time)")
            st.info("💡 Click 'Run fitting and generate plots' to recompute with current table data.")
        else:
            st.info("⏳ No fit results yet. Click 'Run fitting and generate plots' to perform fitting.")

    st.markdown("---")
    st.markdown("### 🎯 Calculate time required to reach critical limit using fitted data")
    st.caption("Use the best-fit model to predict when your product will reach the critical quality threshold")

    # derive A0 from current table (available even if fitting not yet run)
    # derive A0 from table, but allow user to override
    try:
        t_vals = df["time"].values.astype(float)
        A_vals = df["A"].values.astype(float)

        default_A0 = float(A_vals[0])

        st.markdown("### ⚙️ Initial Concentration (A₀)")
        A0 = st.number_input(
            "Set initial concentration (A₀)",
            value=default_A0,
            help="Default is the first value in your dataset. Adjust if needed.",
            key="kin_const_A0"
        )

        increasing_guess_table = A_vals[-1] > A0

    except Exception:
        st.error("Table must contain numeric 'time' and 'A' columns.")
        return

    Ai = st.number_input(
        "Critical value (Ai)",
        value=A0 / 2.0,
        key="kin_const_Ai",
    )
    increasing = st.checkbox("Attribute is increasing (else decreasing)", value=increasing_guess_table, key="kin_const_increasing")

    if st.button("Calculate shelf life from selected model", key="kin_const_calc_btn"):
        # Ensure we have a fit
        if "kin_const_fits" not in st.session_state:
            st.error("No fitted model available — run 'Run fitting and generate plots' first.")
        else:
            fits = st.session_state["kin_const_fits"]
            best_order = st.session_state.get("kin_const_best_order", 1)
            best = fits[best_order]
            t_val, msg = compute_shelf_life_isothermal(A0, Ai, best["k"], int(best_order), increasing)
            if t_val is None:
                st.error(msg)
            else:
                st.success("Shelf life computed.")
                st.success("Shelf life computed.")
                # st.metric("Shelf life (hours)", f"{t_val:.2f}") # Hidden
                st.metric("Shelf life (days)", f"{t_val/24:.2f}")


# ============================================================
#  TAB 3 — Q10 / ACCELERATION FACTOR
# ============================================================

def shelf_life_from_q10_tab():
    st.header("🌡️ Temperature Scaling & Accelerated Testing")
    st.caption("Predict shelf life at different temperatures using Q10 or acceleration factors")
    
    with st.expander("ℹ️ **Q10 Temperature Model** — Click to learn more", expanded=False):
        st.markdown("""
        ### Temperature Acceleration (Q10 Model)
        
        The Q10 factor describes how reaction rates change with a 10°C temperature increase.
        
        **Q10 Formula:**
        $$t_2 = t_1 \\times Q_{10}^{\\frac{T_1 - T_2}{10}}$$
        
        **Interpretation:**
        - **Q10 = 2**: Reaction rate doubles for every 10°C increase
        - **Q10 = 3**: Reaction rate triples for every 10°C increase
        - Typical Q10 for food: 2-3
        
        **Higher temperature → Shorter shelf life** (for spoilage reactions)
        """)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        t_ref = st.number_input(
            "Known Shelf Life (days)",
            value=30.0,
            key="q10_tref",
        )
    with c2:
        T_ref = st.number_input(
            "At Temperature (°C)",
            value=25.0,
            key="q10_Tref",
        )
    with c3:
        T_new = st.number_input(
            "Predict at Temperature (°C)",
            value=35.0,
            key="q10_Tnew",
        )
    with c4:
        Q10 = st.number_input(
            "Q10 Factor",
            value=2.0,
            min_value=0.1,
            max_value=10.0,
            step=0.1,
            key="q10_Q10",
        )

    # If an Arrhenius model exists in session, compute Q10 automatically at T_ref
    from arrhenius_ui import _load_model_from_session
    current_model = _load_model_from_session()

    if current_model is not None:
        try:
            k_ref = current_model.predict_k(T_ref, out_unit="1/h")
            k_ref_plus10 = current_model.predict_k(T_ref + 10.0, out_unit="1/h")
            if k_ref > 0 and k_ref_plus10 > 0:
                Q10_auto = k_ref_plus10 / k_ref
            else:
                Q10_auto = None
        except Exception:
            Q10_auto = None
    else:
        Q10_auto = None

    if Q10_auto is not None:
        st.success(f"Estimated Q10 from fitted Arrhenius model at {T_ref:.1f}°C: {Q10_auto:.3f}")
        use_Q10 = st.checkbox("Use Arrhenius-estimated Q10 for shelf-life calc", value=True, key="q10_use_auto")
        if use_Q10:
            Q10 = float(Q10_auto)

    if st.button("🧮 Calculate Shelf Life", key="q10_calc_btn"):
        t2 = t_ref * (Q10 ** ((T_ref - T_new) / 10.0))
        st.success("Q10-based shelf life estimated.")
        st.metric("Shelf life at target T (days)", f"{t2:.2f}")
        st.metric("t₂ / t₁ ratio", f"{t2/t_ref:.3f}")

    st.markdown("---")
    st.subheader("🔬 Calculate Acceleration Factor")
    st.caption("Determine the acceleration factor from rate constants at two temperatures")

    k1 = st.number_input("k₁ at T₁", value=0.01, key="q10_k1")
    k2 = st.number_input("k₂ at T₂", value=0.03, key="q10_k2")

    if st.button("Compute AF and effective Q10", key="q10_af_btn"):
        if k1 <= 0 or k2 <= 0:
            st.error("k₁ and k₂ must be > 0.")
        else:
            AF = k2 / k1
            if T_new == T_ref:
                Q10_eff = float("nan")
            else:
                Q10_eff = AF ** (10.0 / (T_new - T_ref))

            st.success("Acceleration factor and effective Q10 estimated.")
            st.metric("Acceleration factor AF = k₂/k₁", f"{AF:.3f}")
            st.metric("Effective Q10", f"{Q10_eff:.3f}")

    st.markdown("---")
    st.subheader("🎯 Use Arrhenius Model for Prediction")
    st.caption("Calculate shelf life at any temperature using your fitted Arrhenius model")
    if current_model is None:
        st.info("Fit an Arrhenius model in the 'Arrhenius fitting' tab to enable this feature.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            A0 = st.number_input("Initial value (A0)", value=100.0, key="q10_A0")
            Ai = st.number_input("Critical value (Ai)", value=50.0, key="q10_Ai")
        with c2:
            order = st.selectbox("Reaction order", [0, 1, 2], index=1, key="q10_order")
            increasing = st.checkbox("Attribute is increasing", value=False, key="q10_increasing")

        T_pred = st.number_input("Temperature for prediction (°C)", value=T_new, key="q10_Tpred")

        if st.button("Predict shelf life from Arrhenius k", key="q10_predict_arrhenius"):
            k_pred = current_model.predict_k(T_pred, out_unit="1/h")
            if not (k_pred and k_pred > 0):
                st.error("Predicted k is not finite/positive; check Arrhenius model and temperature.")
            else:
                t_val, msg = compute_shelf_life_isothermal(A0, Ai, float(k_pred), int(order), increasing)
                if t_val is None:
                    st.error(msg)
                else:
                    st.success("Shelf life computed using Arrhenius-derived k.")
                    st.metric("k (1/h)", f"{k_pred:.6e}")
                    st.metric("Shelf life (hours)", f"{t_val:.2f}")
                    st.metric("Shelf life (days)", f"{t_val/24.0:.2f}")


# ============================================================
#  TAB 6 — DIRECT k SHELF-LIFE
# ============================================================

def kinetic_direct_k_tab():
    st.header("⚡ Quick Shelf Life Calculator")
    st.caption("Calculate shelf life directly if you already know the degradation rate constant (k)")


    st.markdown("""
    **When to use this tool:** If you already have a rate constant **k** from literature or prior experiments, 
    use this calculator to directly determine shelf life without fitting data.
    """)

    # Input blocks
    try:
        c1, c2 = st.columns(2)

        with c1:
            A0 = st.number_input("📊 Initial Quality Level (A₀)", value=100.0, key="kd_A0")
            Ai = st.number_input("🎯 Minimum Acceptable Quality (Aᵢ)", value=50.0, key="kd_Ai")

        with c2:
            order = st.selectbox(
                "⚗️ Degradation Pattern (Reaction Order)",
                [0, 1, 2],
                index=1,
                key="kd_order",
            )
            behaviour = st.radio(
                "📉 Quality Change Direction",
                ["Decreasing (e.g., vitamin loss, color fading)", "Increasing (e.g., rancidity, microbial growth)"],
                index=0,
                key="kd_behaviour",
            )

        increasing = behaviour.startswith("Increasing")

    except Exception as e:
        st.error(f"Widget rendering error: {e}")
        A0, Ai, order, increasing = 100.0, 50.0, 1, False

    st.info("💡 For increasing attributes → Ai > A0. For decreasing attributes → Ai < A0.")

    st.subheader("📍 Degradation Rate Constant (k)")

    col1, col2, col3 = st.columns(3)

    with col1:
        k_unit = st.selectbox(
            "Unit",
            ["1/h", "1/day"],
            index=0,
            key="kd_k_unit",
        )

    with col2:
        k_val = st.number_input(
            "k value",
            value=0.1,
            min_value=1e-12,
            format="%.6f",
            key="kd_k_val",
        )

    with col3:
        st.write("")  
        st.write("")  
        if st.button("🧮 Calculate Shelf Life", key="kd_calc_btn"):

            k_base = k_val / 24.0 if k_unit == "1/day" else k_val

            t_raw, msg = compute_shelf_life_isothermal(A0, Ai, k_base, int(order), increasing)

            if t_raw is None:
                st.error(f"❌ {msg}")
            else:
                st.success("✅ Shelf life computed!")
                ch, cd = st.columns(2)
                ch.metric("📅 Shelf Life (Hours)", f"{t_raw:.2f}")
                cd.metric("📅 Shelf Life (Days)", f"{t_raw/24:.2f}")

# ============================================================
#  MAIN APP WITH 6 TABS
# ============================================================


# ============================================================
#  MAIN APP
# ============================================================

from auth_manager import AuthManager

def main_app_logic():
    st.set_page_config(page_title="Shelf Life Prediction", layout="wide")
    # Apply enhanced styling for modern design, better alignment and spacing
    apply_app_style(max_width=1200)
    st.title("📦 Shelf Life Prediction Application")
    st.caption("Predict product shelf life using kinetic modeling, temperature effects, and real-world conditions")

    if not st.session_state.get("authentication_status"):
        st.error("You must log in to continue.")
        st.stop()

    # Sidebar Logout
    with st.sidebar:
        st.markdown("---")
        
        # Show user role badge
        from user_roles import show_role_badge
        show_role_badge()
        
        auth_manager = AuthManager()
        auth_manager.logout_widget()
        st.caption(f"Logged in as: {auth_manager.get_user_name()}")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏠 Home",
        "📊 Degradation Analysis",
        "🔥 Arrhenius Model",
        "📈 Dynamic Simulation",
        "🌡️ Q10 Scaling",
        "⚡ Quick Calculator",
    ])

    with tab1:
        landing_page_tab()

    with tab2:
        kinetic_analysis_constant_temp_tab()

    with tab3:
        arrhenius_fitting_tab()

    with tab4:
        ensemble_dynamic_simulation_tab()

    with tab5:
        shelf_life_from_q10_tab()

    with tab6:
        kinetic_direct_k_tab()

from login_page import show_login_page

def main():
    # Check authentication status
    if not st.session_state.get("authentication_status"):
        show_login_page()
    else:
        main_app_logic()

if __name__ == "__main__":
    main()
