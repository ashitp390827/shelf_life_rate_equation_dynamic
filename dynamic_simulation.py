# dynamic_simulation.py

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from arrhenius_ui import ArrheniusModel, R_GAS

# Optional: Open-Meteo helper (used to fetch temperature series by city)
try:
    import openmeteo_requests
    import requests_cache
    import requests
    from retry_requests import retry
    _OPENMETEO_AVAILABLE = True
except Exception:
    _OPENMETEO_AVAILABLE = False


# ============================================================
#                  KINETICS & SIMULATION CLASSES
# ============================================================

@dataclass
class KineticsSettings:
    A0: float
    Ai: float
    order: int = 1  # 0, 1, 2
    increasing: Optional[bool] = None  # if None → auto from A0/Ai

    def is_increasing(self) -> bool:
        if self.increasing is not None:
            return self.increasing
        return self.Ai > self.A0


@dataclass
class DynamicSimulationResult:
    history: List[Dict[str, Any]]
    status: str
    shelf_life_hours: Optional[float]
    hit_index: Optional[int]


class DynamicSimulator:
    """
    Time-varying simulation engine driven by an Arrhenius model.
    k(T) is computed at each time step from the ArrheniusModel.
    """

    def __init__(self, model: ArrheniusModel, kinetics: KineticsSettings):
        self.model = model
        self.kinetics = kinetics

    def integrate_profile(
        self,
        times_h: List[float],
        temps_C: List[float],
        max_days: float = 365.0,
        cycles: int = 1,
    ) -> DynamicSimulationResult:
        """
        Integrate over a temperature profile using vectorized NumPy operations.

        times_h : list[float]
            Time stamps in hours.
        temps_C : list[float]
            Temperatures in °C for each time stamp.
        max_days : float
            Max allowed simulation time (days).
        cycles : int
            Number of times to repeat the temperature pattern.

        Returns
        -------
        DynamicSimulationResult
        """
        if not temps_C or not times_h or len(temps_C) != len(times_h):
            return DynamicSimulationResult([], "Invalid temperature/time inputs", None, None)

        # 1. Prepare time and temperature arrays (handling cycles)
        times_arr = np.array(times_h, dtype=float)
        temps_arr = np.array(temps_C, dtype=float)

        # Auto-calculate cycles if needed to cover max_days
        if len(times_arr) > 1:
            data_duration_h = times_arr[-1] - times_arr[0]
            # Add one interval if it looks like a periodic series missing the last point
            # or just use the span. Let's use the span + potential last dt.
            dt_est = times_arr[1] - times_arr[0] if len(times_arr) > 1 else 1.0
            cycle_duration = data_duration_h + dt_est
            
            req_hours = max_days * 24.0
            if req_hours > data_duration_h:
                needed_cycles = math.ceil(req_hours / cycle_duration)
                # If user requested fewer cycles, override it to ensure we cover the time
                # Or should we? The user might want to stop early?
                # The prompt says "USE TEMP TIME DATE IN A LOOP UNITL REACH THE max simulajtion"
                # So we should ensure we have enough data.
                if needed_cycles > cycles:
                    cycles = int(needed_cycles)

        if cycles > 1:
            # Calculate cycle duration
            if len(times_arr) > 1:
                dt_last = times_arr[1] - times_arr[0]
                if dt_last <= 0:
                    dt_last = 1.0
            else:
                dt_last = 1.0
            
            cycle_duration = (times_arr[-1] - times_arr[0]) + dt_last
            
            # Tile arrays
            temps_arr = np.tile(temps_arr, int(cycles))
            
            # Create time offsets for each cycle
            offsets = np.arange(int(cycles)) * cycle_duration
            # Repeat times and add offsets
            # times_arr (N,) -> (cycles, N) -> add offsets -> flatten
            times_tiled = np.tile(times_arr, (int(cycles), 1))
            times_tiled += offsets[:, np.newaxis]
            times_arr = times_tiled.flatten()

        # 2. Limit to max_days
        t0 = times_arr[0]
        max_hours = max_days * 24.0
        
        # Find index where time exceeds max_hours
        valid_mask = (times_arr - t0) <= max_hours
        invalid_positions = np.where(~valid_mask)[0]
        if invalid_positions.size > 0:
            # Slice arrays to valid range (up to first invalid position)
            # We intentionally exclude the first point that exceeds the max_hours.
            last_valid_idx = int(invalid_positions[0])
            times_arr = times_arr[:last_valid_idx]
            temps_arr = temps_arr[:last_valid_idx]
            status_limit = "Max time exceeded"
        else:
            status_limit = "Completed simulation (Ai not reached)"

        if len(times_arr) < 2:
             return DynamicSimulationResult([], "Insufficient data points", None, None)

        # 3. Calculate dt and k
        # dt[i] is the interval from i to i+1
        # We use T[i] for the interval [t[i], t[i+1]] (Euler forward)
        # or we could use average T. Let's stick to T[i] as per plan (start of interval).
        
        dt = np.diff(times_arr)
        # Ensure dt > 0
        dt = np.maximum(dt, 0.0)
        
        # Calculate k for all T (except possibly the last one if we only need it for intervals)
        # We need k at T[0...N-1] to propagate to T[1...N]
        temps_for_k = temps_arr[:-1]
        
        # Vectorized k prediction
        # ArrheniusModel.predict_k is scalar, so we use the internal formula directly or map it.
        # Direct formula is faster: k = A * exp(-Ea/RT)
        # self.model.lnA is available.
        
        # Compute k for each temperature. Prefer the analytical Arrhenius
        # formula if the provided model exposes `lnA` and `Ea_J_mol` (fast, vectorized).
        # Otherwise fall back to calling a `predict_k(T, out_unit)` method per T.
        if hasattr(self.model, "lnA") and hasattr(self.model, "Ea_J_mol"):
            T_K = temps_for_k + 273.15
            # Avoid division by zero or negative T
            with np.errstate(divide='ignore', invalid='ignore'):
                ln_k = self.model.lnA - self.model.Ea_J_mol / (R_GAS * T_K)
                k_arr = np.exp(ln_k)
        else:
            # Fallback: call predict_k for each temperature (scalar API).
            k_list = []
            for T in temps_for_k:
                try:
                    if hasattr(self.model, "predict_k"):
                        k_val = float(self.model.predict_k(float(T), out_unit="1/h"))
                    else:
                        k_val = 0.0
                except Exception:
                    k_val = 0.0
                k_list.append(k_val)
            k_arr = np.array(k_list, dtype=float)
        
        # Handle invalid k (nan/inf) -> 0
        k_arr = np.nan_to_num(k_arr, nan=0.0, posinf=0.0, neginf=0.0)
        # Ensure k >= 0
        k_arr = np.maximum(k_arr, 0.0)

        # 4. Integrate Concentration
        kinetics = self.kinetics
        order = kinetics.order
        increasing = kinetics.is_increasing()
        A0 = float(kinetics.A0)
        Ai = float(kinetics.Ai)
        
        # Cumulative reaction extent: sum(k * dt)
        # This is the "exposure" or "severity"
        exposure = np.cumsum(k_arr * dt)
        # Prepend 0 for the initial state
        exposure = np.insert(exposure, 0, 0.0)
        
        # Calculate C based on order
        # C_arr will have same length as times_arr
        
        if order == 0:
            # C = C0 ± k*t
            if increasing:
                C_arr = A0 + exposure
            else:
                C_arr = A0 - exposure
                
        elif order == 1:
            # C = C0 * exp(± k*t)
            if increasing:
                C_arr = A0 * np.exp(exposure)
            else:
                C_arr = A0 * np.exp(-exposure)
                
        elif order == 2:
            # 1/C = 1/C0 ∓ k*t
            # C = 1 / (1/C0 ∓ exposure)
            inv_C0 = 1.0 / A0 if A0 != 0 else float('inf')
            
            if increasing:
                # 1/C = 1/C0 - exposure
                # Singularity when exposure >= 1/C0
                denom = inv_C0 - exposure
                
                # Handle singularity: if denom <= 0, C -> inf
                with np.errstate(divide='ignore'):
                    C_arr = 1.0 / denom
                
                # Where denom <= 0, C is effectively infinite (or negative in math, but physically infinite/exploded)
                # We set it to infinity
                C_arr[denom <= 1e-12] = float('inf')
                
            else:
                # 1/C = 1/C0 + exposure
                denom = inv_C0 + exposure
                with np.errstate(divide='ignore'):
                    C_arr = 1.0 / denom
        else:
             return DynamicSimulationResult([], f"Unsupported order {order}", None, None)

        # 5. Check for critical limit Ai
        shelf_life_h = None
        hit_index = None
        status = status_limit
        
        # Find first index where condition is met
        if increasing:
            # C >= Ai
            # Handle inf: inf >= Ai is True
            reached = C_arr >= Ai
        else:
            # C <= Ai
            reached = C_arr <= Ai
            
        if np.any(reached):
            idx = np.argmax(reached) # Returns first True index
            hit_index = int(idx)
            
            # Interpolate for more precise shelf life?
            # For now, just take the time at that index
            shelf_life_h = float(times_arr[idx] - t0)
            status = "Reached critical limit"
            
            # Clip arrays to the hit point (optional, but cleaner for plotting)
            # We keep the hit point
            C_arr = C_arr[:idx+1]
            times_arr = times_arr[:idx+1]
            temps_arr = temps_arr[:idx+1]
            # k_arr is one shorter, but we need to align for history
            # We'll reconstruct k_arr for the history output
        
        # 6. Construct History (only if needed, or for plotting)
        # For ensemble, we might skip this if we only need the result, 
        # but the current signature returns history.
        # We can optimize by not creating the list of dicts if it's too large?
        # The UI expects a list of dicts.
        
        # Re-calculate k for the final arrays (for display)
        # This is cheap compared to the loop
        # Recompute k for the final arrays (for display). Use same logic
        # as above: analytical if possible, otherwise call predict_k per T.
        if hasattr(self.model, "lnA") and hasattr(self.model, "Ea_J_mol"):
            T_K_final = temps_arr + 273.15
            with np.errstate(divide='ignore', invalid='ignore'):
                ln_k_final = self.model.lnA - self.model.Ea_J_mol / (R_GAS * T_K_final)
                k_final = np.exp(ln_k_final)
            k_final = np.nan_to_num(k_final, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            k_list = []
            for T in temps_arr:
                try:
                    if hasattr(self.model, "predict_k"):
                        k_val = float(self.model.predict_k(float(T), out_unit="1/h"))
                    else:
                        k_val = 0.0
                except Exception:
                    k_val = 0.0
                k_list.append(k_val)
            k_final = np.array(k_list, dtype=float)
            k_final = np.nan_to_num(k_final, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create DataFrame then convert to dicts (faster than list comprehension?)
        # Or just simple list comp
        history = [
            {
                "time_h": float(t),
                "temperature_C": float(T),
                "k_1_per_h": float(k),
                "concentration": float(c)
            }
            for t, T, k, c in zip(times_arr, temps_arr, k_final, C_arr)
        ]

        return DynamicSimulationResult(
            history=history,
            status=status,
            shelf_life_hours=shelf_life_h,
            hit_index=hit_index,
        )


# ============================================================
#                 HELPER FUNCTIONS FOR UI
# ============================================================

def _get_arrhenius_model_from_session() -> Optional[ArrheniusModel]:
    data = st.session_state.get("arrhenius_model")
    if not data:
        return None
    try:
        return ArrheniusModel.from_dict(data)
    except Exception:
        return None


def _summarize_single_simulation(
    result: DynamicSimulationResult,
    kinetics: KineticsSettings,
) -> Dict[str, Any]:
    """Show summary of one simulation and return stats dict."""
    if not result.history:
        st.error("No simulation data available.")
        return {"status": "no_data"}

    df = pd.DataFrame(result.history)

    increasing = kinetics.is_increasing()
    Ai = kinetics.Ai

    if result.shelf_life_hours is None:
        finalC = df["concentration"].iloc[-1]
        st.info(
            f"Critical limit Ai = {Ai:.4g} was **not reached** in the simulated period.\n\n"
            f"Final concentration = {finalC:.4g} "
            f"({'increasing' if increasing else 'decreasing'} attribute)."
        )
        st.metric("Final concentration", f"{finalC:.4g}")
        return {
            "status": "not_reached",
            "final_concentration": float(finalC),
            "increasing": increasing,
        }
    else:
        t_h = float(result.shelf_life_hours)
        t_days = t_h / 24.0
        st.success(
            f"Critical limit Ai = {Ai:.4g} was reached at **t = {t_h:.2f} h** "
            f"(≈ {t_days:.2f} days, {'increasing' if increasing else 'decreasing'} attribute)."
        )
        c1, c2 = st.columns(2)
        c1.metric("Shelf life (hours)", f"{t_h:.2f}")
        c2.metric("Shelf life (days)", f"{t_days:.2f}")
        return {
            "status": "reached",
            "shelf_life_h": t_h,
            "shelf_life_days": t_days,
            "increasing": increasing,
        }


def _plot_single_history(df: pd.DataFrame, k_unit: str = "1/h"):
    """Plot time series: temperature and concentration on dual axes.

    k_unit: desired unit for k display ('1/h' or '1/day').
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # If the time series spans many hours, plot time in days for readability
    # Always plot in days as per requirement
    x_vals = df["time_h"] / 24.0
    x_label = "Time (days)"

    # Create figure with secondary y-axis and a clean white template
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add Temperature trace (Left Y-axis) - Orange
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=df["temperature_C"],
            name="Temperature (°C)",
            mode="lines+markers",
            marker=dict(size=4, color="#FF6F00", opacity=0.9),
            line=dict(color="#FF6F00", width=2)
        ),
        secondary_y=False,
    )

    # Add Concentration trace (Right Y-axis) - Green
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=df["concentration"],
            name="Concentration",
            mode="lines",
            line=dict(color="#2E7D32", width=2.5),
            fill='tozeroy',
            fillcolor='rgba(46, 125, 50, 0.15)'
        ),
        secondary_y=True,
    )

    # X and Y axis labels and styling
    fig.update_xaxes(title_text=x_label, title_font_size=13)
    fig.update_yaxes(title_text="Temperature (°C)", secondary_y=False, title_font=dict(color="#FF6F00", size=13))
    fig.update_yaxes(title_text="Concentration", secondary_y=True, title_font=dict(color="#2E7D32", size=13))

    fig.update_layout(
        title_text="Temperature and Concentration vs Time",
        title_font_size=16,
        title_font_color="#2E7D32",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        margin=dict(t=60, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Plot k separately with emphasized color and area under curve
    df_k = df.copy()
    if k_unit in ("1/day", "per_day"):
        df_k["k_out"] = df_k["k_1_per_h"] * 24.0
        k_label = "Rate Constant k vs Time (1/day)"
    else:
        df_k["k_out"] = df_k["k_1_per_h"]
        k_label = "Rate Constant k vs Time (1/h)"

    # Use same x unit for k plot
    df_k_plot = df_k.copy()
    # Always plot in days
    df_k_plot["x_plot"] = df_k_plot["time_h"] / 24.0
    k_xlabel = "Time (days)"

    fig_k = px.line(df_k_plot, x="x_plot", y="k_out", title=k_label, template="plotly_white")
    fig_k.update_traces(line_color="#2E7D32", line_width=2.5, mode="lines")
    fig_k.update_xaxes(title_text=k_xlabel, title_font_size=13)
    fig_k.update_yaxes(title_font_size=13)
    fig_k.update_layout(
        title_font_size=16,
        title_font_color="#2E7D32",
        margin=dict(t=60, b=40)
    )
    st.plotly_chart(fig_k, use_container_width=True)


def _ensure_df_temp(df_temp):
    """Return a valid DataFrame for temperature series.

    If `df_temp` is None or doesn't have a `.head()` method, generate a
    synthetic 365-day hourly sinusoidal temperature series and return it.
    """
    if df_temp is None:
        t = np.arange(0, 24 * 365)
        T = 25 + 10 * np.sin(2 * np.pi * (t % 24) / 24.0)
        return pd.DataFrame({"time_h": t, "temp_C": T})
    # defensive: if object doesn't look like a DataFrame
    if not hasattr(df_temp, "head"):
        t = np.arange(0, 24 * 365)
        T = 25 + 10 * np.sin(2 * np.pi * (t % 24) / 24.0)
        return pd.DataFrame({"time_h": t, "temp_C": T})
    return df_temp


# ------------------ Geocoding helper ------------------
GEOCODE_API = "https://geocoding-api.open-meteo.com/v1/search"

def search_city(name, max_results: int = 10):
    """Return list of matching city dicts from Open-Meteo geocoding API.

    Each returned dict typically contains keys like 'name', 'country',
    'latitude', 'longitude', and 'admin1'. Returns empty list on error.
    """
    try:
        params = {"name": name, "count": int(max_results), "language": "en", "format": "json"}
        r = requests.get(GEOCODE_API, params=params, timeout=10)
        if r.status_code != 200:
            return []
        data = r.json()
        return data.get("results", [])
    except Exception:
        return []


# ------------------------------------------------------------
# Compatibility wrapper used by tests
# ------------------------------------------------------------
def integrate_dynamic_profile(
    temps_C,
    times_h,
    A0: float,
    Ai: float,
    arr_model=None,
    predict_k_from_arrhenius=None,
    order: int = 1,
    max_days: float = 365.0,
    cycles: int = 1,
):
    """Compatibility wrapper returning (history, status).

    Tests call this function directly. It builds a DynamicSimulator using
    either a provided Arrhenius-like model or a small wrapper that uses
    `predict_k_from_arrhenius(model, T)` to obtain k for each T.
    """
    # Build model wrapper
    model_obj = None

    if predict_k_from_arrhenius is not None:
        class _Wrapper:
            def __init__(self, predictor):
                self._p = predictor

            def predict_k(self, T, out_unit="1/h"):
                try:
                    return float(self._p(None, T))
                except Exception:
                    return 0.0

        model_obj = _Wrapper(predict_k_from_arrhenius)
    elif arr_model is not None:
        model_obj = arr_model
    else:
        # No model — treat k as zero
        class _Zero:
            def predict_k(self, T, out_unit="1/h"):
                return 0.0

        model_obj = _Zero()

    kinetics = KineticsSettings(A0=A0, Ai=Ai, order=int(order))
    sim = DynamicSimulator(model=model_obj, kinetics=kinetics)

    result = sim.integrate_profile(times_h=times_h, temps_C=temps_C, max_days=max_days, cycles=cycles)

    # Convert DynamicSimulationResult.history to the shape expected by tests
    history_out = []
    for row in result.history:
        history_out.append({
            "time": float(row.get("time_h", row.get("time", 0.0))),
            "concentration": float(row.get("concentration", row.get("C", 0.0))),
        })

    return history_out, result.status


# ============================================================
#                        DYNAMIC TAB
# ============================================================

def dynamic_simulation_tab():
    st.header("Dynamic Shelf-Life Simulation")

    st.markdown(
        """
Professional, reproducible simulations of shelf life under a time-varying temperature profile.

- Uses a fitted **ArrheniusModel** (from the Arrhenius tab) or manually entered parameters.
- Integrates 0th, 1st, or 2nd order kinetics over the provided temperature profile.
- Use the form below to configure and run a single simulation; results include a downloadable CSV and publication-ready plots.
        """
    )

    # ---------------- Arrhenius model ----------------
    with st.expander("Arrhenius model", expanded=True):
        existing_model = _get_arrhenius_model_from_session()
        existing_model = _get_arrhenius_model_from_session()
        
        # Enforce using model from Arrhenius tab
        model: Optional[ArrheniusModel] = None

        if existing_model is None:
            st.warning("No model found in session. Please fit or load one in the Arrhenius tab.")
        else:
            model = existing_model
            st.success("Using Arrhenius model from session.")
            st.write(f"**ln A:** {model.lnA:.4f}")
            st.write(f"**A (1/h):** {math.exp(model.lnA):.3e}")
            st.write(f"**Ea:** {model.Ea_J_mol/1000.0:.3f} kJ/mol")

    if model is None:
        st.warning("No Arrhenius model available. Fit or load a model in the Arrhenius tab.")
        return

    # ---------------- Temperature profile ----------------
    st.subheader("Temperature profile")

    uploaded = st.file_uploader(
        "Upload temperature profile CSV (columns: time_h, temp_C) - Note: time_h is treated as hours internally, ensure your data is consistent or use synthetic.",
        type=["csv"],
        key="dyn_tempfile",
    )

    if uploaded is not None:
        df_temp = pd.read_csv(uploaded)
    else:
        st.caption("No file uploaded — using synthetic 5-day hourly sinusoidal profile as example.")
        t = np.arange(0, 24 * 5 + 1)
        T = 25 + 10 * np.sin(2 * np.pi * t / 24.0)
        df_temp = pd.DataFrame({"time_h": t, "temp_C": T})

    # If no data source produced a dataframe, fall back to a synthetic series
    if df_temp is None:
        st.info("No temperature data provided — using synthetic 365-day hourly data (sinusoidal daily cycle).")
        t = np.arange(0, 24 * 365)  # one year
        T = 25 + 10 * np.sin(2 * np.pi * (t % 24) / 24.0)
        df_temp = pd.DataFrame({"time_h": t, "temp_C": T})

    # Ensure df_temp is a DataFrame; fall back to synthetic if still None
    if df_temp is None:
        st.info("No temperature data available (upload or fetch). Using synthetic 365-day hourly data.")
        t = np.arange(0, 24 * 365)  # one year
        T = 25 + 10 * np.sin(2 * np.pi * (t % 24) / 24.0)
        df_temp = pd.DataFrame({"time_h": t, "temp_C": T})

    df_temp = _ensure_df_temp(df_temp)
    
    with st.expander("View Temperature Data", expanded=False):
        st.dataframe(df_temp.head(), width='stretch')

    if not {"time_h", "temp_C"}.issubset(df_temp.columns):
        st.error("CSV must contain 'time_h' and 'temp_C'.")
        return

    times_h = df_temp["time_h"].values.astype(float).tolist()
    temps_C = df_temp["temp_C"].values.astype(float).tolist()

    # ---------------- Kinetics ----------------
    st.subheader("Kinetics and simulation settings")

    # Use a small form for running a single simulation to avoid accidental reruns
    with st.form(key="dyn_run_form"):
        c1, c2 = st.columns(2)
        with c1:
            A0 = st.number_input("Initial value (A₀)", value=100.0, key="dyn_A0")
            Ai = st.number_input("Critical value (Aᵢ)", value=50.0, key="dyn_Ai")
            order = st.selectbox("Reaction order", [0, 1, 2], index=1, key="dyn_order")
        with c2:
            max_days = st.number_input(
                "Max simulation length (days)",
                min_value=0.1,
                value=365.0,
                key="dyn_maxdays",
            )
            cycles = st.number_input(
                "Repeat profile cycles",
                min_value=1,
                max_value=100,
                value=1,
                key="dyn_cycles",
            )

        kinetics = KineticsSettings(A0=A0, Ai=Ai, order=int(order))
        direction_label = (
            "increasing (e.g. peroxide/microbes)" if kinetics.is_increasing()
            else "decreasing (e.g. vitamin/nutrient loss)"
        )
        st.caption(f"Model interprets attribute as **{direction_label}** based on A₀ and Aᵢ.")

        submit = st.form_submit_button("Run dynamic simulation")

    if submit:
        # Basic validation
        if A0 is None or Ai is None:
            st.error("Please provide A₀ and Aᵢ values.")
        else:
            sim = DynamicSimulator(model=model, kinetics=kinetics)
            with st.spinner("Running simulation..."):
                result = sim.integrate_profile(
                    times_h=times_h,
                    temps_C=temps_C,
                    max_days=float(max_days),
                    cycles=int(cycles),
                )

            st.success(f"Simulation finished — status: {result.status}")

            if not result.history:
                st.warning("Simulation produced no history. Check inputs and model.")
            else:
                df_hist = pd.DataFrame(result.history)
                
                # Create tabs for results
                tab_overview, tab_rate, tab_data = st.tabs(["📊 Overview", "📉 Rate Constant", "💾 Data"])
                
                with tab_overview:
                    _summarize_single_simulation(result, kinetics)
                    _plot_single_history(df_hist, k_unit="1/day")
                
                with tab_rate:
                    st.markdown("#### Rate Constant Evolution")
                    # Force k_unit to 1/day for plotting
                    # We need to extract the k plotting logic from _plot_single_history or just call it?
                    # _plot_single_history plots BOTH. Let's refactor _plot_single_history slightly or just copy the k-plot part here.
                    # Actually, _plot_single_history does both. Let's split it or just call a new helper.
                    # For now, I will inline the k-plot logic here to separate it, or modify _plot_single_history.
                    # To avoid changing helper signature too much, I'll just manually plot k here using the same code.
                    
                    # Plot k separately
                    import plotly.express as px
                    df_k = df_hist.copy()
                    # Always 1/day
                    df_k["k_out"] = df_k["k_1_per_h"] * 24.0
                    k_label = "Rate Constant k vs Time (1/day)"
                    df_k["x_plot"] = df_k["time_h"] / 24.0
                    k_xlabel = "Time (days)"

                    fig_k = px.line(df_k, x="x_plot", y="k_out", title=k_label, template="plotly_white")
                    fig_k.update_traces(line_color="#2E7D32", line_width=2.5, mode="lines")
                    fig_k.update_xaxes(title_text=k_xlabel, title_font_size=13)
                    fig_k.update_yaxes(title_font_size=13)
                    fig_k.update_layout(
                        title_font_size=16,
                        title_font_color="#2E7D32",
                        margin=dict(t=60, b=40)
                    )
                    st.plotly_chart(fig_k, use_container_width=True)

                with tab_data:
                    st.markdown("#### Simulation data (head)")
                    st.dataframe(df_hist.head(), width='stretch')
                    
                    # Download with timestamped filename
                    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                    csv_data = df_hist.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download dynamic simulation CSV",
                        data=csv_data,
                        file_name=f"dynamic_simulation_history_{ts}.csv",
                        mime="text/csv",
                        key="dyn_download_hist",
                    )


# ============================================================
#                 ENSEMBLE DYNAMIC SIMULATION
# ============================================================

def ensemble_dynamic_simulation_tab():
    st.header("📈 Simulate Shelf Life with Real Weather Data")
    st.caption("Run multiple simulations using actual temperature patterns to see seasonal variation in shelf life")

    # ---------------- Arrhenius model ----------------
    with st.expander("🌡️ Temperature Model Setup", expanded=True):
        existing_model = _get_arrhenius_model_from_session()
        existing_model = _get_arrhenius_model_from_session()
        
        # Enforce using model from Arrhenius tab
        model: Optional[ArrheniusModel] = None
        
        if existing_model is None:
            st.warning("No model found in session. Please fit or load one in the Arrhenius tab.")
        else:
            model = existing_model
            st.success("Using Arrhenius model from session.")
            # Show full model parameters for clarity
            st.write(f"**Name:** {model.name}")
            st.write(f"**ln A:** {model.lnA:.4f}")
            st.write(f"**A (1/h):** {math.exp(model.lnA):.3e}")
            st.write(f"**Ea:** {model.Ea_J_mol/1000.0:.3f} kJ/mol")
            st.write(f"**k base unit:** {model.k_unit}")
            try:
                st.code(model.to_json(), language="json")
            except Exception:
                pass

    if model is None:
        st.warning("No Arrhenius model found for ensemble. Fit or load one in the Arrhenius tab.")
        return

    # ---------------- Temperature profile ----------------
    st.subheader("📊 Temperature Data Source")
    st.caption("Choose how to provide temperature data for your simulations")

    # Check if user has premium access
    from user_roles import is_premium_user
    has_premium = is_premium_user()
    
    # Build temperature data source choices based on user role
    if has_premium:
        temp_choices = ["Fetch by city (Open-Meteo)", "Generate synthetic"]
        default_idx = 0 if _OPENMETEO_AVAILABLE else 1
    else:
        # Basic users only get synthetic option
        temp_choices = ["Generate synthetic"]
        default_idx = 0
        st.info("🔒 ** Simulation using historic weather data** is a disabled for this account. Contact through ashitp02@gamil.com to access real-world temperature data from any city.")
    
    input_mode = st.radio(
        "Temperature data source",
        temp_choices,
        index=default_idx,
        key="ens_temp_source",
    )

    # Persist temperature data and readiness in session_state so it survives Streamlit reruns
    if "ens_df_temp_ready" not in st.session_state:
        st.session_state["ens_df_temp_ready"] = False
    if "ens_df_temp" not in st.session_state:
        st.session_state["ens_df_temp"] = None

    df_temp = st.session_state.get("ens_df_temp")
    if input_mode == "Fetch by city (Open-Meteo)":
        if not _OPENMETEO_AVAILABLE:
            st.error("Open-Meteo integration not available: missing dependencies (openmeteo_requests, requests_cache, retry_requests).")
        else:
            # Keep a small built-in list of presets for convenience
            cities = {
                "Mumbai": (19.0760, 72.8777),
                "Delhi": (28.6139, 77.2090),
                "Bangalore": (12.9716, 77.5946),
                "Chennai": (13.0827, 80.2707),
                "Kolkata": (22.5726, 88.3639),
                "Pune": (18.5204, 73.8567),
            }

            col1, col2 = st.columns(2)
            with col1:
                st.caption("Select a preset city or search by name")
                city = st.selectbox("Preset city", list(cities.keys()), key="ens_city_select")
                search_term = st.text_input("Or search city name", key="ens_city_search")
                if st.button("Search", key="ens_city_search_btn"):
                    # Perform geocoding search and store results in session state
                    results = search_city(search_term, max_results=10)
                    st.session_state["ens_city_search_results"] = results

                # If there are search results, show an indexed selectbox (stores int)
                matches = st.session_state.get("ens_city_search_results", [])
                if matches:
                    display = [f"{m.get('name')} ({m.get('country','')}) — {m.get('latitude'):.3f},{m.get('longitude'):.3f}" for m in matches]
                    sel_idx = st.selectbox("Search results (choose to override preset)", list(range(len(display))), format_func=lambda i: display[i], key="ens_city_search_select")
                else:
                    sel_idx = None

            with col2:
                start_date = st.date_input("Start date", value=pd.to_datetime("2024-01-01").date(), key="ens_fetch_start")
                end_date = st.date_input("End date", value=pd.to_datetime("2024-12-31").date(), key="ens_fetch_end")

            if st.button("Fetch temperature series", key="ens_fetch_btn"):
                # Determine coordinates: preference = search result (if chosen), else preset
                lat = None
                lon = None
                chosen_label = None
                matches = st.session_state.get("ens_city_search_results", [])
                if matches and sel_idx is not None:
                    try:
                        chosen = matches[int(sel_idx)]
                        lat = float(chosen.get("latitude"))
                        lon = float(chosen.get("longitude"))
                        chosen_label = f"{chosen.get('name')}, {chosen.get('country','')}"
                    except Exception:
                        lat = None

                if lat is None:
                    # fallback to preset dictionary
                    lat, lon = cities.get(city, (None, None))
                    chosen_label = city

                if lat is None or lon is None:
                    st.error("Could not determine coordinates for selected city.")
                else:
                    try:
                        cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
                        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
                        openmeteo = openmeteo_requests.Client(session=retry_session)
                        url = "https://archive-api.open-meteo.com/v1/archive"
                        params = {
                            "latitude": float(lat),
                            "longitude": float(lon),
                            "start_date": str(start_date),
                            "end_date": str(end_date),
                            "hourly": ["temperature_2m"],
                        }
                        responses = openmeteo.weather_api(url, params=params)
                        response = responses[0]
                        hourly = response.Hourly()
                        temps = hourly.Variables(0).ValuesAsNumpy()
                        times = pd.date_range(
                            start=pd.to_datetime(hourly.Time(), unit='s', utc=True),
                            end=pd.to_datetime(hourly.TimeEnd(), unit='s', utc=True),
                            freq=pd.Timedelta(seconds=hourly.Interval()),
                            inclusive='left'
                        )
                        df_temp = pd.DataFrame({"datetime_utc": times, "temp_C": temps})
                        # convert to time_h relative to start
                        df_temp = df_temp.reset_index(drop=True)
                        df_temp["time_h"] = (df_temp.index.astype(float))
                        df_temp = df_temp[["time_h", "temp_C", "datetime_utc"]]
                        st.success(f"Fetched {len(df_temp)} hourly points for {chosen_label}.")
                        # Persist fetched data and readiness in session state so it survives reruns
                        st.session_state["ens_df_temp"] = df_temp
                        st.session_state["ens_df_temp_ready"] = True

                        # We keep the datetime_utc column; the UI will use this
                        # as the calendar anchor (time_h == 0) when available.
                    except Exception as e:
                        st.error(f"Failed to fetch data: {e}")

    else:
        st.caption("Generate a synthetic temperature series (hourly by default).")
        synth_days = st.number_input("Synthetic length (days)", min_value=1, max_value=3650, value=365, key="ens_synth_days")
        synth_baseline = st.number_input("Baseline temperature (°C)", value=25.0, key="ens_synth_baseline")
        synth_amplitude = st.number_input("Daily amplitude (°C)", value=10.0, key="ens_synth_amplitude")
        synth_noise = st.number_input("Gaussian noise (std °C)", min_value=0.0, value=0.0, key="ens_synth_noise")
        hourly = st.checkbox("Hourly resolution", value=True, key="ens_synth_hourly")
        synth_start_date = st.date_input("Synthetic series start date (calendar)", value=pd.to_datetime("2025-01-01").date(), key="ens_synth_start_date")

        if hourly:
            t = np.arange(0, 24 * int(synth_days))
            daily_frac = (t % 24) / 24.0
        else:
            t = np.arange(0, int(synth_days))
            daily_frac = np.zeros_like(t)

        T = synth_baseline + synth_amplitude * np.sin(2 * np.pi * daily_frac)
        if float(synth_noise) > 0.0:
            rng = np.random.default_rng(0)
            T = T + rng.normal(0.0, float(synth_noise), size=T.shape)

        # create datetime_utc to support inferred calendar dates
        freq = "H" if hourly else "D"
        try:
            times_dt = pd.date_range(start=pd.to_datetime(synth_start_date), periods=len(t), freq=freq)
        except Exception:
            times_dt = pd.date_range(start=pd.to_datetime("2025-01-01"), periods=len(t), freq=freq)

        df_temp = pd.DataFrame({"datetime_utc": times_dt, "time_h": t, "temp_C": T})
        # Persist synthetic data and mark ready
        st.session_state["ens_df_temp"] = df_temp
        st.session_state["ens_df_temp_ready"] = True

        # Keep datetime_utc available for calendar anchoring when present.
        # (No UI date input is required — the app will use the first timestamp.)

    df_temp = _ensure_df_temp(df_temp)

    # If the user chose fetching by city, require the fetch button to have been pressed
    if input_mode.startswith("Fetch") and not st.session_state.get("ens_df_temp_ready", False):
        st.info("Please click 'Fetch temperature series' to retrieve data for the selected city. Simulation controls will appear once fetch completes.")
        return

    # Ensure we use the persisted df_temp (fall back to local variable)
    saved_df = st.session_state.get("ens_df_temp")
    if saved_df is not None and isinstance(saved_df, pd.DataFrame):
        df_temp = saved_df
    
    with st.expander("View Selected Temperature Series", expanded=False):
        st.markdown("**Selected temperature series (head)**")
        st.dataframe(df_temp.head(), use_container_width=True)

    if not {"time_h", "temp_C"}.issubset(df_temp.columns):
        st.error("CSV must contain 'time_h' and 'temp_C'.")
        return

    times_h_full = df_temp["time_h"].values.astype(float)
    temps_C_full = df_temp["temp_C"].values.astype(float)
    # ---------------- Window length auto-selection ----------------
    # Infer total available days from the temperature series and offer an
    # automatic window option that uses the full series length.
    total_hours = float(times_h_full[-1] - times_h_full[0]) if len(times_h_full) > 1 else 0.0
    total_days = max(1.0, total_hours / 24.0)

    st.subheader("⏱️ Simulation Time Windows")
    st.caption("Each simulation starts at a different time to capture seasonal variation")
    auto_window = st.checkbox(
        "Use full series length as window (auto)", value=True, key="ens_auto_window"
    )
    if auto_window:
        window_days = float(total_days)
        st.caption(f"Using full series length: {window_days:.2f} days")
    else:
        # Allow user to override window length, but cap to total_days
        window_days = st.number_input(
            "Window length for each simulation (days)",
            min_value=1.0,
            max_value=max(1.0, float(total_days)),
            value=min(60.0, float(total_days)),
            key="ens_windowdays",
        )

    # ---------------- Kinetics + ensemble settings ----------------
    st.subheader("🧪 Product Degradation Settings")
    st.caption("Define your product's quality parameters and degradation behavior")

    c1, c2, c3 = st.columns(3)
    with c1:
        A0 = st.number_input("Initial value (A₀)", value=100.0, key="ens_A0")
        Ai = st.number_input("Critical value (Aᵢ)", value=50.0, key="ens_Ai")
        order = st.selectbox("Reaction order", [0, 1, 2], index=1, key="ens_order")
    with c2:
        n_sims = st.number_input(
            "Number of simulations",
            min_value=3,
            max_value=500,
            value=50,
            key="ens_nsims",
        )
        window_days = st.number_input(
            "Window length for each simulation (days)",
            min_value=1.0,
            max_value=365.0,
            value=60.0,
            key="ens_windowdays",
        )
    with c3:
        # Determine a calendar anchor (base_dt) for `time_h == 0`.
        # Priority: df_temp['datetime_utc'] first timestamp -> fetch `start_date` input -> synthetic `synth_start_date`.
        base_dt = None
        try:
            if hasattr(df_temp, "columns") and "datetime_utc" in df_temp.columns and not df_temp["datetime_utc"].isnull().all():
                base_dt = pd.to_datetime(df_temp["datetime_utc"].iloc[0])
                # st.caption(f"Calendar anchor: `time_h = 0` corresponds to {base_dt.isoformat()}.")
            else:
                # Fall back to fetch start_date (if the user used Fetch by city)
                try:
                    if input_mode.startswith("Fetch") and 'start_date' in locals():
                        base_dt = pd.to_datetime(start_date)
                        st.caption(f"Calendar anchor inferred from fetch start date: {base_dt.date().isoformat()}.")
                    elif 'synth_start_date' in locals():
                        base_dt = pd.to_datetime(synth_start_date)
                        st.caption(f"Calendar anchor inferred from synthetic series start date: {base_dt.date().isoformat()}.")
                    else:
                        st.caption("No calendar timestamps in the temperature series; calendar dates will not be shown.")
                except Exception:
                    base_dt = None
                    st.caption("No calendar anchor could be inferred from inputs.")
        except Exception:
            base_dt = None
            st.caption("Could not infer calendar anchor from the temperature series.")

    kinetics = KineticsSettings(A0=A0, Ai=Ai, order=int(order))
    direction_label = (
        "increasing (e.g. peroxide/microbes)" if kinetics.is_increasing()
        else "decreasing (e.g. vitamin/nutrient loss)"
    )
    st.caption(f"Model interprets attribute as **{direction_label}** based on A₀ and Aᵢ.")

    # Censored data handling
    censored_option = st.radio(
        "Handle 'Not reached' (censored) runs in stats:",
        ["Exclude from stats", "Use window length (conservative)"],
        index=0,
        key="ens_censored_opt"
    )

    # Allow user to choose k unit for outputs/plots (1/h or 1/day)
    # Hardcode k_output_unit to 1/day
    k_output_unit = "1/day"

    if st.button("Run ensemble simulations", key="ens_run_btn"):
        # Wrap long-running ensemble execution in a spinner to improve UX
        with st.spinner("Running ensemble simulations — this may take a few moments..."):
            # We no longer block if total_hours <= window_hours, because we loop the data.
            window_hours = float(window_days * 24.0)
            
            # Choose start times so windows are evenly distributed across the series
            n_sims_int = int(n_sims)
            if n_sims_int <= 0:
                st.error("Number of simulations must be >= 1")
                return

            # With wrapping, ANY index is a valid start index.
            # We want to sample n_sims start times from the available data duration.
            n_points = len(times_h_full)
            if n_points < 2:
                st.error("Not enough data points.")
                return
            
            # Indices from 0 to n_points-1
            # We pick n_sims indices evenly spaced
            if n_sims_int >= n_points:
                selected_indices = np.arange(n_points)
            else:
                idx_positions = np.linspace(0, n_points - 1, n_sims_int)
                selected_indices = np.round(idx_positions).astype(int)

            start_times = times_h_full[selected_indices]
            start_indices = selected_indices

            sim = DynamicSimulator(model=model, kinetics=kinetics)
            runs: List[Dict[str, Any]] = []

            # Pre-calculate cycle duration for wrapping logic
            dt_last = times_h_full[1] - times_h_full[0] if len(times_h_full) > 1 else 1.0
            cycle_duration = (times_h_full[-1] - times_h_full[0]) + dt_last

            for idx, (start_t, start_idx) in enumerate(zip(start_times, start_indices)):
                # Construct a rotated profile starting at start_idx
                # We take the full profile and roll it so start_idx becomes 0
                # But we need to adjust times.
                
                # Roll temperatures: T[start], T[start+1], ..., T[end], T[0], ...
                T_rotated = np.roll(temps_C_full, -start_idx)
                
                # Construct times for this rotated profile
                # We can just assume the original dt pattern rotates too, or simpler:
                # Reconstruct time from 0 using average dt? 
                # Better: Use the actual time differences, rolled.
                
                # dt array: dt[i] = t[i+1] - t[i]
                # We need to roll dt as well.
                # Let's just use the original times_h_full shifted, and handle the wrap jump.
                # Actually, integrate_profile now handles cycles. 
                # So we just need to pass the rotated T and a consistent time axis of the same length.
                
                # If we just pass 0, dt, 2dt... it might drift if dt varies.
                # Let's reconstruct time from the rolled dt.
                
                # Original dt
                dts = np.diff(times_h_full)
                # Append last interval
                dts = np.append(dts, dt_last)
                
                # Roll dts
                dts_rotated = np.roll(dts, -start_idx)
                
                # Reconstruct time
                # t[0] = 0
                # t[i] = sum(dts_rotated[:i])
                t_local = np.concatenate(([0.0], np.cumsum(dts_rotated[:-1])))
                
                # Now we have a full cycle starting at the desired season.
                # integrate_profile will loop this cycle as needed to reach window_days.
                
                result = sim.integrate_profile(
                    times_h=t_local.tolist(),
                    temps_C=T_rotated.tolist(),
                    max_days=window_days,
                    cycles=1, # It will auto-increase cycles if window_days > cycle length
                )

                if result.shelf_life_hours is None:
                    shelf_h = None
                    shelf_dt = None
                else:
                    shelf_h = float(result.shelf_life_hours)
                    global_hours = start_t + shelf_h
                    if base_dt is not None:
                        # base_dt is a pandas.Timestamp or datetime
                        try:
                            shelf_dt = (pd.to_datetime(base_dt) + timedelta(hours=float(global_hours)))
                        except Exception:
                            shelf_dt = None
                    else:
                        shelf_dt = None

                runs.append(
                    {
                        "run_id": idx,
                        "start_time_h": float(start_t),
                        "shelf_life_h": shelf_h,
                        "shelf_datetime": shelf_dt,
                        "status": result.status,
                        # Store start_idx to reconstruct profile for worst/best case plots
                        "start_idx": start_idx 
                    }
                )

        if not runs:
            st.error("No valid simulations were performed.")
            return



        df_runs = pd.DataFrame(runs)
        
        # Create tabs for ensemble results
        tab_sum, tab_season, tab_extreme, tab_raw = st.tabs([
            "📊 Summary & Distribution", 
            "📅 Seasonal Analysis", 
            "🔴/🟢 Extreme Cases", 
            "💾 Raw Data"
        ])

        with tab_raw:
            st.markdown("#### Ensemble results (head)")
            st.dataframe(df_runs.head(), use_container_width=True)

            # ---------- Export CSV ----------
            csv_data = df_runs.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download ensemble results CSV",
                data=csv_data,
                file_name="ensemble_dynamic_results.csv",
                mime="text/csv",
                key="ens_download_csv",
            )

        with tab_sum:
            # ---------- Numeric summary ----------
            st.markdown("### 📊 Ensemble summary")

            total_runs = len(df_runs)
            reached_mask = df_runs["shelf_life_h"].notnull()
            n_reached = int(reached_mask.sum())
            n_censored = int(total_runs - n_reached)

            ca, cb, cc = st.columns(3)
            ca.metric("Total simulations", f"{total_runs}")
            cb.metric("Reached Ai", f"{n_reached}")
            cc.metric("Not reached (censored)", f"{n_censored}")

            # Prepare data for stats based on censored option
            df_stats = df_runs.copy()
            
            if censored_option == "Use window length (conservative)":
                # Fill NaNs with window_hours
                df_stats["shelf_life_h"] = df_stats["shelf_life_h"].fillna(window_hours)
                # All runs are considered for stats
                valid_stats_mask = pd.Series([True] * len(df_stats), index=df_stats.index)
                st.info(f"Including {n_censored} censored runs as {window_days:.2f} days (conservative).")
            else:
                # Exclude NaNs
                valid_stats_mask = reached_mask
                if n_censored > 0:
                    st.info(f"Excluding {n_censored} censored runs from statistics.")

            if valid_stats_mask.sum() > 0:
                shelf_days = df_stats.loc[valid_stats_mask, "shelf_life_h"].astype(float) / 24.0
                p5 = float(shelf_days.quantile(0.05))
                p50 = float(shelf_days.quantile(0.50))
                p95 = float(shelf_days.quantile(0.95))
                min_d = float(shelf_days.min())
                max_d = float(shelf_days.max())

                st.markdown("**Shelf-life distribution:**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Median shelf life", f"{p50:.2f} days")
                c2.metric("5–95% range", f"{p5:.2f} – {p95:.2f} days")
                c3.metric("Min / Max", f"{min_d:.2f} – {max_d:.2f} days")

                # Colorful histogram with percentile/median markers
                fig_hist = px.histogram(
                    shelf_days,
                    x=shelf_days,
                    nbins=30,
                    title="Shelf life distribution (days)",
                    labels={"x": "Shelf life (days)"},
                    color_discrete_sequence=["#1f77b4"],
                    template="plotly_white",
                )
                fig_hist.update_layout(margin=dict(t=40, b=30))
                # Add vertical lines for p5/p50/p95
                fig_hist.add_vline(x=p50, line_width=2, line_dash="dash", line_color="#000000")
                fig_hist.add_vline(x=p5, line_width=1, line_dash="dot", line_color="#555555")
                fig_hist.add_vline(x=p95, line_width=1, line_dash="dot", line_color="#555555")
                fig_hist.add_annotation(x=p50, y=1, yanchor="bottom", text=f"Median {p50:.2f}d", showarrow=False)

                # ECDF (cumulative failure curve)
                try:
                    fig_cdf = px.ecdf(shelf_days, x=shelf_days, title="Cumulative failure (ECDF)", labels={"x": "Shelf life (days)"}, template="plotly_white")
                except Exception:
                    s = np.sort(shelf_days.to_numpy())
                    cum = np.arange(1, len(s) + 1) / float(len(s))
                    fig_cdf = px.line(x=s, y=cum, title="Cumulative failure (ECDF)", labels={"x": "Shelf life (days)", "y": "Cumulative fraction"}, template="plotly_white")
                # Mark key percentiles on the ECDF
                fig_cdf.add_vline(x=p50, line_width=2, line_dash="dash", line_color="#000000")
                fig_cdf.add_vline(x=p95, line_width=1, line_dash="dot", line_color="#555555")

                # Violin + box for spread (compact, smaller marker)
                N_points = int(len(shelf_days))
                violin_points = "all" if N_points <= 300 else "outliers"
                fig_violin = px.violin(
                    y=shelf_days,
                    box=True,
                    points=violin_points,
                    title="Distribution (violin + box)",
                    template="plotly_white",
                    color_discrete_sequence=["#EF553B"]
                )
                fig_violin.update_traces(marker=dict(size=4, opacity=0.6))

                # Arrange a compact dashboard: histogram + violin on the left, ECDF to the right
                col_left, col_right = st.columns([2, 1])
                with col_left:
                    st.plotly_chart(fig_hist, use_container_width=True)
                    st.plotly_chart(fig_violin, use_container_width=True)
                with col_right:
                    st.plotly_chart(fig_cdf, use_container_width=True)
                    st.markdown("**Notes:** Vertical dashed line = median; dotted = 5/95th percentiles.")
            else:
                st.warning("No valid data for statistics (all runs censored and excluded).")

        with tab_season:
            # ---------- Month & year analysis ----------
            # For month/year, we need a date. If censored, we can't easily assign a failure date.
            # So we only plot failures that actually happened, OR we project the date for censored ones?
            # Let's stick to actual failures for the calendar plots to avoid confusion, 
            # or maybe include them if "Use window length" is selected?
            # A conservative approach for calendar is to use the end of window date.
            
            if censored_option == "Use window length (conservative)":
                 # Fill missing dates with start_time + window
                 mask_nan_date = df_stats["shelf_datetime"].isnull()
                 if mask_nan_date.any():
                     # Calculate end dates for these
                     # We need start times.
                     # df_runs has start_time_h.
                     # base_dt is start_date (calendar).
                     # shelf_datetime = start_date + start_time_h + window_hours
                     
                     # Vectorized date calc might be tricky with pandas series of floats + date object
                     # Do a loop or apply. Use the inferred `base_dt` (calendar anchor) if available.
                     if base_dt is not None:
                         base_dt_ts = pd.to_datetime(base_dt)

                         def _calc_end_date(row):
                             if pd.isnull(row["shelf_datetime"]):
                                 hours = row["start_time_h"] + window_hours
                                 return base_dt_ts + timedelta(hours=hours)
                             return row["shelf_datetime"]

                         df_stats["shelf_datetime"] = df_stats.apply(_calc_end_date, axis=1)
                     else:
                         # Cannot fill calendar dates without an anchor; leave as NaN
                         pass

            if df_stats["shelf_datetime"].notnull().any():
                # Filter for valid stats mask if we are excluding
                if censored_option == "Exclude from stats":
                     df_dt = df_stats[valid_stats_mask].copy()
                else:
                     df_dt = df_stats.copy()

                if not df_dt.empty:
                    df_dt["shelf_datetime"] = pd.to_datetime(df_dt["shelf_datetime"])
                    df_dt["year"] = df_dt["shelf_datetime"].dt.year
                    df_dt["month"] = df_dt["shelf_datetime"].dt.month
                    df_dt["shelf_days"] = df_dt["shelf_life_h"] / 24.0
        
                    st.markdown("### Calendar analysis")
        
                    # (Removed failure-counts-by-year histogram as requested)

                    # Compute simulation start datetime (if base_dt is available) so we can
                    # group by the month the simulation started. Use start_time_h (hours offset)
                    # from the original runs.
                    if base_dt is not None and "start_time_h" in df_dt.columns:
                        try:
                            df_dt["start_datetime"] = pd.to_datetime(base_dt) + pd.to_timedelta(df_dt["start_time_h"].astype(float), unit='h')
                            df_dt["start_year"] = df_dt["start_datetime"].dt.year
                            df_dt["start_month"] = df_dt["start_datetime"].dt.month
                            # Map month numbers to short names for display and enforce chronological order
                            month_names = {i: pd.Timestamp(2000, i, 1).strftime('%b') for i in range(1, 13)}
                            df_dt["start_month_name"] = df_dt["start_month"].map(month_names)

                            # Ensure chronological order Jan..Dec
                            month_order = [pd.Timestamp(2000, i, 1).strftime('%b') for i in range(1, 13)]
                            try:
                                df_dt["start_month_name"] = pd.Categorical(df_dt["start_month_name"], categories=month_order, ordered=True)
                            except Exception:
                                pass

                            # Box plot of shelf life by simulation START month (chronological order)
                            N_month = len(df_dt)
                            month_points = "all" if N_month <= 300 else "outliers"
                            fig_start_month_box = px.box(
                                df_dt,
                                x="start_month_name",
                                y="shelf_days",
                                points=month_points,
                                title="Shelf life (days) by simulation start month",
                                labels={"start_month_name": "Start month", "shelf_days": "Shelf life (days)"},
                                template="plotly_white",
                                category_orders={"start_month_name": month_order},
                                hover_data=["run_id", "shelf_days"],
                            )
                            fig_start_month_box.update_traces(marker=dict(size=4, opacity=0.7))
                            # Display start-month boxplot and year boxplot side-by-side for quick comparison
                            col_month, col_year = st.columns([2, 1])
                            with col_month:
                                st.plotly_chart(fig_start_month_box, use_container_width=True)
                                st.caption("Months refer to the calendar month when each simulation started (based on the calendar anchor). Boxes show distribution of shelf life; individual points are simulation runs.")
                            # The year boxplot is generated below; render it in the right column after creation.
                        except Exception:
                            # Fallback: if anything fails, show a simple month histogram
                            fig_month = px.histogram(
                                df_dt,
                                x="month",
                                title="Counts of failures by month",
                                template="plotly_white",
                            )
                            st.plotly_chart(fig_month, use_container_width=True)
                    else:
                        # No base_dt/start_time — fall back to failure-month histogram
                        fig_month = px.histogram(
                            df_dt,
                            x="month",
                            title="Counts of failures by month",
                            template="plotly_white",
                        )
                        st.plotly_chart(fig_month, use_container_width=True)

                    # Keep the failure-year boxplot (shelf_datetime year)
                    fig_box = px.box(
                        df_dt,
                        x="year",
                        y="shelf_days",
                        points="all",
                        title="Shelf life (days) by failure year",
                        template="plotly_white",
                    )
                    # If we previously created the month/year columns, put this in the right column; otherwise render normally
                    try:
                        with col_year:
                            st.plotly_chart(fig_box, use_container_width=True)
                    except Exception:
                        st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.info("No shelf_datetime info; month/year plots not available.")

        with tab_extreme:
            # ---------- Worst and best case runs ----------
            st.markdown("### 🔴 Worst-case & 🟢 Best-case runs")

            # Use df_stats which might include censored values
            if valid_stats_mask.sum() == 0:
                st.info("No valid runs to determine worst/best cases.")
            else:
                df_valid = df_stats[valid_stats_mask].copy()
                df_valid["shelf_days"] = df_valid["shelf_life_h"] / 24.0

                worst_row = df_valid.loc[df_valid["shelf_days"].idxmin()]
                best_row = df_valid.loc[df_valid["shelf_days"].idxmax()]

                st.write(
                    f"**Worst-case run_id:** {int(worst_row['run_id'])}, "
                    f"shelf life ≈ {worst_row['shelf_days']:.2f} days"
                )
                st.write(
                    f"**Best-case run_id:** {int(best_row['run_id'])}, "
                    f"shelf life ≈ {best_row['shelf_days']:.2f} days"
                )

                sim = DynamicSimulator(model=model, kinetics=kinetics)

                def _get_segment(run_row):
                    # Reconstruct the rotated profile used for this run
                    start_idx = int(run_row["start_idx"])
                    
                    # Roll temperatures
                    T_rotated = np.roll(temps_C_full, -start_idx)
                    
                    # Reconstruct time axis
                    # We need dts and dt_last which are in the outer scope
                    dts = np.diff(times_h_full)
                    dts = np.append(dts, dt_last)
                    dts_rotated = np.roll(dts, -start_idx)
                    t_local = np.concatenate(([0.0], np.cumsum(dts_rotated[:-1])))
                    
                    return t_local.tolist(), T_rotated.tolist()

                t_worst, T_worst = _get_segment(worst_row)
                t_best, T_best = _get_segment(best_row)

                res_worst = sim.integrate_profile(t_worst, T_worst, max_days=window_days, cycles=1)
                res_best = sim.integrate_profile(t_best, T_best, max_days=window_days, cycles=1)

                # Render worst / best side-by-side for easier comparison
                col_w, col_b = st.columns(2)
                with col_w:
                    st.markdown("#### 🔴 Worst-case profile (shortest shelf life)")
                    if res_worst.history:
                        df_worst = pd.DataFrame(res_worst.history)
                        _plot_single_history(df_worst, k_unit=k_output_unit)
                with col_b:
                    st.markdown("#### 🟢 Best-case profile (longest shelf life)")
                    if res_best.history:
                        df_best = pd.DataFrame(res_best.history)
                        _plot_single_history(df_best, k_unit=k_output_unit)
