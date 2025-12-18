# arrhenius_ui.py

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

R_GAS = 8.3145  # J/mol-K


# ============================================================
#                    ARRHENIUS MODEL CLASS
# ============================================================

@dataclass
class ArrheniusModel:
    """
    Arrhenius model:

        ln k = ln A - Ea / (R * T)

    Internally, k is expressed in 1/h (per hour).
    """
    lnA: float
    Ea_J_mol: float
    k_unit: str = "1/h"     # base unit for k inside the model
    name: str = "Arrhenius model"

    # ---------- Core prediction ----------

    def _k_base_1_per_h(self, T_C: float) -> float:
        """Return k in 1/h (base unit)."""
        T_K = T_C + 273.15
        if T_K <= 0:
            return float("nan")
        ln_k = self.lnA - self.Ea_J_mol / (R_GAS * T_K)
        return math.exp(ln_k)

    def predict_k(self, T_C: float, out_unit: str = "1/h") -> float:
        """
        Predict k at temperature T_C (°C).

        out_unit:
            "1/h"   → per hour
            "1/day" → per day
        """
        k_h = self._k_base_1_per_h(T_C)
        if not math.isfinite(k_h):
            return float("nan")

        out_unit = out_unit.strip().lower()
        if out_unit in ("1/h", "1/hour", "per_hour"):
            return k_h
        elif out_unit in ("1/day", "per_day"):
            return k_h * 24.0
        else:
            # default to 1/h if unknown
            return k_h

    # ---------- Serialization ----------

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["Ea_J_mol"] = self.Ea_J_mol
        return d

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict) -> "ArrheniusModel":
        return cls(
            lnA=float(d["lnA"]),
            Ea_J_mol=float(d["Ea_J_mol"]),
            k_unit=str(d.get("k_unit", "1/h")),
            name=str(d.get("name", "Arrhenius model")),
        )

    @classmethod
    def from_json(cls, s: str) -> "ArrheniusModel":
        d = json.loads(s)
        return cls.from_dict(d)

    # ---------- Fitting ----------

    @staticmethod
    def fit_from_data(
        temps_C: np.ndarray,
        ks: np.ndarray,
        k_unit: str = "1/h",
        name: str = "Arrhenius model",
    ) -> Tuple[Optional["ArrheniusModel"], Optional[float]]:
        """
        Fit Arrhenius parameters from arrays of temperatures (°C) and k values.

        Parameters
        ----------
        temps_C : np.ndarray
            Temperatures in °C.
        ks : np.ndarray
            Rate constants (in k_unit).
        k_unit : str
            Unit of k passed in (UI selection). Will be converted to 1/h internally.

        Returns
        -------
        model : ArrheniusModel or None
        r2 : float or None
        """
        if temps_C is None or ks is None or len(temps_C) < 2 or len(temps_C) != len(ks):
            return None, None

        # Convert to base 1/h internally
        k_unit = k_unit.strip().lower()
        ks = np.array(ks, dtype=float)
        if k_unit in ("1/day", "per_day"):
            ks_base = ks / 24.0
        else:
            ks_base = ks

        T_K = np.array(temps_C, dtype=float) + 273.15
        if np.any(T_K <= 0) or np.any(ks_base <= 0):
            return None, None

        x = 1.0 / T_K
        y = np.log(ks_base)

        x_mean = np.mean(x)
        y_mean = np.mean(y)
        num = np.sum((x - x_mean) * (y - y_mean))
        den = np.sum((x - x_mean) ** 2)
        if den == 0:
            return None, None

        m = num / den
        b = y_mean - m * x_mean

        # ln(k) = ln(A) - Ea/R * 1/T  → slope = -Ea/R
        Ea = -m * R_GAS
        lnA = b

        # R²
        y_pred = m * x + b
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

        model = ArrheniusModel(
            lnA=float(lnA),
            Ea_J_mol=float(Ea),
            k_unit="1/h",
            name=name,
        )
        return model, (float(r2) if r2 is not None else None)


# ============================================================
#                    ARRHENIUS FITTING UI
# ============================================================

def _load_model_from_session() -> Optional[ArrheniusModel]:
    data = st.session_state.get("arrhenius_model")
    if not data:
        return None
    try:
        return ArrheniusModel.from_dict(data)
    except Exception:
        return None


def arrhenius_fitting_tab():
    st.header("🔥 Model Temperature Effects on Shelf Life")
    st.caption("Determine how temperature affects degradation rate using the Arrhenius equation")

    with st.expander("ℹ️ **Arrhenius Equation** — Click to learn more", expanded=False):
        st.markdown(
            """
            ### Temperature Dependence of Reaction Rates
            
            The Arrhenius equation describes how reaction rates change with temperature:
            
            $$\\ln k = \\ln A - \\frac{E_a}{RT}$$
            
            Or equivalently:
            $$k = A \\cdot e^{-E_a / (RT)}$$
            
            **Parameters:**
            - **A** = Pre-exponential factor (frequency factor) [1/time]
            - **Ea** = Activation energy [J/mol] — typically 50-150 kJ/mol for food reactions
            - **R** = Gas constant = 8.314 J/(mol·K)
            - **T** = Absolute temperature [K]
            - **k** = Rate constant [1/time]
            
            **Interpretation:**
            - Higher Ea = Stronger temperature dependence (steeper slope)
            - Higher A = Faster reaction at all temperatures
            - Plot ln(k) vs 1/T gives a straight line → linear regression
            
            **Typical Food Science Values:**
            - Nutrients (L-ascorbic acid, B vitamins): 40-60 kJ/mol
            - Color changes, browning: 50-80 kJ/mol
            - Microbial growth: 60-100 kJ/mol
            """
        )

    # --------------------------------------------------------
    # Section: existing model
    # --------------------------------------------------------
    existing_model = _load_model_from_session()
    with st.expander("✅ **Saved Temperature Model**", expanded=existing_model is not None):
        if existing_model is None:
            st.info("No Arrhenius model stored in the session yet.")
        else:
            st.success("✅ A model is available from a previous fit / load.")
            st.write(f"**Model Name:** {existing_model.name}")
            col1, col2, col3 = st.columns(3)
            col1.metric("ln A", f"{existing_model.lnA:.4f}")
            col2.metric("A (1/h)", f"{math.exp(existing_model.lnA):.3e}")
            col3.metric("Ea (kJ/mol)", f"{existing_model.Ea_J_mol/1000.0:.3f}")
            with st.expander("📋 Model JSON"):
                st.code(existing_model.to_json(), language="json")

    st.markdown("---")

    # --------------------------------------------------------
    # Section: data input
    # --------------------------------------------------------
    st.subheader("📥 Enter Rate Constants at Different Temperatures")
    st.caption("Provide rate constants (k) measured at different storage temperatures (use time unit as day) to build your model")

    # Hardcode k_input_unit to 1/day (hidden from user)
    k_input_unit = "1/day"
    st.session_state.arr_k_input_unit = k_input_unit

    st.caption("Enter temperature (°C) and corresponding k values in the table below.")

    # Initialize the DF only once
    if "arr_input_df" not in st.session_state:
        st.session_state.arr_input_df = pd.DataFrame({
            "T_C": [25.0, 35.0, 45.0],
            "k": [0.05, 0.15, 0.45],
        })

    # Use stable column config - don't include dynamic unit in the header
    # This prevents the data_editor from resetting when unit changes
    edited_df = st.data_editor(
        st.session_state.arr_input_df,
        column_config={
            "T_C": st.column_config.NumberColumn(
                "Temperature (°C)", format="%.1f", required=True
            ),
            "k": st.column_config.NumberColumn(
                "Rate Constant (k) (1/day)", format="%.4f", required=True
            ),
        },
        num_rows="dynamic",
        use_container_width=True,
        key="arrhenius_data_editor",
    )

    # Don't update session state automatically - only when Fit button is clicked
    # This prevents refresh issues during data entry

    if st.button("🔬 Fit Arrhenius Model", key="arr_fit_btn"):
        # Update session state with edited data when button is clicked
        st.session_state.arr_input_df = edited_df
        
        # Validate and prepare data
        try:
            df_fit = edited_df.dropna().astype(float)
            temps = df_fit["T_C"].values
            ks = df_fit["k"].values
        except Exception:
            st.error("Invalid data. Please ensure all cells contain numbers.")
            return

        model, r2 = ArrheniusModel.fit_from_data(
            temps_C=temps,
            ks=ks,
            k_unit=k_input_unit,
            name="Fitted from data",
        )
        if model is None:
            st.error("❌ Could not fit Arrhenius model. Check that k > 0, T > 0, and you have ≥2 points.")
            return

        # Save into session_state for use by other tabs
        st.session_state["arrhenius_model"] = model.to_dict()
        st.session_state["arrhenius_model_json"] = model.to_json()

        st.success("✅ Arrhenius model fitted and stored in session.")

        st.markdown("### 📊 Fitting Results")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("ln A", f"{model.lnA:.4f}")
        with c2:
            st.metric("A (1/h)", f"{math.exp(model.lnA):.3e}")
        with c3:
            st.metric("Ea (kJ/mol)", f"{model.Ea_J_mol/1000.0:.3f}")
        with c4:
            st.metric("R² (Fit Quality)", f"{(r2 if r2 is not None else 0):.4f}", 
                     delta="Perfect ✓" if r2 and r2 > 0.99 else "Good" if r2 and r2 > 0.95 else None)

        # Plot ln(k) vs 1/T (using base 1/h k)
        T_K = temps + 273.15
        if k_input_unit.lower().startswith("1/day"):
            ks_base = ks / 24.0
        else:
            ks_base = ks

        x = 1.0 / T_K
        y = np.log(ks_base)
        lnA = model.lnA
        Ea = model.Ea_J_mol
        m = -Ea / R_GAS
        b = lnA
        y_fit = m * x + b

        df_plot = pd.DataFrame(
            {
                "1/T (1/K)": x,
                "ln k (data)": y,
                "ln k (fit)": y_fit,
            }
        )
        fig = px.scatter(
            df_plot,
            x="1/T (1/K)",
            y="ln k (data)",
            title="Arrhenius fit: ln k vs 1/T",
            template="plotly_white",
        )
        fig.update_traces(marker=dict(size=12, color="#2E7D32", line=dict(width=2, color="white")))
        
        # Add fitted line
        line_trace = px.line(df_plot, x="1/T (1/K)", y="ln k (fit)").data[0]
        line_trace.line.color = "#FF6F00"
        line_trace.line.width = 3
        line_trace.name = "Arrhenius fit"
        fig.add_trace(line_trace)
        
        fig.update_layout(
            title_font_size=16,
            title_font_color="#2E7D32",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Download Arrhenius model (JSON)")
        json_str = model.to_json()
        st.download_button(
            label="Download Arrhenius model JSON",
            data=json_str,
            file_name="arrhenius_model.json",
            mime="application/json",
            key="arr_download_btn",
        )

    st.markdown("---")
    st.info("💾 Download your model as JSON to reuse it later or share with others.")
    st.markdown("---")

    # --------------------------------------------------------
    # Section: predict k with current model
    # --------------------------------------------------------
    st.subheader("🔮 Predict Degradation Rate at Any Temperature")
    st.caption("Use your fitted model to estimate the rate constant at any storage temperature")

    current_model = _load_model_from_session()
    if current_model is None:
        st.info("⏳ Fit an Arrhenius model first to enable predictions.")
        return

    c1, c2 = st.columns(2)
    with c1:
        T_pred = st.number_input(
            "🌡️ Temperature for prediction (°C)",
            value=25.0,
            key="arr_T_pred",
        )
    with c2:
        st.markdown("**Output unit**")
        st.caption("Per day (1/day)")
        out_unit = "1/day"

    if st.button("🧮 Predict k", key="arr_predict_btn"):
        k_pred = current_model.predict_k(T_pred, out_unit=out_unit)
        st.metric(
            f"k at {T_pred:.1f} °C ({out_unit})",
            f"{k_pred:.4e}",
        )
