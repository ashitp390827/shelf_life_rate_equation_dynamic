import streamlit as st

"""
Landing page module for Shelf Life Prediction Application.
Contains the home/welcome tab with scientific principles and usage guidance.
"""

def landing_page_tab():
    """Display landing page with scientific principles and workflow guide."""

    st.markdown("## 🎯 Welcome to Shelf Life Prediction Application")
    st.markdown(
        """
        This application provides a complete workflow for predicting product shelf life using **kinetic modelling**.

        - Food spoilage and quality-loss reactions follow **chemical rate equations**.  
        - By determining the **rate constant (k)** at a given temperature, it is possible to estimate how long it takes for a component (e.g., Vitamin C) to reach a critical concentration (shelf life).  
        - In real-world storage, temperature fluctuates, causing **k** to change over time.  
        - To address this, the application uses the **Arrhenius model** to describe how k varies with temperature.

        In the **first tab**, users can calculate the rate constant at a particular temperature by entering concentration-vs-time data. The app automatically identifies the correct **reaction order**.

        After obtaining k values at multiple temperatures, users can fit the **Arrhenius model**, and then run a **dynamic simulation** that combines temperature variations and kinetics to predict realistic shelf life.
        """
    )

    st.markdown(
        """
        Suitable for food, pharmaceutical, cosmetic, and other perishable products.
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**📊 Data Analysis:** Fit degradation models from experimental data.")
    with col2:
        st.info("**🌡️ Temperature Effects:** Model how changing temperature alters reaction rates.")
    with col3:
        st.info("**🔮 Predictions:** Simulate realistic shelf life using dynamic conditions.")

    st.markdown("---")

    # SCIENTIFIC PRINCIPLES

    st.markdown("---")
    # SCIENTIFIC PRINCIPLES (continue original)
    with st.expander("📚 **Scientific Principles** — Understanding the Science", expanded=True):

        st.markdown(
            """
            ### 🧪 Chemical Kinetics and Food Spoilage

            Many food spoilage and quality-loss reactions follow **chemical rate equations**, such as:
            - Nutrient degradation (e.g., Vitamin C)
            - Lipid oxidation
            - Color loss
            - Active ingredient breakdown

            These processes can be described using the general rate equation:

            $$\\frac{dA}{dt} = -k A^n$$

            Where:
            - **A** = Concentration or quality attribute
            - **k** = Rate constant (temperature‑dependent)
            - **n** = Reaction order (0, 1, or 2)
            """
        )

        st.markdown("---")

        st.markdown(
            """
            ### ⏱️ Shelf Life from Rate Constant

            Once the rate constant **k** is determined at a particular temperature, the **shelf life** can be
            calculated using integrated rate laws:
            """
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Zero‑Order:** $$t = \\frac{A_0 - A_i}{k}$$")
        with col2:
            st.markdown("**First‑Order:** $$t = \\frac{1}{k}\ln\left(\\frac{A_0}{A_i}\\right)$$")
        with col3:
            st.markdown("**Second‑Order:** $$t = \\frac{1}{k}\left(\\frac{1}{A_i} - \\frac{1}{A_0}\\right)$$")

        st.markdown("---")

        st.markdown("### 🧮 How k Is Calculated From Experimental Data")

        st.markdown(
            """
            Determining the **rate constant (k)** is the foundation of kinetic shelf-life prediction.
            In practice, k is extracted by fitting  degradation data to **integrated rate laws**.

            The app automatically tests **zero-order, first-order, and second-order** kinetics and selects the best model.
            """
        )

        # Better table-like explanation
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                """
                #### **Zero-Order**
                Integrated form:  
                $$A = A_0 - k t$$  
                Plot: **A vs t**  
                Slope = **–k**  
                """
            )
        with col2:
            st.markdown(
                """
                #### **First-Order**
                Integrated form:  
                $$\\ln(A) = \\ln(A_0) - k t$$  
                Plot: **ln(A) vs t**  
                Slope = **–k**
                """
            )
        with col3:
            st.markdown(
                """
                #### **Second-Order**
                Integrated form:  
                $$\\frac{1}{A} = \\frac{1}{A_0} + k t$$  
                Plot: **1/A vs t**  
                Slope = **+k**
                """
            )



        st.markdown("---")

        st.markdown(
            """
            ### ⚠️ Minimum 30% Degradation Rule

            To reliably determine **k** and reaction order:

            - The data must show **at least 30% degradation** from the initial concentration.
            - This improves accuracy, reduces noise interference, and avoids incorrect model fitting.
            """
        )

        st.warning("Collect **6–10 time points** and ensure **≥30% degradation** for reliable fitting.")
        st.markdown(
            """
            ### ⚠️ Why k Becomes Unreliable Below 30% Change
            - Analytical instrument noise (±0.5–2%) becomes **larger than the actual change**  
            - A small error (e.g., ±1%) can flip the model between **0th, 1st, and 2nd order**  
            - k may vary **5–20×** depending on noise, not chemistry  
            - ln(A) and 1/A transformations amplify errors  
            - Regression slope approaches zero → **huge uncertainty**

            In real studies, this produces:
            - wrong reaction order  
            - k values with ±50–200% uncertainty  
            - meaningless Arrhenius plots

            This is why the **≥30% degradation rule** is the scientific standard for reliable kinetic modeling."""

        )

        st.markdown("### ⚠️ Analytical Precision & Prediction Error Guide")


        st.markdown(
            """
            The accuracy of kinetic predictions depends on **analytical precision** and the **percentage change** in the
            monitored attribute.

            **Key points:**
            - Lower precision instruments require **larger percentage changes** for acceptable accuracy.
            - <10% observed change can lead to **very unreliable** estimates.
            - ≥20–30% change is recommended for good kinetic analysis.

            Below is the analytical precision table (from the user guide) showing expected **prediction error (%)** based on
            instrument precision and observed percentage change."""
            


        )

        import pandas as pd

        ERROR_TABLE = {
            0.1: {1: 14, 5: 2.8, 10: 1.4, 20: 0.7, 30: 0.5, 40: 0.4, 50: 0.3},
            0.5: {1: 70, 5: 14, 10: 7, 20: 3.5, 30: 2.5, 40: 2.0, 50: 1.5},
            1.0: {1: 100, 5: 28, 10: 14, 20: 7, 30: 5, 40: 4, 50: 3},
            2.0: {1: 100, 5: 56, 10: 28, 20: 14, 30: 10, 40: 8, 50: 6},
            5.0: {1: 100, 5: 100, 10: 70, 20: 35, 30: 25, 40: 20, 50: 15},
            10.0: {1: 100, 5: 100, 10: 100, 20: 70, 30: 50, 40: 40, 50: 30}
        }

        precision_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        change_values = [1, 5, 10, 20, 30, 40, 50]

        table_data = []
        for prec in precision_values:
            row = {'Analytical Precision ±%': f'±{prec}%'}
            for change in change_values:
                if change in ERROR_TABLE.get(prec, {}):
                    val = ERROR_TABLE[prec][change]
                    row[f'{change}% Change'] = " >100%" if val == 100 else f"{val}%"
                else:
                    row[f'{change}% Change'] = ">100%"
            table_data.append(row)

        guide_df = pd.DataFrame(table_data)
        st.dataframe(guide_df, width='stretch', hide_index=True)

        st.markdown(
            """
            ### 💡 Interpretation
            - **≥30% change + high precision (±0.1–0.5%) → Excellent accuracy**
            - **≥20% change + moderate precision (±1–2%) → Acceptable accuracy**
            - **<10% change → High risk of incorrect reaction order or wrong k**
            """
        )
        st.markdown("---")

        st.markdown(
            """
            ### 🌡️ Temperature Is Dynamic in Real Life

            Storage and distribution temperatures fluctuate due to:
            - Transport and logistics
            - Retail and display conditions
            - Seasonal variation
            - Consumer handling

            Because the rate constant **k changes with temperature**, constant‑temperature shelf life
            predictions are often unrealistic.
            """
        )

        st.markdown("---")

        st.markdown(
            """
            ### 📈 The Arrhenius Equation

            The **Arrhenius model** is used to estimate how **k varies with temperature**:

            $$k = A e^{-E_a/(RT)}$$

            - **A** = Pre‑exponential factor
            - **Eₐ** = Activation energy
            - **R** = Gas constant (8.314 J/mol·K)
            - **T** = Temperature (K)

            By fitting **k values at multiple temperatures**, the app can predict **k at any temperature**.
            """
        )

        st.success("Arrhenius modeling enables temperature‑corrected, realistic shelf life prediction.")

    st.markdown("---")

    # WORKFLOW
    with st.expander("🔄 **How to Use This Application** — Step‑by‑Step Workflow", expanded=True):

        st.markdown("### Step 1️⃣: Determine Rate Constant (Tab 2 — Degradation Analysis)")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(
                """
                - Upload degradation data (A vs time)
                - App fits 0th, 1st, and 2nd order models
                - Determines **k** and reaction order

                ⚠️ Ensure ≥30% degradation for reliable fitting.
                """
            )
        with c2:
            st.info("📍 Go to **Tab 2** — Example datasets available.")

        st.markdown("---")

        st.markdown("### Step 2️⃣: Build Arrhenius Model (Tab 3 — Temperature Modeling)")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(
                """
                - Enter k values at 3–4 temperatures ( give k value is expected temperature ranges)
                - Fit Arrhenius equation → obtain **Eₐ** and **A**
                - Predict k at any temperature
                """
            )
        with c2:
            st.info("📍 Go to **Tab 3** — Requires rate constants from Step 1.")

        st.markdown("---")

        st.markdown("### Step 3️⃣: Dynamic Simulation (Tab 4 — Realistic Shelf Life)")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(
                """
                - Combine Arrhenius model with real temperature data
                - Simulate degradation hour‑by‑hour
                - Predict realistic shelf life (best, worst, average)
                """
            )
        with c2:
            st.success("📍 Go to **Tab 4** — Most accurate and realistic predictions.")

        st.markdown("---")

        st.markdown(
            """
            ### Quick Tools
            - **Tab 5 — Q10 Approximation:** Fast scaling when detailed data is unavailable
            - **Tab 6 — Quick Calculator:** For direct shelf life using known k
            """
        )

    st.markdown("---")

    # BEST PRACTICES
    with st.expander("✅ **Best Practices**", expanded=False):
        st.markdown(
            """
            **For Accurate Kinetic Modeling:**
            - Use **6–10 data points**
            - Ensure **≥30% degradation**
            - Maintain **constant temperature (±1°C)** during experiments
            - Use **≥3 temperatures** for Arrhenius fitting

            **Model Validation:**
            - R² > 0.95 preferred
            - Check residuals for randomness
            - Validate with new data where possible
            """
        )

    st.markdown("---")

    # QUICK START
    with st.expander("🚀 **Quick Start Guide**", expanded=False):
        st.markdown(
            """
            1. Tab 2 → Analyze example degradation data
            2. Tab 3 → Fit Arrhenius model
            3. Tab 4 → Run dynamic simulation
            """
        )

    st.markdown("---")

    # FOOTER
    st.markdown(
        """
        ### 📖 References & Further Reading
        - Chemical kinetics in food systems
        - Arrhenius modeling for shelf life
        - Accelerated shelf life testing

        **Start Now:** Open **Tab 2** to begin the analysis! 🚀
        """
    )
