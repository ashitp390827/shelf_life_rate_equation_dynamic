# Shelf-life Predictor — How it works

This document explains how the Shelf-life Predictor app  is structured, how it processes data, the scientific principles and equations used (kinetic models and Arrhenius/ASLT), and important assumptions and limitations.

**Overview**
- Purpose: fit simple kinetic models (zero, first, second order) to an attribute A measured over time (eg. concetration of Vitamin C), estimate rate constants, compute shelf life for a user-specified critical limit, and provide Arrhenius/ASLT tools for temperature extrapolation (including Q10).


**User workflow / UI flow**
1. **Kinetic analysis tab** (at a constant temperature):
   - Input time-varying data for attribute `A` (e.g., concentration of Vitamin C) vs `time` either by editing the shown table or by pasting tab/space/comma-separated text.
   - Click **Fit model** to validate and commit values for analysis.
   - Choose model selection mode (`Auto` or pick `Zero`, `First`, `Second`) and set analytical precision, critical limit (Ai), and time unit.
   - The app fits all three kinetic models, shows R², highlights the chosen/best model, computes shelf life, plots data and linearized forms, and provides an error/precision guide.
   - **Negative shelf-life handling**: If the attribute increases over time (or fitted trend moves away from the critical limit), the app detects this and displays a clear message: "Critical limit already exceeded or fitted trend moves away from the limit; shelf life is not meaningful."

2. **Arrhenius / ASLT tab**:
   - Enter temperature (°C) vs rate constant (`k`) pairs, apply the table, and fit an Arrhenius model to estimate activation energy (Ea) and pre-exponential factor (A). (You can use the Kinetic analysis tab to find the k value for each temperature.)
   - **Compute Q10 / Acceleration factor**: Use the calculator section to compute k-ratio (acceleration factor) and generalized Q10 between any two temperatures using the fitted Arrhenius model.
   - Results are automatically available for use in the Shelf-life tools tab.

3. **Shelf-life tools tab**:
   - **Arrhenius calculator**: If an Arrhenius model is available, compute k at two different temperatures and their acceleration factor / Q10 ratio.
   - **Shelf-life projection using Q10**: Choose between two methods:
     - **Enter known Q10 value**: Manually provide a Q10 value (e.g., from literature or prior experiments).
     - **Compute from two known temperatures**: Use the fitted Arrhenius model to compute k(T₁) and k(T₂), then automatically calculate the acceleration factor and generalized Q10. The app uses the computed Q10 for shelf-life projection.
   - **Correct Q10 shelf-life formula**: Shelf life scales *inversely* with the rate multiplier. If Q10 = 2 and you go 10°C cooler, the shelf life doubles: $t_{target} = \dfrac{t_{accel}}{Q10^{(T_{target} - T_{accel})/10}}$.
   - **Shelf life using kinetic equations directly**: Enter k, A₀, and Ai to compute shelf life for any reaction order (zero, first, or second).

4. **How the App Works tab**:
   - Displays this markdown document with all scientific principles and usage guidance.

**Data parsing & expected formats**
- Editable table: streamlit `data_editor` with columns `time` and `A`.
- Paste box: free text where each line contains two numeric columns (time, attribute). Separators allowed: whitespace, comma, semicolon, or tab. A header line is allowed and automatically skipped if detected.
- Multi-temperature paste: blocks separated by blank lines may contain a leading temperature label (e.g., `Temp: 25`) followed by the table; the parser attempts to detect temperature lines and then parse the following numeric rows.

**Kinetic models and fitting method**
The app fits three simple integrated rate laws. All linear fits are performed using ordinary least-squares (via `scipy.stats.linregress`) on the appropriate linearized form. The code then converts slope/intercept back to the kinetic parameters.

- Zero-order kinetics
  - Differential: $\dfrac{d[A]}{dt} = -k$.
  - Integrated: $A(t) = A_0 - k\,t$.
  - Fit performed on $A$ vs $t$ (linear regression). If slope = m and intercept = b from the fit, then:
    - $k = -m$
    - $A_0 = b$
  - Shelf-life (time to reach critical limit $A_i$):
    $$t = \frac{A_0 - A_i}{k}$$

- First-order kinetics
  - Differential: $\dfrac{d[A]}{dt} = -k\,[A]$.
  - Integrated: $\ln A = \ln A_0 - k\,t$.
  - Fit performed on $\ln A$ vs $t$ (only for positive A values). If slope = m and intercept = b,
    - $k = -m$
    - $A_0 = e^{b}$
  - Shelf-life:
    $$t = \frac{\ln A_0 - \ln A_i}{k}$$

- Second-order kinetics
  - Differential: $\dfrac{d[A]}{dt} = -k\,[A]^2$.
  - Integrated: $\dfrac{1}{A} = \dfrac{1}{A_0} + k\,t$.
  - Fit performed on $1/A$ vs $t$ (only for positive A values). If slope = m and intercept = b,
    - $k = m$
    - $A_0 = 1 / b$ (if intercept ≠ 0)
  - Shelf-life:
    $$t = \frac{1/A_i - 1/A_0}{k}$$

Notes about fitting:
- The app uses the experimental A0 (value at the minimum time, usually t=0) for shelf-life calculations rather than the fitted A0. This reduces bias when users provide an experimental initial measurement.
- Linear regression results are converted to R² by squaring the returned correlation coefficient: $R^2 = r^2$.
- When data contain non-positive values where logarithms or reciprocals are required, those rows are excluded for that particular linearization.

**Arrhenius modelling (ASLT)**
- Fundamental equation: $\ln k = \ln A - \dfrac{E_a}{R}\left(\dfrac{1}{T}\right)$ where $T$ is temperature in kelvin.
- Implementation: fit $\ln k$ versus $1/T$ (with $T = T_{°C} + 273.15$) using linear regression.
  - Slope = $-E_a/R$ → $E_a = -\text{slope} \times R$ (gas constant $R = 8.314\,$J/mol·K)
  - Intercept = $\ln A$ (pre-exponential factor)
- Prediction: given a fitted model (Ea/R and lnA), predict $k$ at any $T$ by computing $\ln k$ and exponentiating.

**Q10 calculation and shelf-life projection**
- **Q10 definition**: The rate multiplier when temperature increases by 10°C. If Q10 = 2, then k(T+10) = 2 × k(T).
- **Generalized Q10**: For any temperature gap, compute: $Q_{10} = \left(\dfrac{k(T_2)}{k(T_1)}\right)^{10/(T_2-T_1)}$.
- **Acceleration factor**: The ratio $\dfrac{k(T_2)}{k(T_1)}$ tells you how much faster the reaction proceeds at T₂ vs T₁.
- **Shelf-life with Q10** (corrected formula):
  - Given shelf life $t_{accel}$ at accelerated temperature $T_{accel}$, predict shelf life at target temperature $T_{target}$:
  - $$t_{target} = \frac{t_{accel}}{Q10^{(T_{target} - T_{accel})/10}}$$
  - Since reaction slows at lower temperatures (lower k), shelf life *increases* (longer product lasts).
  - Example: If shelf life = 30 days at 30°C with Q10 = 2, then at 20°C: $t = 30 / (2^{(20-30)/10}) = 30 / 0.5 = 60$ days.
- **Two ways to specify Q10 in shelf-life projection**:
  1. **Known Q10 value**: Enter the Q10 directly (from literature, prior experiments, or from the Arrhenius / ASLT tab).
  2. **Compute from Arrhenius model**: Specify two temperatures (T₁, T₂); the app uses the fitted Arrhenius model to compute k values and calculate the generalized Q10 automatically.

**Analytical precision and prediction error guide**
- The app contains a reference table (`ERROR_TABLE`) mapping analytical precision (±%) and observed percent change to expected percent error in predicted shelf life. This table is used by `estimate_prediction_error` to provide an approximate expected prediction error and a short interpretation note.


**Assumptions, edge cases, and limitations**
- **Kinetic models**: Simple (0th/1st/2nd order) and assume homogenous reaction-like behaviour — real shelf-life mechanisms can be more complex.
- **Linear regression fitting**: The app uses ordinary least-squares (no weighted or non-linear least squares). This is appropriate for the chosen linearized forms but has limits when residuals are non-normal or heteroskedastic.
- **Data handling**: The first- and second-order fits exclude non-positive A values (log and reciprocal are undefined). Such exclusions reduce data available for fitting.
- **Negative shelf-life prevention**: If computed shelf life is negative (trend moves away from threshold), the app returns NaN and displays a clear message explaining why shelf life is not meaningful.
- **Error estimation**: The prediction error guidance is approximate (based on a lookup table) and not a rigorous statistical confidence interval.
- **Arrhenius extrapolation**: Assumes a single activation energy over the studied temperature range; real systems may show curvature or multiple regimes. Extrapolating far beyond the fitted temperature range may introduce significant error.
- **Q10 interpretation**: Q10 values > 1 indicate faster degradation at higher temperatures. Very high Q10 values (e.g., Q10 > 4) are rare but possible for enzyme-catalyzed or highly temperature-sensitive reactions.

**Troubleshooting / common messages**
- "Need at least 2 valid rows": provide at least two numeric rows of (time, A).
- "k values must be positive": for Arrhenius fitting, all provided k values must be > 0.
- Low R² warnings: consider collecting more data or monitoring for a longer time range.


**Contact / Notes**
- contact at ashitp02@gmail.com.
