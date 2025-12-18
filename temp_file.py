import numpy as np
import math
from scipy import stats

# Example measured k (per day) at different °C (replace with your data)
# Each row: [temp_C, k_value]
data = np.array([
    [4.0, 0.0023],
    [15.0, 0.0078],
    [25.0, 0.0234],
    [35.0, 0.0672],
])

R = 8.314  # J/mol/K

temps_C = data[:,0]
kvals = data[:,1]

T_K = temps_C + 273.15
x = 1.0 / T_K
y = np.log(kvals)

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
Ea = -slope * R          # in J/mol
A = math.exp(intercept)

print(f"Arrhenius fit: ln(k) = {intercept:.4f} + ({slope:.4f})*(1/T)")
print(f"Pre-exponential A = {A:.4e} units-of-k")
print(f"Activation energy Ea = {Ea/1000:.2f} kJ/mol")
print(f"R^2 = {r_value**2:.4f}")

# Example: predict k at 25°C and compute shelf-life for a criterion
Tpred = 25.0 + 273.15
k_pred = A * math.exp(-Ea/(R*Tpred))
print(f"k(25°C) = {k_pred:.5e} per day")

# For zero order shelf life:
C0 = 100.0
Ccrit = 80.0
t_shelf_zero = (C0 - Ccrit) / k_pred
print(f"Zero-order shelf life at 25°C = {t_shelf_zero:.2f} days")

# For first order shelf life:
t_shelf_first = (1.0 / k_pred) * math.log(C0 / Ccrit)
print(f"First-order shelf life at 25°C = {t_shelf_first:.2f} days")


