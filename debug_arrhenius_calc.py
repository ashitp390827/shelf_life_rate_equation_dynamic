import numpy as np
import math

R_GAS = 8.314

def fit_arrhenius(temps_C, ks):
    T_K = np.array(temps_C) + 273.15
    x = 1.0 / T_K
    y = np.log(ks)
    
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sum((x - x_mean) ** 2)
    
    m = num / den
    b = y_mean - m * x_mean
    
    Ea = -m * R_GAS
    lnA = b
    return Ea, lnA

# Target Ea = 88000 J/mol
target_Ea = 88000
target_A = 1e10
temps_C = [25, 35, 45]
ks = []

print(f"Target Ea: {target_Ea}")
print(f"Temps (C): {temps_C}")

for t in temps_C:
    T_K = t + 273.15
    k = target_A * math.exp(-target_Ea / (R_GAS * T_K))
    ks.append(k)
    print(f"T={t}C, k={k:.4e}")

Ea_calc, lnA_calc = fit_arrhenius(temps_C, ks)
print(f"Calculated Ea: {Ea_calc:.2f} J/mol")
print(f"Calculated Ea (kJ/mol): {Ea_calc/1000:.2f}")

diff = abs(Ea_calc - target_Ea)
print(f"Difference: {diff:.2f} J/mol")
