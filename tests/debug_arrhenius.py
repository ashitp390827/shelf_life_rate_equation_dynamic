
import numpy as np
import pandas as pd

def check_arrhenius_values():
    # Default values from arrhenius_ui.py
    # i=0: T=25, k=0.05
    # i=1: T=35, k=0.10
    # i=2: T=45, k=0.15
    
    rows = [
        {"T_C": 25.0, "k": 0.05},
        {"T_C": 35.0, "k": 0.10},
        {"T_C": 45.0, "k": 0.15},
    ]
    df = pd.DataFrame(rows)
    
    temps = df["T_C"].values.astype(float)
    ks = df["k"].values.astype(float)
    
    # Logic from arrhenius_ui.py
    T_K = temps + 273.15
    ks_base = ks # assuming 1/h
    
    x = 1.0 / T_K
    y = np.log(ks_base)
    
    print("Data Points:")
    for i in range(len(x)):
        print(f"Point {i}: T={temps[i]} C, k={ks[i]}, 1/T={x[i]:.6f}, ln(k)={y[i]:.4f}")
        
    # Check order on graph (sorted by x)
    sorted_indices = np.argsort(x)
    print("\nPoints sorted by 1/T (Left to Right on Graph):")
    for i in sorted_indices:
        print(f"Point {i}: T={temps[i]} C, k={ks[i]}, 1/T={x[i]:.6f}, ln(k)={y[i]:.4f}")

if __name__ == "__main__":
    check_arrhenius_values()
