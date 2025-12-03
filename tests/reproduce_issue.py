
import sys
import os
import numpy as np
import pandas as pd

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dynamic_simulation import DynamicSimulator, KineticsSettings, ArrheniusModel

def test_2nd_order_singularity():
    print("Testing 2nd order singularity...")
    # 2nd order increasing: dC/dt = k * C^2
    # Analytical solution: 1/C = 1/C0 - k*t
    # Singularity when k*t = 1/C0, i.e., t = 1/(k*C0)
    
    # Setup
    A0 = 1.0
    k_val = 1.0
    # Singularity should happen at t = 1.0
    
    # Create a model that gives k=1.0 at 25C (298.15 K)
    # k = A * exp(-Ea/RT) -> 1 = A * exp(-Ea/RT) -> A = exp(Ea/RT)
    # Let's pick Ea = 0 for simplicity, so k = A. Then A = 1.0.
    # lnA = 0.0
    model = ArrheniusModel(lnA=0.0, Ea_J_mol=0.0, k_unit="1/h", name="Test Model")

    kinetics = KineticsSettings(A0=A0, Ai=100.0, order=2, increasing=True)
    sim = DynamicSimulator(model=model, kinetics=kinetics)
    
    # Time steps crossing the singularity
    # t = 0, 0.5, 0.9, 1.1 (should be inf), 1.5 (inf)
    times_h = [0.0, 0.5, 0.9, 1.1, 1.5]
    temps_C = [25.0] * len(times_h)
    
    result = sim.integrate_profile(times_h, temps_C, max_days=1.0, cycles=1)
    
    df = pd.DataFrame(result.history)
    print(df[['time_h', 'concentration']])
    
    # Check if concentration became negative (bug) or huge/inf (correct)
    # At t=1.1, 1/C = 1/1 - 1*1.1 = -0.1 => C = -10.0 (Incorrect behavior for reaction)
    # Correct behavior: C should be inf or capped at max
    
    c_at_1_1 = df.loc[df['time_h'] == 1.1, 'concentration'].values[0]
    print(f"Concentration at t=1.1: {c_at_1_1}")
    
    if c_at_1_1 < 0:
        print("FAIL: Concentration became negative, indicating 2nd order singularity bug.")
    elif c_at_1_1 == float('inf') or c_at_1_1 > 1e6:
        print("PASS: Concentration handled correctly (inf or large).")
    else:
        print(f"UNKNOWN: Concentration is {c_at_1_1}")

if __name__ == "__main__":
    test_2nd_order_singularity()
