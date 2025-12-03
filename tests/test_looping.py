import sys
import os
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dynamic_simulation import integrate_dynamic_profile, DynamicSimulator, KineticsSettings
from arrhenius_ui import ArrheniusModel

def test_auto_looping():
    print("Testing auto-looping in integrate_profile...")
    
    # Create a short 24-hour profile
    times_h = list(range(24))
    temps_C = [25.0] * 24
    
    # Request 48 hours (2 days)
    # This should trigger auto-looping (cycles=2)
    
    # Use a simple model
    model = ArrheniusModel(lnA=20.0, Ea_J_mol=80000.0, k_unit="1/h", name="Test")
    
    # Use the class directly to check internal behavior if needed, 
    # but the wrapper is what we use in tests usually.
    # Let's use the class to inspect the result history length.
    
    kinetics = KineticsSettings(A0=100.0, Ai=50.0, order=1)
    sim = DynamicSimulator(model=model, kinetics=kinetics)
    
    result = sim.integrate_profile(times_h, temps_C, max_days=2.0, cycles=1)
    
    # Check if history covers ~48 hours
    if not result.history:
        print("FAIL: No history returned")
        return
    
    last_time = result.history[-1]["time_h"]
    print(f"Input duration: 24h. Requested: 48h. Result duration: {last_time}h")
    
    if last_time >= 47.0:
        print("PASS: Simulation covered requested duration (auto-looped).")
    else:
        print("FAIL: Simulation stopped early.")

def test_wrapping_logic():
    print("\nTesting wrapping logic (manual simulation)...")
    
    # Simulate the ensemble wrapping logic
    # Data: [10, 20, 30] (3 hours)
    temps_full = np.array([10.0, 20.0, 30.0])
    times_full = np.array([0.0, 1.0, 2.0])
    dt_last = 1.0
    
    # Start at index 1 (value 20)
    start_idx = 1
    
    # Roll
    T_rotated = np.roll(temps_full, -start_idx) # [20, 30, 10]
    
    # Reconstruct time
    dts = np.diff(times_full) # [1, 1]
    dts = np.append(dts, dt_last) # [1, 1, 1]
    dts_rotated = np.roll(dts, -start_idx) # [1, 1, 1]
    
    t_local = np.concatenate(([0.0], np.cumsum(dts_rotated[:-1]))) # [0, 1, 2]
    
    print(f"Original T: {temps_full}")
    print(f"Rotated T (start_idx={start_idx}): {T_rotated}")
    print(f"Reconstructed t: {t_local}")
    
    expected_T = np.array([20.0, 30.0, 10.0])
    if np.allclose(T_rotated, expected_T):
        print("PASS: Temperature rotation correct.")
    else:
        print("FAIL: Temperature rotation incorrect.")

if __name__ == "__main__":
    test_auto_looping()
    test_wrapping_logic()
