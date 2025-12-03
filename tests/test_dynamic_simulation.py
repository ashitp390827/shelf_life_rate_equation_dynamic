import math
import numpy as np
from dynamic_simulation import integrate_dynamic_profile


def constant_k_predictor(k):
    return lambda model, T: float(k)


def test_first_order_constant_k():
    # first-order kinetics: A(t)=A0*exp(-k t)
    k = 0.1  # per hour
    A0 = 100.0
    Ai = 50.0
    temps = [20.0] * 100
    times = list(range(len(temps)))

    history, status = integrate_dynamic_profile(
        temps_C=temps,
        times_h=times,
        A0=A0,
        Ai=Ai,
        arr_model=None,
        predict_k_from_arrhenius=constant_k_predictor(k),
        order=1,
        max_days=10,
        cycles=1
    )

    assert status == "Reached critical limit"
    # continuous solution: t = ln(A0/Ai)/k
    t_cont = math.log(A0 / Ai) / k
    # find reported time when concentration <= Ai
    idx = next((i for i, h in enumerate(history) if h['concentration'] <= Ai), None)
    assert idx is not None
    t_report = history[idx]['time']
    # Should be within 1 hour of continuous solution (discrete steps)
    assert abs(t_report - t_cont) <= 1.0


def test_zero_order_constant_k():
    # zero-order: A(t)=A0 - k t
    k = 2.0  # per hour
    A0 = 100.0
    Ai = 50.0
    temps = [25.0] * 200
    times = list(range(len(temps)))

    history, status = integrate_dynamic_profile(
        temps_C=temps,
        times_h=times,
        A0=A0,
        Ai=Ai,
        arr_model=None,
        predict_k_from_arrhenius=constant_k_predictor(k),
        order=0,
        max_days=20,
        cycles=1
    )

    assert status == "Reached critical limit"
    t_cont = (A0 - Ai) / k
    idx = next((i for i, h in enumerate(history) if h['concentration'] <= Ai), None)
    assert idx is not None
    t_report = history[idx]['time']
    assert abs(t_report - t_cont) <= 1e-6


def test_second_order_constant_k():
    # second-order: 1/A = 1/A0 + k t  => t = (1/Ai - 1/A0)/k
    k = 0.001  # per hour
    A0 = 100.0
    Ai = 50.0
    temps = [10.0] * 200
    times = list(range(len(temps)))

    history, status = integrate_dynamic_profile(
        temps_C=temps,
        times_h=times,
        A0=A0,
        Ai=Ai,
        arr_model=None,
        predict_k_from_arrhenius=constant_k_predictor(k),
        order=2,
        max_days=10,
        cycles=1
    )

    assert status == "Reached critical limit"
    t_cont = ((1.0 / Ai) - (1.0 / A0)) / k
    idx = next((i for i, h in enumerate(history) if h['concentration'] <= Ai), None)
    assert idx is not None
    t_report = history[idx]['time']
    assert abs(t_report - t_cont) <= 1.0
