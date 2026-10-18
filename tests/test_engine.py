import sys; sys.path.insert(0, "engine")
import numpy as np, pandas as pd
from attacks import attack, prepare, TARGET

def make_clean(n=24*30):
    t = pd.date_range("2023-06-01", periods=n, freq="h")
    return prepare(pd.DataFrame({
        "time": t, "station": "S1", "SOIL_MOISTURE_5": 0.2 + 0.001*np.sin(np.arange(n)/10),
        "P_CALC": 0.0, "SOIL_TEMP_5": 15.0, "flag_incoherence_verticale": 0}))

P = {"attack_type": "slow_drift", "run_id": "d1", "station": "S1",
     "duration_h": 100, "magnitude": 0.05, "seed": 1}

def test_unattacked_rows_unchanged():
    c = make_clean(); a, _, _ = attack(c, P)
    assert (a.loc[~a.is_attacked, TARGET] == c.loc[~a.is_attacked, TARGET]).all()

def test_deterministic():
    c = make_clean()
    assert attack(c, P)[0].equals(attack(c, P)[0])

def test_mimicry_runs():
    c = make_clean()
    p = {**P, "attack_type": "mimicry", "run_id": "m1", "duration_h": 48,
         "amplitude": 0.05}
    a, _, ev = attack(c, p)
    assert a.is_attacked.sum() == 48