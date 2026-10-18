import numpy as np
import pandas as pd

TARGET = "SOIL_MOISTURE_5"
LABELS = ["is_attacked", "attack_type", "attack_run_id"]


def prepare(df):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(["station", "time"]).reset_index(drop=True)
    if "is_attacked" not in df.columns:
        df["is_attacked"] = False
        df["attack_type"] = "none"
        df["attack_run_id"] = ""
    return df


def pick_start(df, station, n, rng, dry=False, tries=5000):
    """Random valid window start: contiguous hourly data, no NaN, no flag,
    not already attacked. If dry: no rain for 48h before and during, soil not frozen."""
    s = df[df["station"] == station]
    t = s["time"].values
    sm = s[TARGET].to_numpy(float)
    pr = s["P_CALC"].to_numpy(float)
    st = s["SOIL_TEMP_5"].to_numpy(float)
    fl = s["flag_incoherence_verticale"].fillna(0).astype(bool).to_numpy()
    ia = s["is_attacked"].to_numpy(bool)
    lead = 48 if dry else 0
    for _ in range(tries):
        i = int(rng.integers(lead, len(s) - n))
        a, b = i - lead, i + n
        if (t[b - 1] - t[a]) != np.timedelta64(b - 1 - a, "h"):
            continue
        if np.isnan(sm[i:b]).any() or fl[i:b].any() or ia[i:b].any():
            continue
        if dry and (np.isnan(pr[a:b]).any() or pr[a:b].sum() > 0
                    or np.isnan(st[i:b]).any() or st[i:b].min() < 2):
            continue
        return pd.Timestamp(t[i])
    raise RuntimeError("No valid window found: shorten duration or widen the sample")


def _label(df, m, attack_type, run_id):
    df.loc[m, "is_attacked"] = True
    df.loc[m, "attack_type"] = attack_type
    df.loc[m, "attack_run_id"] = run_id


def _event(df, m, p, resolved):
    t = df.loc[m, "time"]
    return {"attack_run_id": p["run_id"], "attack_type": p["attack_type"],
            "station": p["station"], "start": t.min().isoformat(),
            "end": t.max().isoformat(), "n_points": int(m.sum()),
            "seed": int(p["seed"]), "params": {**p, **resolved}}


# ---------------- slow drift ----------------
def slow_drift(df, p, rng):
    df = prepare(df)
    n = int(p["duration_h"])
    start = pd.Timestamp(p["start"]) if p.get("start") else pick_start(df, p["station"], n, rng)
    end = start + pd.Timedelta(hours=n)
    m = ((df["station"] == p["station"]) & (df["time"] >= start)
         & (df["time"] < end) & df[TARGET].notna())
    k = int(m.sum())
    ramp = (np.arange(1, k + 1) / k) ** p.get("power", 1.0)
    delta = p.get("sign", 1) * p["magnitude"] * ramp
    
    release = int(p.get("release_h", 0))      # retour progressif à la normale
    if 0 < release < k:
        delta[-release:] = delta[-release:] * np.linspace(1, 0, release)

    df.loc[m, TARGET] = np.clip(df.loc[m, TARGET].to_numpy() + delta, 0, p.get("vmax", 0.6))
    _label(df, m, "slow_drift", p["run_id"])
    ref = df.loc[(df.station == p["station"]) & ~df.is_attacked, TARGET].diff().abs().median()
    resolved = {"start": start.isoformat(),
                "step_per_h": p["magnitude"] / n, "median_hourly_change": float(ref)}
    return df, df[LABELS], _event(df, m, p, resolved)


# ---------------- context-aware mimicry ----------------
def learn_amplitudes(df, station, rain_mm=5.0, win=12):
    s = df[df["station"] == station]
    sm, pr = s[TARGET].to_numpy(float), s["P_CALC"].to_numpy(float)
    out = []
    for i in np.where(pr >= rain_mm)[0]:
        if i < 1 or i + win >= len(sm):
            continue
        seg = sm[i - 1:i + win]
        if not np.isnan(seg).any():
            out.append(seg.max() - seg[0])
    out = [a for a in out if a > 0.005]
    return np.array(out) if len(out) >= 5 else np.array([0.04, 0.06, 0.08])


def wetting_curve(k, A, rise_h, tau_h):
    t = np.arange(k, dtype=float)
    up = A * (1 - np.exp(-3 * t / rise_h)) / (1 - np.exp(-3))
    down = A * np.exp(-(t - rise_h) / tau_h)
    return np.where(t <= rise_h, up, down)


def mimicry(df, p, rng):
    df = prepare(df)
    n = int(p["duration_h"])
    start = (pd.Timestamp(p["start"]) if p.get("start")
             else pick_start(df, p["station"], n, rng, dry=True))
    end = start + pd.Timedelta(hours=n)
    m = ((df["station"] == p["station"]) & (df["time"] >= start)
         & (df["time"] < end) & df[TARGET].notna())
    k = int(m.sum())
    A = p.get("amplitude", "auto")
    if A == "auto":
        A = float(rng.choice(learn_amplitudes(df, p["station"])))
    rise, tau = p.get("rise_h", 3), p.get("tau_h", 24)
    curve = wetting_curve(k, A, rise, tau)
    df.loc[m, TARGET] = np.clip(df.loc[m, TARGET].to_numpy() + curve, 0, p.get("vmax", 0.6))
    _label(df, m, "mimicry", p["run_id"])
    resolved = {"start": start.isoformat(), "amplitude": A, "rise_h": rise, "tau_h": tau}
    return df, df[LABELS], _event(df, m, p, resolved)


REGISTRY = {"slow_drift": slow_drift, "mimicry": mimicry}


def attack(clean_df, params):
    """Shared interface. Returns attacked_df (USCRN columns + labels), labels, event."""
    rng = np.random.default_rng(params["seed"])
    return REGISTRY[params["attack_type"]](clean_df, params, rng)