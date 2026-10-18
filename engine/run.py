import sys
import yaml
import pandas as pd
from attacks import attack, prepare
from outputs import write_all


def to_ts(value, series):
    """Timestamp compatible avec la colonne time (gère le cas tz-aware)."""
    ts = pd.Timestamp(value)
    tz = series.dt.tz
    if tz is not None and ts.tzinfo is None:
        ts = ts.tz_localize(tz)
    return ts


cfg = yaml.safe_load(open(sys.argv[1]))
clean = prepare(pd.read_parquet(cfg["data_path"]))

# ---- 1. Restriction à la période (2024) et aux stations ----
per = cfg["period"]
p_start = to_ts(per["start"], clean["time"])
p_end = to_ts(per["end"], clean["time"])

mask = (clean["time"] >= p_start) & (clean["time"] < p_end)
if per.get("stations"):
    mask &= clean["station"].isin(per["stations"])
clean = clean[mask].reset_index(drop=True)
assert len(clean) > 0, "Aucune donnée dans la période/stations demandées"

# ---- 2. Vérifier que les runs visent les stations chargées ----
loaded = set(clean["station"].unique())
for run in cfg["runs"]:
    assert run["station"] in loaded, f'Station {run["station"]} absente des données filtrées'
    if run.get("start"):
        s = to_ts(run["start"], clean["time"])
        assert p_start <= s < p_end, f'{run["run_id"]}: start hors période {per["start"]} -> {per["end"]}'

# ---- 3. Génération des attaques ----
attacked, events = clean.copy(), []
for run in cfg["runs"]:
    attacked, labels, ev = attack(attacked, run)
    assert ev["n_points"] > 0, f'{run["run_id"]}: aucune ligne attaquée'
    events.append(ev)

# ---- 4. Garde-fou : aucune ligne attaquée hors période ----
t = attacked.loc[attacked["is_attacked"], "time"]
assert t.between(p_start, p_end - pd.Timedelta(hours=1)).all(), \
    "Une attaque sort de la période autorisée"

# ---- 5. Écriture des sorties ----
write_all(clean, attacked, events, cfg, cfg["output_dir"])
print(f"{int(attacked.is_attacked.sum())} lignes attaquées, {len(events)} événements "
      f"({p_start.date()} -> {p_end.date()})")