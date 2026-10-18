import json, hashlib
from pathlib import Path
from attacks import TARGET


def config_hash(cfg):
    b = json.dumps(cfg, sort_keys=True, default=str).encode()
    return "sha256:" + hashlib.sha256(b).hexdigest()


def write_all(clean, attacked, events, cfg, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    attacked.to_parquet(out / "attacked.parquet", index=False)

    with open(out / "events.jsonl", "w") as f:
        for e in events:
            f.write(json.dumps(e, default=str) + "\n")

    seeds = {e["attack_run_id"]: e["seed"] for e in events}
    rows = attacked[attacked["is_attacked"]]
    orig = clean.loc[rows.index, TARGET].to_numpy()
    new = rows[TARGET].to_numpy()
    attacks = [{
        "station_id": r.station,
        "timestamp": r.time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "original_value": round(float(o), 4),
        "attacked_value": round(float(a), 4),
        "attack_type": r.attack_type,
        "attack_magnitude": round(float(a - o), 4),
        "seed": seeds[r.attack_run_id],
    } for r, o, a in zip(rows.itertuples(index=False), orig, new)]

    doc = {"config_hash": config_hash(cfg), "tier": cfg.get("tier", 2), "attacks": attacks}
    with open(out / "attack_output.json", "w") as f:
        json.dump(doc, f, indent=2)