# scripts/plot_runs.py
import json, pandas as pd, matplotlib.pyplot as plt
out = "outputs/sample_001"
att = pd.read_parquet(f"{out}/attacked.parquet")
clean = att.copy()
doc = json.load(open(f"{out}/attack_output.json"))
orig = pd.DataFrame(doc["attacks"]).set_index("timestamp")["original_value"]
ev = [json.loads(l) for l in open(f"{out}/events.jsonl")]
for e in ev:
    a, b = pd.Timestamp(e["start"]), pd.Timestamp(e["end"])
    w = att[(att.time >= a - pd.Timedelta(hours=48)) & (att.time <= b + pd.Timedelta(hours=48))]
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
    ax[0].plot(w.time, w["SOIL_MOISTURE_5"], label="attacked")
    ax[0].axvspan(a, b, alpha=0.2, color="red")
    o = w[w.is_attacked]
    ax[0].plot(o.time, o.time.dt.strftime("%Y-%m-%dT%H:%M:%SZ").map(orig), "--", label="original")
    ax[0].legend(); ax[0].set_ylabel("SM 5 cm")
    ax[1].bar(w.time, w["P_CALC"], width=0.04); ax[1].set_ylabel("P_CALC")
    fig.suptitle(f'{e["attack_type"]} ({e["attack_run_id"]})')
    fig.savefig(f'{out}/{e["attack_run_id"]}.png', dpi=150)