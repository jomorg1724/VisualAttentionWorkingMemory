"""Print a compact live view of the current local v2 run (read-only)."""
import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    run_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(json.loads((HERE / "local_run.json").read_text())["run"])
    live = json.loads((run_root / "live_status.json").read_text())
    print("run", run_root.name, "| status", live["status"], "| active", (live.get("active") or {}).get("tag"))
    metrics = run_root / "training" / "metrics.csv"
    if metrics.exists():
        rows = list(csv.DictReader(metrics.open(newline="")))
        if rows:
            last = rows[-1]
            window = rows[-100:]
            print(f"step {last['step']} episodes {last['episodes']} | last-100 loss {sum(float(r['loss']) for r in window)/len(window):.4f} acc {sum(float(r['accuracy']) for r in window)/len(window):.3f} | s/update {sum(float(r['seconds']) for r in window)/len(window):.2f}")
            per_task = {}
            for r in window:
                for task, value in json.loads(r["task_losses"]).items():
                    per_task.setdefault(task, []).append(value)
            print("  last-100 task loss:", {t: round(sum(v) / len(v), 4) for t, v in per_task.items()})
    diagnostics = run_root / "training" / "live_diagnostics.jsonl"
    if diagnostics.exists():
        lines = diagnostics.read_text().splitlines()
        if lines:
            last = json.loads(lines[-1])
            print(f"diagnostics at step {last['step']} ({time.strftime('%H:%M:%S', time.localtime(last['unix']))}):")
            for task, d in last["diagnostics"].items():
                print(
                    f"  {task:22s} priorityH {d['priority_entropy_fraction']:.4f}  far[h0,h1] "
                    f"{d['attention_mass_beyond_3_by_head'][0]:.2e},{d['attention_mass_beyond_3_by_head'][1]:.3f}  "
                    f"evVar {d['evidence_spatial_variance']:.3e} Hvar {d['field_spatial_variance']:.3e}  "
                    f"lambda {[round(x, 3) for x in d['locality_by_head']]}  selW {d['selection_weight_norm']:.3f}"
                )
    for step in live.get("validation_steps", []):
        summary = run_root / f"validation_{step}" / "summary.json"
        if summary.exists():
            s = json.loads(summary.read_text())
            print(f"validation {step}: rank {s['rank']} entropy {{{', '.join(f'{k[:12]}:{v:.4f}' for k, v in s['priority_entropy_fraction_by_task'].items())}}} far {s['attention_mass_beyond_3_by_head']}")
            for task, v in s["task_scores"].items():
                print(f"   {task:22s} nBA {v['normalized_ba']:+.4f} AUC {v['mean_auc']:.4f}")
    if live.get("gate"):
        print("gate:", json.dumps(live["gate"]))


if __name__ == "__main__":
    main()
