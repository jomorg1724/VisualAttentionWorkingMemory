"""Mirror live metrics, diagnostics and immutable artifacts of the v2 cloud arm every 45 s."""
import csv
import hashlib
import io
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(config_path):
    config = read(config_path)
    arm = config["arm"]
    root = Path(config["local_run"])
    remote_results = config["remote_results"]
    deadline = float(config["deadline_unix"])
    interval = 45
    common = ["-i", config["ssh_identity"], "-o", "UserKnownHostsFile=" + config["known_hosts"], "-o", "StrictHostKeyChecking=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "KexAlgorithms=curve25519-sha256"]
    destination = f'{config["ssh_user"]}@{config["ssh_host"]}'
    ssh = ["ssh", "-n", *common, "-p", str(config["ssh_port"]), destination]
    incremental = root / "incremental"
    incremental.mkdir(parents=True, exist_ok=True)
    transfers_path = root / "incremental_transfers.json"
    transfers = read(transfers_path) if transfers_path.exists() else []
    mirrored = {row["remote"]: row["sha256"] for row in transfers}

    def remote_json():
        code = f"""
import json,pathlib,subprocess
r=pathlib.Path({remote_results!r}); arm={arm!r}
def load(p):
  return json.loads(p.read_text()) if p.is_file() else {{}}
b=load(r/'budget.json'); a=load(r/'aggregate.json')
m=r/arm/'training/metrics.csv'
lines=m.read_text().splitlines() if m.is_file() else []
d=r/arm/'training/live_diagnostics.jsonl'
diag=d.read_text().splitlines()[-2:] if d.is_file() else []
idx=r/arm/'training/checkpoint_index.jsonl'
checkpoints=[json.loads(x) for x in idx.read_text().splitlines()] if idx.is_file() else []
validations=[]
for p in sorted(r.glob('validation_*_result.json')):
 v=json.loads(p.read_text()); validations.append(dict(file=p.name,step=v.get('step'),status=v.get('status'),rank=v.get('rank'),task_scores=v.get('task_scores'),priority_entropy_fraction_by_task=v.get('priority_entropy_fraction_by_task'),attention_mass_beyond_3_by_head=v.get('attention_mass_beyond_3_by_head'),locality_by_head=v.get('locality_by_head')))
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True).strip()
print(json.dumps(dict(budget=b,aggregate={{k:v for k,v in a.items() if k!='run'}},run=a.get('run',{{}}).get(arm,{{}}),metrics=([lines[0]]+lines[1:][-256:] if lines else []),total_rows=max(0,len(lines)-1),diagnostics=[json.loads(x) for x in diag],checkpoints=checkpoints,validations=validations,gpu=gpu,exited=(r/'exit.json').is_file())))
"""
        result = subprocess.run([*ssh, "python -c " + shlex.quote(code)], capture_output=True, text=True, timeout=45, check=True)
        return json.loads(result.stdout)

    def transfer(relative, expected_sha):
        relative = relative.replace("\\", "/")
        if not re.fullmatch(r"((profile|" + re.escape(arm) + r"/(validation_[0-9]+|selected_test|terminal_test))/[A-Za-z0-9_.-]+|validation_[0-9]+_(job|result)\.json|validation_[0-9]+\.log|(selected|terminal)_test(_result|_job)?\.(json|log))", relative):
            raise RuntimeError("Unexpected remote artifact " + relative)
        local = incremental / Path(relative)
        if local.is_file() and sha(local) == expected_sha:
            return
        local.parent.mkdir(parents=True, exist_ok=True)
        temporary = local.with_suffix(local.suffix + ".download")
        subprocess.run(["scp", *common, "-P", str(config["ssh_port"]), destination + ":" + remote_results + "/" + relative, str(temporary)], capture_output=True, text=True, timeout=900, check=True)
        if sha(temporary) != expected_sha:
            raise RuntimeError("Transfer hash mismatch " + relative)
        temporary.replace(local)
        transfers.append(dict(remote=relative, local=str(local), bytes=local.stat().st_size, sha256=expected_sha, mirrored_unix=time.time()))
        mirrored[relative] = expected_sha
        write(transfers_path, transfers)

    failures = 0
    while time.time() < deadline + 900:
        try:
            payload = remote_json()
            now = time.time()
            rows = list(csv.DictReader(io.StringIO("\n".join(payload["metrics"]) + "\n"))) if payload["metrics"] else []
            rolling = {}
            for size in (20, 100, 256):
                selected = rows[-min(size, len(rows)):] if rows else []
                if selected:
                    rolling[str(size)] = dict(
                        updates=len(selected), step_start=int(selected[0]["step"]), step_end=int(selected[-1]["step"]),
                        mean_loss=sum(float(r["loss"]) for r in selected) / len(selected),
                        mean_accuracy=sum(float(r["accuracy"]) for r in selected) / len(selected),
                        mean_gradient_norm=sum(float(r["gradient_norm"]) for r in selected) / len(selected),
                        mean_update_seconds=sum(float(r["seconds"]) for r in selected) / len(selected),
                    )
                    per_task = {}
                    for r in selected:
                        for task, value in json.loads(r["task_losses"]).items():
                            per_task.setdefault(task, []).append(value)
                    rolling[str(size)]["task_losses"] = {t: sum(v) / len(v) for t, v in per_task.items()}
            latest = {k: (int(rows[-1][k]) if k in ("step", "episodes", "frames_total") else float(rows[-1][k])) for k in ("step", "episodes", "frames_total", "loss", "accuracy", "gradient_norm", "seconds")} if rows else None
            status = dict(
                status="live", checked_unix=now, pod_id=config["pod_id"], gpu=payload["gpu"],
                remote_total_metric_rows=payload["total_rows"], latest=latest, rolling=rolling,
                latest_diagnostics=payload["diagnostics"][-1] if payload["diagnostics"] else None,
                budget_status=payload["budget"].get("status"), active=payload["budget"].get("active"),
                profile_projection=payload["aggregate"].get("profile_projection"),
                validations=payload["validations"], checkpoints=payload["checkpoints"],
                selected_step=payload["run"].get("selected_step"), gate=payload["run"].get("gate"),
                remote_exited=payload["exited"], deadline_unix=deadline, seconds_to_deadline=deadline - now,
                cadence_seconds=interval, cleanup_owner=config["cleanup_owner"],
            )
            write(root / "live_status.json", status)
            if rows:
                (root / "live_metrics_tail.csv").write_text("\n".join(payload["metrics"]) + "\n")
            aggregate_text = json.dumps(dict(payload["aggregate"], run={arm: payload["run"]}), indent=2)
            digest = hashlib.sha256(aggregate_text.encode()).hexdigest()
            latest_aggregate = root / "aggregate_latest.json"
            if not latest_aggregate.exists() or sha(latest_aggregate) != digest:
                latest_aggregate.write_text(aggregate_text)
            listing_code = f"""
import hashlib,json,pathlib
r=pathlib.Path({remote_results!r})
paths=[]
for p in r.rglob('*'):
 rel=p.relative_to(r).as_posix()
 if p.is_file() and ('validation_' in rel or 'selected_test' in rel or 'terminal_test' in rel) and not rel.endswith('.pt'):
  paths.append(dict(path=rel,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
print(json.dumps(paths))
"""
            listing = subprocess.run([*ssh, "python -c " + shlex.quote(listing_code)], capture_output=True, text=True, timeout=120, check=True)
            for artifact in json.loads(listing.stdout):
                if mirrored.get(artifact["path"]) != artifact["sha256"]:
                    transfer(artifact["path"], artifact["sha256"])
            failures = 0
            write(root / "watcher_status.json", dict(status="terminal_artifacts_mirrored" if payload["exited"] else "monitoring", updated_unix=now, cadence_seconds=interval, remote_checkpoint_count=len(payload["checkpoints"]), immutable_artifact_transfers=len(transfers)))
            if payload["exited"]:
                return
        except Exception as error:
            failures += 1
            write(root / "watcher_status.json", dict(status="degraded", error=repr(error), consecutive_failures=failures, updated_unix=time.time()))
            if failures >= 40:
                raise
        time.sleep(interval)


if __name__ == "__main__":
    main(sys.argv[1])
