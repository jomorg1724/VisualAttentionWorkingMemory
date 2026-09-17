"""Mirror live metrics and immutable artifacts at a 45-second cadence."""
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

HERE = Path(__file__).resolve().parent


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


def main():
    config = read(HERE / "cloud_provisioning.json")
    if config.get("authorized_arms") != ["spatial_priority_readout_scratch"]:
        raise RuntimeError("Watcher restricted to spatial-priority-readout arm")
    root = Path(config["local_run"])
    remote_root = config["remote_root"]
    remote_results = config["remote_results"]
    deadline = float(config["deadline_unix"])
    interval = 45
    common = [
        "-i",
        config["ssh_identity"],
        "-o",
        "UserKnownHostsFile=" + config["known_hosts"],
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "KexAlgorithms=curve25519-sha256",
    ]
    destination = f'{config["ssh_user"]}@{config["ssh_host"]}'
    ssh = [
        "ssh",
        "-n",
        *common,
        "-p",
        str(config["ssh_port"]),
        destination,
    ]
    incremental = root / "incremental" / "spatial_priority_results"
    incremental.mkdir(parents=True, exist_ok=True)
    transfers_path = root / "incremental_transfers.json"
    transfers = read(transfers_path) if transfers_path.exists() else []
    mirrored = {row["remote"]: row["sha256"] for row in transfers}

    def remote_json():
        code = f"""
import json,pathlib,subprocess
r=pathlib.Path({remote_results!r})
b=json.loads((r/'budget.json').read_text()) if (r/'budget.json').is_file() else {{}}
a=json.loads((r/'aggregate.json').read_text()) if (r/'aggregate.json').is_file() else {{}}
m=r/'spatial_priority_readout_scratch/training/metrics.csv'
lines=m.read_text().splitlines() if m.is_file() else []
idx=r/'spatial_priority_readout_scratch/training/checkpoint_index.jsonl'
checkpoints=[json.loads(x) for x in idx.read_text().splitlines()] if idx.is_file() else []
validations=[]
for p in sorted(r.glob('validation_*_result.json')):
 d=json.loads(p.read_text()); validations.append(dict(file=p.name,step=d.get('step'),status=d.get('status'),rank=d.get('rank'),task_scores=d.get('task_scores')))
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True).strip()
print(json.dumps(dict(budget=b,aggregate=a,metrics=([lines[0]]+lines[1:][-256:] if lines else []),total_rows=max(0,len(lines)-1),checkpoints=checkpoints,validations=validations,gpu=gpu)))
"""
        result = subprocess.run(
            [*ssh, "python -c " + shlex.quote(code)],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        return json.loads(result.stdout)

    def transfer(relative, expected_sha):
        relative = relative.replace("\\", "/")
        if not re.fullmatch(
            r"((profile|spatial_priority_readout_scratch/(training|validation_[0-9]+|selected_test|terminal_test))/[A-Za-z0-9_.-]+|validation_[0-9]+_(job|result)\.json|validation_[0-9]+\.log)",
            relative,
        ):
            raise RuntimeError("Unexpected remote artifact " + relative)
        local = incremental / Path(relative)
        if local.is_file() and sha(local) == expected_sha:
            return
        local.parent.mkdir(parents=True, exist_ok=True)
        temporary = local.with_suffix(local.suffix + ".download")
        subprocess.run(
            [
                "scp",
                *common,
                "-P",
                str(config["ssh_port"]),
                destination + ":" + remote_results + "/" + relative,
                str(temporary),
            ],
            capture_output=True,
            text=True,
            timeout=900,
            check=True,
        )
        if sha(temporary) != expected_sha:
            raise RuntimeError("Transfer hash mismatch " + relative)
        temporary.replace(local)
        transfers.append(
            dict(
                remote=relative,
                local=str(local),
                bytes=local.stat().st_size,
                sha256=expected_sha,
                mirrored_unix=time.time(),
            )
        )
        mirrored[relative] = expected_sha
        write(transfers_path, transfers)

    failures = 0
    while time.time() < deadline + 900:
        try:
            payload = remote_json()
            now = time.time()
            rows = []
            if payload["metrics"]:
                rows = list(
                    csv.DictReader(io.StringIO("\n".join(payload["metrics"]) + "\n"))
                )
            rolling = {}
            for size in (20, 100, 256):
                selected = rows[-min(size, len(rows)) :] if rows else []
                if selected:
                    rolling[str(size)] = dict(
                        updates=len(selected),
                        step_start=int(selected[0]["step"]),
                        step_end=int(selected[-1]["step"]),
                        mean_loss=sum(float(row["loss"]) for row in selected)
                        / len(selected),
                        mean_accuracy=sum(
                            float(row["accuracy"]) for row in selected
                        )
                        / len(selected),
                        mean_gradient_norm=sum(
                            float(row["gradient_norm"]) for row in selected
                        )
                        / len(selected),
                        mean_update_seconds=sum(
                            float(row["seconds"]) for row in selected
                        )
                        / len(selected),
                    )
            latest = (
                {
                    key: (
                        int(rows[-1][key])
                        if key in ("step", "episodes", "frames_total")
                        else float(rows[-1][key])
                    )
                    for key in (
                        "step",
                        "episodes",
                        "frames_total",
                        "loss",
                        "accuracy",
                        "gradient_norm",
                        "seconds",
                    )
                }
                if rows
                else None
            )
            status = dict(
                status="live",
                checked_unix=now,
                pod_id=config["pod_id"],
                gpu=payload["gpu"],
                remote_total_metric_rows=payload["total_rows"],
                latest=latest,
                rolling=rolling,
                budget_status=payload["budget"].get("status"),
                active=payload["budget"].get("active"),
                validations=payload["validations"],
                checkpoints=payload["checkpoints"],
                deadline_unix=deadline,
                seconds_to_deadline=deadline - now,
                cadence_seconds=interval,
                cleanup_owner=config["cleanup_owner"],
            )
            write(root / "live_status.json", status)
            if rows:
                (root / "live_metrics_tail.csv").write_text(
                    "\n".join(payload["metrics"]) + "\n"
                )
            aggregate_text = json.dumps(payload["aggregate"], indent=2)
            aggregate_digest = hashlib.sha256(aggregate_text.encode()).hexdigest()
            latest_aggregate = root / "aggregate_latest.json"
            if (
                not latest_aggregate.exists()
                or sha(latest_aggregate) != aggregate_digest
            ):
                latest_aggregate.write_text(aggregate_text)
                snapshot = (
                    root
                    / "aggregate_snapshots"
                    / f"{int(now)}_{aggregate_digest[:12]}.json"
                )
                snapshot.parent.mkdir(exist_ok=True)
                snapshot.write_text(aggregate_text)
            listing_code = f"""
import hashlib,json,pathlib
r=pathlib.Path({remote_results!r})
paths=[]
for p in r.rglob('*'):
 rel=str(p.relative_to(r)).replace('\\\\','/')
 if p.is_file() and ('/validation_' in '/'+rel or '/selected_test/' in '/'+rel or '/terminal_test/' in '/'+rel):
  paths.append(dict(path=rel,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
print(json.dumps(paths))
"""
            listing = subprocess.run(
                [*ssh, "python -c " + shlex.quote(listing_code)],
                capture_output=True,
                text=True,
                timeout=120,
                check=True,
            )
            for artifact in json.loads(listing.stdout):
                if mirrored.get(artifact["path"]) != artifact["sha256"]:
                    transfer(artifact["path"], artifact["sha256"])
            failures = 0
            write(
                root / "watcher_status.json",
                dict(
                    status="monitoring",
                    updated_unix=now,
                    cadence_seconds=interval,
                    remote_checkpoint_count=len(payload["checkpoints"]),
                    immutable_artifact_transfers=len(transfers),
                    cleanup_owner=config["cleanup_owner"],
                ),
            )
            terminal = (
                payload["budget"].get("active") is None
                and payload["budget"].get("status") in ("completed", "failed")
            )
            if terminal:
                write(
                    root / "watcher_status.json",
                    dict(
                        status="terminal_artifacts_mirrored",
                        terminal_status=payload["budget"].get("status"),
                        updated_unix=now,
                        cleanup_owner=config["cleanup_owner"],
                    ),
                )
                return
        except Exception as error:
            failures += 1
            write(
                root / "watcher_status.json",
                dict(
                    status="retrying",
                    failures=failures,
                    error=repr(error),
                    updated_unix=time.time(),
                    cleanup_owner=config["cleanup_owner"],
                ),
            )
            if failures >= 20:
                raise
        time.sleep(interval)
    raise TimeoutError("Watcher exceeded deadline plus retrieval grace")


if __name__ == "__main__":
    main()
