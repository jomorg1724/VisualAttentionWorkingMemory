"""Bounded hash-index checkpoint sync and verified result retrieval."""
import csv
import hashlib
import io
import json
import re
import shlex
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def write_text(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def required(config, key):
    value = config.get(key)
    if value is None or value == "":
        raise ValueError("cloud_provisioning.json requires runtime value " + key)
    return value


def main():
    config = read(HERE / "cloud_provisioning.json")
    for key in (
        "pod_id",
        "deadline_unix",
        "remote_root",
        "remote_results",
        "ssh_host",
        "ssh_port",
        "ssh_user",
        "ssh_identity",
        "known_hosts",
        "local_run",
    ):
        required(config, key)
    if config.get("authorized_arms") != ["prospective_query"]:
        raise RuntimeError("Watcher is restricted to the one authorized arm")

    root = Path(config["local_run"])
    root.mkdir(parents=True, exist_ok=True)
    deadline = float(config["deadline_unix"])
    grace = min(int(config.get("retrieval_grace_seconds", 600)), 900)
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
    ]
    destination = f'{config["ssh_user"]}@{config["ssh_host"]}'
    ssh = ["ssh", "-n", *common, "-p", str(config["ssh_port"]), destination]
    remote_root = config["remote_root"]
    remote_results = config["remote_results"]
    status = root / "watcher_status.json"
    incremental = root / "incremental" / "prospective_query_results"
    incremental.mkdir(parents=True, exist_ok=True)
    transfers_path = root / "incremental_transfers.json"
    transfers = read(transfers_path) if transfers_path.exists() else []
    metrics_status = root / "live_metrics_status.json"
    metrics_tail = root / "live_metrics_tail.csv"

    def sync_live_metrics():
        code = (
            "import json,pathlib; "
            f"r=pathlib.Path({remote_results!r}); "
            "m=(r/'prospective_query/training/metrics.csv').read_text().splitlines(); "
            "i=r/'prospective_query/training/checkpoint_index.jsonl'; "
            "print(json.dumps(dict("
            "budget=json.loads((r/'budget.json').read_text()),"
            "metrics=([m[0]]+m[-256:]),"
            "total_rows=max(0,len(m)-1),"
            "validations=[{k:d.get(k) for k in "
            "('status','step','gamma','rank','task_scores','comparison_status')} "
            "for p in sorted(r.glob('validation_*_result.json')) "
            "for d in [json.loads(p.read_text())]],"
            "checkpoints=([json.loads(x) for x in i.read_text().splitlines()] "
            "if i.is_file() else []))))"
        )
        response = subprocess.run(
            [*ssh, "python -c " + shlex.quote(code)],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        payload = json.loads(response.stdout)
        text = "\n".join(payload["metrics"]) + "\n"
        rows = list(csv.DictReader(io.StringIO(text)))
        if not rows:
            raise RuntimeError("Remote metrics CSV has no data rows")
        numeric = (
            "loss",
            "accuracy",
            "gradient_norm",
            "seconds",
            "gamma",
            "gamma_gradient_preclip",
        )
        integer = ("step", "episodes", "frames_total", "clipped", "frames")
        for row in rows:
            for key in numeric:
                row[key] = float(row[key])
            for key in integer:
                row[key] = int(row[key])

        def rolling(size):
            selected = rows[-min(size, len(rows)) :]
            return dict(
                updates=len(selected),
                step_start=selected[0]["step"],
                step_end=selected[-1]["step"],
                mean_loss=sum(row["loss"] for row in selected) / len(selected),
                mean_accuracy=sum(row["accuracy"] for row in selected) / len(selected),
                mean_gradient_norm_preclip=(
                    sum(row["gradient_norm"] for row in selected) / len(selected)
                ),
                clip_fraction=(
                    sum(row["clipped"] for row in selected) / len(selected)
                ),
                mean_update_seconds=(
                    sum(row["seconds"] for row in selected) / len(selected)
                ),
                mean_gamma_gradient_preclip=(
                    sum(row["gamma_gradient_preclip"] for row in selected)
                    / len(selected)
                ),
            )

        diagnostic = None
        for row in reversed(rows):
            detail = json.loads(row["diagnostics"])
            if any(detail.values()):
                diagnostic = dict(
                    step=row["step"],
                    task_gradients={
                        task: {
                            key: value
                            for key, value in values.items()
                            if key
                            not in ("per_frame_attention", "gradient_semantics")
                        }
                        for task, values in detail.items()
                    },
                )
                break
        latest = rows[-1]
        target = int(config["target_updates"]) + 8400
        rolling_values = {str(size): rolling(size) for size in (20, 100, 256)}
        remaining_updates = max(0, target - latest["step"])
        estimated_remaining_seconds = (
            remaining_updates * rolling_values["100"]["mean_update_seconds"] + 1800
        )
        checked = time.time()
        validations = payload["validations"]
        completed_validation_steps = [
            row["step"] for row in validations if row.get("status") == "completed"
        ]
        next_validation = next(
            (
                step
                for step in (9200, 10000, 10800, 11600, 12400)
                if step not in completed_validation_steps
            ),
            None,
        )
        snapshot = dict(
            status="live",
            checked_unix=checked,
            remote_metrics=(
                remote_results
                + "/prospective_query/training/metrics.csv"
            ),
            locally_mirrored_tail=str(metrics_tail),
            remote_total_metric_rows=payload["total_rows"],
            remote_budget_status=payload["budget"].get("status"),
            remote_active=payload["budget"].get("active"),
            latest={
                key: latest[key]
                for key in (
                    "step",
                    "episodes",
                    "frames_total",
                    "loss",
                    "accuracy",
                    "gradient_norm",
                    "clipped",
                    "frames",
                    "seconds",
                    "gamma",
                    "gamma_gradient_preclip",
                )
            },
            progress=dict(
                parent_step=8400,
                target_step=target,
                completed_updates=latest["step"] - 8400,
                remaining_updates=remaining_updates,
                fraction=(latest["step"] - 8400) / int(config["target_updates"]),
            ),
            rolling=rolling_values,
            gamma=dict(
                mirrored_tail_updates=len(rows),
                mirrored_tail_minimum=min(row["gamma"] for row in rows),
                mirrored_tail_maximum=max(row["gamma"] for row in rows),
                current=latest["gamma"],
            ),
            eta=dict(
                basis=(
                    "Remaining updates times the latest 100-update mean recorded "
                    "compute time, plus the protocol's 1800-second evaluation and "
                    "retrieval reserve; excludes unmeasured orchestration overhead."
                ),
                estimated_remaining_seconds=estimated_remaining_seconds,
                estimated_completion_unix=checked + estimated_remaining_seconds,
                hard_stop_unix=deadline,
                seconds_to_hard_stop=deadline - checked,
            ),
            latest_checkpoint=(
                payload["checkpoints"][-1] if payload["checkpoints"] else None
            ),
            latest_task_gradient_diagnostic=diagnostic,
            metric_scope=(
                "Each CSV row stores the mean loss and accuracy over the five "
                "task microbatches in one optimizer update. Individual per-task "
                "losses are computed during training but are not persisted."
            ),
            validation_status=dict(
                completed_steps=completed_validation_steps,
                latest=(validations[-1] if validations else None),
                next_scheduled_step=next_validation,
                note=(
                    "Validation summaries are read from immutable completed result "
                    "files; final held-out evaluation is separate."
                ),
            ),
        )
        write_text(metrics_tail, text)
        write(metrics_status, snapshot)

    def sync_checkpoints():
        code = (
            "import pathlib,json; "
            f"r=pathlib.Path({remote_results!r}); "
            "print(json.dumps([dict(path=str(p.parent.relative_to(r)/v['file']),"
            "sha256=v['sha256']) for p in r.rglob('checkpoint_index.jsonl') "
            "for v in [json.loads(line) for line in p.read_text().splitlines()]]))"
        )
        listing = subprocess.run(
            [*ssh, "python -c " + shlex.quote(code)],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        for row in json.loads(listing.stdout):
            name = row["path"].replace("\\", "/")
            if not re.fullmatch(
                r"(profile|prospective_query/training)/checkpoint_[0-9]{6}\.pt",
                name,
            ):
                raise RuntimeError("Unexpected checkpoint path " + name)
            local = incremental / Path(name)
            local.parent.mkdir(parents=True, exist_ok=True)
            if local.exists() and sha(local) == row["sha256"]:
                continue
            tick = time.time()
            temporary = local.with_suffix(".download")
            subprocess.run(
                [
                    "scp",
                    *common,
                    "-P",
                    str(config["ssh_port"]),
                    destination + ":" + remote_results + "/" + name,
                    str(temporary),
                ],
                capture_output=True,
                text=True,
                timeout=600,
                check=True,
            )
            if sha(temporary) != row["sha256"]:
                raise RuntimeError("Downloaded checkpoint hash mismatch " + name)
            temporary.replace(local)
            transfers.append(
                dict(
                    file=name,
                    bytes=local.stat().st_size,
                    seconds=time.time() - tick,
                    sha256=row["sha256"],
                )
            )
            write(transfers_path, transfers)

    while time.time() < deadline + grace:
        try:
            sync_live_metrics()
            sync_checkpoints()
            terminal_code = (
                "import json,pathlib\n"
                f"r=pathlib.Path({remote_results!r})\n"
                "required=[r/'budget.json',r/'exit.json',r/'artifact_index.json']\n"
                "if not all(path.is_file() for path in required): raise SystemExit(1)\n"
                "b=json.loads(required[0].read_text())\n"
                "if b.get('active') is not None: raise SystemExit(1)\n"
                "if b.get('status') not in ('completed','failed'): raise SystemExit(1)\n"
                "print(required[1].read_text())"
            )
            check = subprocess.run(
                [*ssh, "python -c " + shlex.quote(terminal_code)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if check.returncode == 0:
                exit_info = json.loads(check.stdout)
                write(
                    status,
                    dict(status="retrieving", exit=exit_info, updated_unix=time.time()),
                )
                remote_archive = remote_root + "/prospective_query_results.tar.gz"
                pack = subprocess.run(
                    [
                        *ssh,
                        "cd "
                        + shlex.quote(remote_root)
                        + " && tar --exclude='*.pt' -czf "
                        + shlex.quote(remote_archive)
                        + " prospective_query_results && sha256sum "
                        + shlex.quote(remote_archive),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=180,
                    check=True,
                )
                digest = pack.stdout.split()[0]
                archive = root / "prospective_query_results.tar.gz"
                subprocess.run(
                    [
                        "scp",
                        *common,
                        "-P",
                        str(config["ssh_port"]),
                        destination + ":" + remote_archive,
                        str(archive),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=600,
                    check=True,
                )
                if sha(archive) != digest:
                    raise RuntimeError("Result archive hash mismatch")
                retrieved = root / "retrieved"
                retrieved.mkdir(exist_ok=True)
                with tarfile.open(archive, "r:gz") as handle:
                    for member in handle.getmembers():
                        target = (retrieved / member.name).resolve()
                        if (
                            not target.is_relative_to(retrieved.resolve())
                            or member.issym()
                            or member.islnk()
                        ):
                            raise RuntimeError("Unsafe archive member")
                    handle.extractall(retrieved)
                fetched = retrieved / "prospective_query_results"
                shutil.copytree(incremental, fetched, dirs_exist_ok=True)
                artifact_index = read(fetched / "artifact_index.json")
                for name, expected in artifact_index.items():
                    target = (fetched / name).resolve()
                    if (
                        not target.is_relative_to(fetched.resolve())
                        or not target.is_file()
                        or sha(target) != expected
                    ):
                        raise RuntimeError("Artifact verification failed " + name)
                actual = {
                    str(path.relative_to(fetched)).replace("\\", "/")
                    for path in fetched.rglob("*")
                    if path.is_file() and path.name != "artifact_index.json"
                }
                if actual != set(artifact_index):
                    raise RuntimeError("Retrieved artifact set differs from index")
                receipt = dict(
                    status="verified",
                    pod_id=config["pod_id"],
                    archive=str(archive),
                    archive_sha256=digest,
                    archive_excludes_checkpoints=True,
                    checkpoints_restored_from=str(incremental),
                    verified_files=len(artifact_index),
                    results=str(fetched),
                    exit=exit_info,
                    verified_unix=time.time(),
                    cleanup=(
                        "Ready for external coordinator stop/delete; retrieval does "
                        "not itself alter pod lifecycle or billing"
                    ),
                )
                write(root / "retrieval_receipt.json", receipt)
                write(status, receipt)
                print(json.dumps(receipt), flush=True)
                return
            write(
                status,
                dict(
                    status="waiting",
                    checked_unix=time.time(),
                    returncode=check.returncode,
                    stderr=check.stderr[-300:],
                    metrics_status=str(metrics_status),
                ),
            )
        except Exception as error:
            write(
                status,
                dict(status="retrying", error=repr(error), updated_unix=time.time()),
            )
            print(repr(error), flush=True)
        time.sleep(45)
    write(
        status,
        dict(
            status="deadline_retrieval_pending",
            updated_unix=time.time(),
            action=(
                "External coordinator must stop the pod; preserve its disk until "
                "hash-verified retrieval completes"
            ),
        ),
    )


if __name__ == "__main__":
    main()
