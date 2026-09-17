"""Detached owner of the v2 cloud pod: watch, verified retrieval, stop and delete."""
import json
import os
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))

from WorkingMemory.cloud_shutdown import api_key, provider_request, read, sha, write


def main(config_path):
    config = read(config_path)
    root = Path(config["local_run"])
    deadline = float(config["deadline_unix"])
    common = ["-i", config["ssh_identity"], "-o", "UserKnownHostsFile=" + config["known_hosts"], "-o", "StrictHostKeyChecking=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "KexAlgorithms=curve25519-sha256"]
    destination = f'{config["ssh_user"]}@{config["ssh_host"]}'
    ssh = ["ssh", "-n", *common, "-p", str(config["ssh_port"]), destination]
    watcher_log = (root / "watcher_detached.log").open("a")

    def start_watcher():
        return subprocess.Popen([sys.executable, "-B", str(HERE / "watch_remote.py"), str(config_path)], cwd=HERE.parents[3], stdout=watcher_log, stderr=subprocess.STDOUT)

    watcher = start_watcher()
    restarts = 0
    write(root / "lifecycle_status.json", dict(status="monitoring", lifecycle_pid=os.getpid(), watcher_pid=watcher.pid, deadline_unix=deadline, updated_unix=time.time()))
    terminal = False
    while time.time() < deadline + float(config.get("retrieval_grace_seconds", 900)):
        status = read(root / "watcher_status.json") if (root / "watcher_status.json").exists() else {}
        if status.get("status") == "terminal_artifacts_mirrored":
            terminal = True
            break
        if watcher.poll() is not None:
            # A watcher crash is a monitoring problem, never a reason to delete the pod.
            watcher_log.flush()
            restarts += 1
            write(root / "lifecycle_status.json", dict(status="watcher_restarted", restarts=restarts, lifecycle_pid=os.getpid(), deadline_unix=deadline, updated_unix=time.time()))
            time.sleep(60)
            watcher = start_watcher()
            continue
        time.sleep(45)
    if watcher.poll() is None:
        watcher.terminate()
        try:
            watcher.wait(timeout=30)
        except subprocess.TimeoutExpired:
            watcher.kill()
    watcher_log.close()
    retrieval = dict(started_unix=time.time(), terminal_seen=terminal)
    remote_results = config["remote_results"]
    remote_parent, remote_name = remote_results.rsplit("/", 1)
    retrieved = root / "retrieved"
    retrieved.mkdir(exist_ok=True)
    verified = False
    try:
        subprocess.run([*ssh, "pkill -TERM -f 'cloud_sweep.py|worker.py' || true"], capture_output=True, text=True, timeout=60)
        remote_archive = f"/workspace/{remote_name}_without_checkpoints.tar.gz"
        pack = "cd " + shlex.quote(remote_parent) + " && tar --exclude='*.pt' -czf " + shlex.quote(remote_archive) + " " + shlex.quote(remote_name) + " && sha256sum " + shlex.quote(remote_archive)
        archive_sha = subprocess.run([*ssh, pack], capture_output=True, text=True, timeout=900, check=True).stdout.split()[0]
        local_archive = root / f"{remote_name}_without_checkpoints.tar.gz"
        subprocess.run(["scp", *common, "-P", str(config["ssh_port"]), destination + ":" + remote_archive, str(local_archive)], timeout=3600, check=True)
        if sha(local_archive) != archive_sha:
            raise RuntimeError("Archive hash mismatch")
        with tarfile.open(local_archive, "r:gz") as archive:
            for member in archive.getmembers():
                (retrieved / member.name).resolve().relative_to(retrieved.resolve())
                if member.issym() or member.islnk():
                    raise RuntimeError("Refusing archive link")
            archive.extractall(retrieved)
        result_root = retrieved / remote_name
        index_path = result_root / config["arm"] / "training" / "checkpoint_index.jsonl"
        rows = [json.loads(line) for line in index_path.read_text().splitlines()] if index_path.is_file() else []
        for row in rows:
            relative = f'{config["arm"]}/training/{row["file"]}'
            target = result_root / relative
            if target.is_file() and sha(target) == row["sha256"]:
                continue
            partial = target.with_suffix(".download")
            subprocess.run(["scp", *common, "-P", str(config["ssh_port"]), destination + ":" + remote_results + "/" + relative, str(partial)], timeout=1800, check=True)
            if sha(partial) != row["sha256"]:
                raise RuntimeError("Checkpoint hash mismatch " + relative)
            partial.replace(target)
        artifact_index = read(result_root / "artifact_index.json") if (result_root / "artifact_index.json").is_file() else {}
        failures = [name for name, digest in artifact_index.items() if not (result_root / name).is_file() or sha(result_root / name) != digest]
        if failures:
            raise RuntimeError("Artifact verification failed: " + repr(failures[:10]))
        verified = True
        retrieval.update(status="verified", indexed_files=len(artifact_index), checkpoint_files=len(rows), archive_sha256=archive_sha, completed_unix=time.time())
    except BaseException as error:
        retrieval.update(status="failed", error=repr(error), failed_unix=time.time())
    finally:
        write(root / "retrieval_receipt.json", retrieval)
        key = api_key()
        base = "https://rest.runpod.io/v1/pods/" + config["pod_id"]
        stopped = provider_request("POST", base + "/stop", key)
        deleted = provider_request("DELETE", base, key)
        absent = provider_request("GET", base, key)
        cleanup = dict(stopped=stopped, deleted=deleted, absent_check=absent, retrieval_verified=verified, cleaned_unix=time.time(), reason="terminal result retrieved" if terminal and verified else "deadline, watcher exit or retrieval failure; billing stopped")
        write(root / "cleanup_receipt.json", cleanup)
        write(root / "lifecycle_status.json", dict(status="cleaned" if deleted["status"] in (200, 204) else "cleanup_failed", retrieval_verified=verified, updated_unix=time.time(), cleanup_receipt=str(root / "cleanup_receipt.json")))


if __name__ == "__main__":
    main(sys.argv[1])
