"""Detached monitor, verified retrieval, and RunPod stop/delete owner."""
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
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
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def api_key(path):
    values = {
        key.strip(): value.strip().strip("\"'")
        for key, value in (
            line.split("=", 1)
            for line in Path(path).read_text().splitlines()
            if "=" in line
        )
    }
    return values["apikey"]


def provider_request(method, url, key):
    request = urllib.request.Request(
        url,
        method=method,
        headers={"Authorization": "Bearer " + key, "User-Agent": "vawm-research"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read()
            return dict(
                status=response.status,
                body=json.loads(body) if body else None,
            )
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")
        return dict(status=error.code, body=body)


def main():
    config = read(HERE / "cloud_provisioning.json")
    root = Path(config["local_run"])
    deadline = float(config["deadline_unix"])
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
    watcher_log = (root / "watcher_detached.log").open("a")
    watcher = subprocess.Popen(
        [sys.executable, "-B", str(HERE / "watch_remote.py")],
        cwd=HERE.parents[1],
        stdout=watcher_log,
        stderr=subprocess.STDOUT,
    )
    write(
        root / "lifecycle_status.json",
        dict(
            status="monitoring",
            lifecycle_pid=os.getpid(),
            watcher_pid=watcher.pid,
            deadline_unix=deadline,
            updated_unix=time.time(),
        ),
    )
    terminal = False
    while time.time() < deadline:
        watcher_status = (
            read(root / "watcher_status.json")
            if (root / "watcher_status.json").exists()
            else {}
        )
        if watcher_status.get("status") == "terminal_artifacts_mirrored":
            terminal = True
            break
        if watcher.poll() is not None:
            watcher_log.flush()
            raise RuntimeError("45-second watcher exited before terminal state")
        time.sleep(45)

    if watcher.poll() is None:
        watcher.terminate()
        watcher.wait(timeout=30)
    watcher_log.close()
    retrieval = dict(started_unix=time.time(), terminal_seen=terminal)
    remote_results = config["remote_results"]
    retrieved = root / "retrieved"
    retrieved.mkdir(exist_ok=True)
    verified = False
    try:
        remote_archive = "/workspace/attention_context_results_without_checkpoints.tar.gz"
        remote_parent, remote_name = remote_results.rsplit("/", 1)
        pack = (
            "cd "
            + shlex.quote(remote_parent)
            + " && tar --exclude='*.pt' -czf "
            + shlex.quote(remote_archive)
            + " "
            + shlex.quote(remote_name)
            + " && sha256sum "
            + shlex.quote(remote_archive)
        )
        packed = subprocess.run(
            [*ssh, pack], capture_output=True, text=True, timeout=600, check=True
        )
        archive_sha = packed.stdout.split()[0]
        local_archive = root / "attention_context_results_without_checkpoints.tar.gz"
        subprocess.run(
            [
                "scp",
                *common,
                "-P",
                str(config["ssh_port"]),
                destination + ":" + remote_archive,
                str(local_archive),
            ],
            timeout=3600,
            check=True,
        )
        if sha(local_archive) != archive_sha:
            raise RuntimeError("Non-checkpoint archive hash mismatch")
        with tarfile.open(local_archive, "r:gz") as archive:
            for member in archive.getmembers():
                target = (retrieved / member.name).resolve()
                target.relative_to(retrieved.resolve())
                if member.issym() or member.islnk():
                    raise RuntimeError("Refusing archive link")
            archive.extractall(retrieved)
        result_root = retrieved / remote_name
        artifact_index = read(result_root / "artifact_index.json")
        checkpoint_rows = [
            (name, digest)
            for name, digest in artifact_index.items()
            if name.endswith(".pt")
        ]
        for name, digest in checkpoint_rows:
            target = result_root / Path(name)
            if target.exists() and sha(target) == digest:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(".download")
            subprocess.run(
                [
                    "scp",
                    *common,
                    "-P",
                    str(config["ssh_port"]),
                    destination + ":" + remote_results + "/" + name,
                    str(partial),
                ],
                timeout=1800,
                check=True,
            )
            if sha(partial) != digest:
                raise RuntimeError("Checkpoint hash mismatch " + name)
            partial.replace(target)
        failures = []
        for name, digest in artifact_index.items():
            path = result_root / Path(name)
            if not path.is_file() or sha(path) != digest:
                failures.append(name)
        if failures:
            raise RuntimeError("Final artifact verification failed: " + repr(failures))
        verified = True
        retrieval.update(
            status="verified",
            files=len(artifact_index),
            checkpoint_files=len(checkpoint_rows),
            archive_sha256=archive_sha,
            completed_unix=time.time(),
        )
        write(root / "retrieval_receipt.json", retrieval)
    except BaseException as error:
        retrieval.update(status="failed", error=repr(error), failed_unix=time.time())
        write(root / "retrieval_receipt.json", retrieval)
    finally:
        key = api_key(r"C:\Users\jomor\.runpod\config-palladio.toml")
        base = "https://rest.runpod.io/v1/pods/" + config["pod_id"]
        stopped = provider_request("POST", base + "/stop", key)
        deleted = provider_request("DELETE", base, key)
        absent = provider_request("GET", base, key)
        cleanup = dict(
            stopped=stopped,
            deleted=deleted,
            absent_check=absent,
            retrieval_verified=verified,
            cleaned_unix=time.time(),
            reason=(
                "terminal result retrieved"
                if terminal and verified
                else "hard deadline or retrieval failure; billing stopped"
            ),
        )
        write(root / "cleanup_receipt.json", cleanup)
        write(
            root / "lifecycle_status.json",
            dict(
                status="cleaned" if deleted["status"] in (200, 204) else "cleanup_failed",
                retrieval_verified=verified,
                updated_unix=time.time(),
                cleanup_receipt=str(root / "cleanup_receipt.json"),
            ),
        )
        if not verified:
            raise RuntimeError("Pod cleaned without complete verified retrieval")


if __name__ == "__main__":
    main()
