"""User-requested pod shutdown: stop remote work, retrieve results, delete the pod.

Usage: python WorkingMemory/cloud_shutdown.py <cloud_provisioning.json>

Sequence: SIGTERM remote supervisor/worker processes; pack the remote results
directory without checkpoints; scp and hash-verify; fetch every checkpoint
listed in any checkpoint_index.jsonl and verify; then RunPod stop, delete and
an absence check. Receipts land in the run's local directory.
"""
import hashlib
import json
import shlex
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
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
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def api_key(path=r"C:\Users\jomor\.runpod\config-palladio.toml"):
    values = {}
    for line in Path(path).read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("\"'")
    return values["apikey"]


def provider_request(method, url, key, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        method=method,
        data=data,
        headers={
            "Authorization": "Bearer " + key,
            "User-Agent": "vawm-research",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read()
            return dict(status=response.status, body=json.loads(body) if body else None)
    except urllib.error.HTTPError as error:
        return dict(status=error.code, body=error.read().decode(errors="replace"))


def ssh_parts(config):
    common = [
        "-i", config["ssh_identity"],
        "-o", "UserKnownHostsFile=" + config["known_hosts"],
        "-o", "StrictHostKeyChecking=yes",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=15",
        "-o", "KexAlgorithms=curve25519-sha256",
    ]
    destination = f'{config["ssh_user"]}@{config["ssh_host"]}'
    ssh = ["ssh", "-n", *common, "-p", str(config["ssh_port"]), destination]
    return common, destination, ssh


def main(config_path, force=False):
    config = read(config_path)
    root = Path(config["local_run"])
    common, destination, ssh = ssh_parts(config)
    remote_results = config["remote_results"]
    remote_parent, remote_name = remote_results.rsplit("/", 1)
    receipt = dict(kind="user_requested_shutdown", pod_id=config["pod_id"], started_unix=time.time(), steps=[])
    retrieved = root / "retrieved_user_stop"
    retrieved.mkdir(exist_ok=True)
    verified = False
    try:
        # 1. Stop remote work so files are stable.
        # Bracketed patterns so pkill never matches this shell's own command line.
        stop_remote = (
            "pkill -TERM -f '[s]weep.py|[s]upervisor_|[w]orker.py' || true; sleep 5; "
            "pkill -KILL -f '[s]weep.py|[s]upervisor_|[w]orker.py' || true; "
            "ps -eo pid,cmd | grep -E '[s]weep|[s]upervisor_|[w]orker' || true"
        )
        stopped = subprocess.run([*ssh, stop_remote], capture_output=True, text=True, timeout=120)
        receipt["steps"].append(dict(step="remote_processes_stopped", remaining=stopped.stdout.strip(), stderr=stopped.stderr[-500:]))
        # 2. Latest metric row and checkpoint inventory.
        probe = f"""
import json,pathlib
r=pathlib.Path({remote_results!r})
out=dict(indexes=[],last_rows={{}})
for idx in r.rglob('checkpoint_index.jsonl'):
  rel=idx.parent.relative_to(r).as_posix()
  rows=[json.loads(x) for x in idx.read_text().splitlines()]
  out['indexes'].append(dict(dir=rel,rows=rows))
  m=idx.parent/'metrics.csv'
  if m.is_file():
    lines=m.read_text().splitlines(); out['last_rows'][rel]=lines[-1][:120] if len(lines)>1 else None
print(json.dumps(out))
"""
        probed = subprocess.run([*ssh, "python -c " + shlex.quote(probe)], capture_output=True, text=True, timeout=120)
        if probed.returncode:
            raise RuntimeError("Inventory probe failed: " + probed.stderr[-2000:])
        inventory = json.loads(probed.stdout)
        receipt["remote_inventory"] = dict(
            checkpoint_dirs={i["dir"]: [r["file"] for r in i["rows"]] for i in inventory["indexes"]},
            last_metric_rows=inventory["last_rows"],
        )
        # 3. Pack and fetch everything except checkpoints.
        remote_archive = f"/workspace/{remote_name}_user_stop_without_checkpoints.tar.gz"
        pack = (
            "cd " + shlex.quote(remote_parent) + " && tar --exclude='*.pt' -czf "
            + shlex.quote(remote_archive) + " " + shlex.quote(remote_name)
            + " && sha256sum " + shlex.quote(remote_archive)
        )
        archive_sha = subprocess.run([*ssh, pack], capture_output=True, text=True, timeout=900, check=True).stdout.split()[0]
        local_archive = root / f"{remote_name}_user_stop_without_checkpoints.tar.gz"
        subprocess.run(["scp", *common, "-P", str(config["ssh_port"]), destination + ":" + remote_archive, str(local_archive)], capture_output=True, text=True, timeout=3600, check=True)
        if sha(local_archive) != archive_sha:
            raise RuntimeError("Archive hash mismatch")
        with tarfile.open(local_archive, "r:gz") as archive:
            for member in archive.getmembers():
                (retrieved / member.name).resolve().relative_to(retrieved.resolve())
                if member.issym() or member.islnk():
                    raise RuntimeError("Refusing archive link")
            archive.extractall(retrieved)
        receipt["steps"].append(dict(step="non_checkpoint_archive_retrieved", sha256=archive_sha, bytes=local_archive.stat().st_size))
        # 4. Fetch every indexed checkpoint and verify.
        fetched = []
        for index in inventory["indexes"]:
            for row in index["rows"]:
                relative = f'{index["dir"]}/{row["file"]}'
                target = retrieved / remote_name / relative
                if target.is_file() and sha(target) == row["sha256"]:
                    fetched.append(relative)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_suffix(".download")
                subprocess.run(["scp", *common, "-P", str(config["ssh_port"]), destination + ":" + remote_results + "/" + relative, str(partial)], capture_output=True, text=True, timeout=1800, check=True)
                if sha(partial) != row["sha256"]:
                    raise RuntimeError("Checkpoint hash mismatch " + relative)
                partial.replace(target)
                fetched.append(relative)
        receipt["steps"].append(dict(step="checkpoints_retrieved", count=len(fetched), files=fetched))
        verified = True
        receipt.update(retrieval_status="verified", retrieved_root=str(retrieved / remote_name))
    except BaseException as error:
        receipt.update(retrieval_status="failed", error=repr(error))
    finally:
        key = api_key()
        base = "https://rest.runpod.io/v1/pods/" + config["pod_id"]
        if not verified and not force:
            # Keep the pod (stopped, so billing for GPU time ends) until retrieval succeeds
            # or the user explicitly forces deletion.
            stopped = provider_request("POST", base + "/stop", key)
            receipt["cleanup"] = dict(stopped=stopped, deleted=None, reason="retrieval failed; pod stopped but NOT deleted (rerun with --force to delete)")
            receipt["completed_unix"] = time.time()
            write(root / "user_stop_receipt.json", receipt)
            print(json.dumps(dict(pod=config["pod_id"], retrieval="failed", error=receipt.get("error"), stop=stopped["status"], deleted=False), indent=1))
            return
        stopped = provider_request("POST", base + "/stop", key)
        deleted = provider_request("DELETE", base, key)
        time.sleep(5)
        absent = provider_request("GET", base, key)
        listing = provider_request("GET", "https://rest.runpod.io/v1/pods", key)
        remaining = [p.get("id") for p in (listing["body"] or [])] if isinstance(listing["body"], list) else listing["body"]
        cleanup = dict(
            stopped=stopped, deleted=deleted, absent_check=absent, remaining_pods=remaining,
            retrieval_verified=verified, cleaned_unix=time.time(),
            reason="user requested shutdown of all pods",
        )
        receipt["cleanup"] = cleanup
        receipt["completed_unix"] = time.time()
        write(root / "user_stop_receipt.json", receipt)
        write(root / "cleanup_receipt.json", cleanup)
        write(root / "lifecycle_status.json", dict(
            status="cleaned" if deleted["status"] in (200, 204) else "cleanup_failed",
            retrieval_verified=verified, updated_unix=time.time(), reason="user requested shutdown",
        ))
        write(root / "live_status.json", dict(status="user_stopped_and_deleted", pod_id=config["pod_id"], updated_unix=time.time(), receipt=str(root / "user_stop_receipt.json")))
        print(json.dumps(dict(pod=config["pod_id"], retrieval=receipt.get("retrieval_status"), error=receipt.get("error"), stop=stopped["status"], delete=deleted["status"], absent=absent["status"], remaining=remaining, checkpoints=len(fetched) if verified else None), indent=1))


if __name__ == "__main__":
    main(sys.argv[1], force="--force" in sys.argv[2:])
