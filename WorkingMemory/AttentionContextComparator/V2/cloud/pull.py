"""Pull the v2 pod's results and every indexed checkpoint locally without stopping training.

Usage: python WorkingMemory/AttentionContextComparator/V2/cloud/pull.py <cloud_provisioning.json>
Files land in <local_run>/pulled/<results dir>/ and are hash-verified against the
remote checkpoint index; a receipt is written to <local_run>/pull_receipt_<unix>.json.
"""
import json
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))

from WorkingMemory.cloud_shutdown import read, sha, ssh_parts, write


def main(config_path):
    config = read(config_path)
    root = Path(config["local_run"])
    common, destination, ssh = ssh_parts(config)
    remote_results = config["remote_results"]
    remote_parent, remote_name = remote_results.rsplit("/", 1)
    started = time.time()
    pulled = root / "pulled"
    pulled.mkdir(exist_ok=True)
    remote_archive = f"/workspace/{remote_name}_pull_without_checkpoints.tar.gz"
    pack = "cd " + shlex.quote(remote_parent) + " && tar --exclude='*.pt' -czf " + shlex.quote(remote_archive) + " " + shlex.quote(remote_name) + " && sha256sum " + shlex.quote(remote_archive)
    archive_sha = subprocess.run([*ssh, pack], capture_output=True, text=True, timeout=900, check=True).stdout.split()[0]
    local_archive = root / f"{remote_name}_pull_{int(started)}_without_checkpoints.tar.gz"
    subprocess.run(["scp", *common, "-P", str(config["ssh_port"]), destination + ":" + remote_archive, str(local_archive)], capture_output=True, text=True, timeout=3600, check=True)
    if sha(local_archive) != archive_sha:
        raise RuntimeError("Archive hash mismatch")
    with tarfile.open(local_archive, "r:gz") as archive:
        for member in archive.getmembers():
            (pulled / member.name).resolve().relative_to(pulled.resolve())
            if member.issym() or member.islnk():
                raise RuntimeError("Refusing archive link")
        archive.extractall(pulled)
    result_root = pulled / remote_name
    index_path = result_root / config["arm"] / "training" / "checkpoint_index.jsonl"
    rows = [json.loads(line) for line in index_path.read_text().splitlines()]
    fetched, skipped = [], []
    for row in rows:
        relative = f'{config["arm"]}/training/{row["file"]}'
        target = result_root / relative
        if target.is_file() and sha(target) == row["sha256"]:
            skipped.append(relative)
            continue
        partial = target.with_suffix(".download")
        subprocess.run(["scp", *common, "-P", str(config["ssh_port"]), destination + ":" + remote_results + "/" + relative, str(partial)], capture_output=True, text=True, timeout=1800, check=True)
        if sha(partial) != row["sha256"]:
            raise RuntimeError("Checkpoint hash mismatch " + relative)
        partial.replace(target)
        fetched.append(relative)
        print("fetched", relative, flush=True)
    metrics = result_root / config["arm"] / "training" / "metrics.csv"
    receipt = dict(
        kind="pull_while_training", pod_id=config["pod_id"], started_unix=started, completed_unix=time.time(),
        archive=str(local_archive), archive_sha256=archive_sha, results_root=str(result_root),
        checkpoints_fetched=fetched, checkpoints_already_present=skipped, checkpoint_steps=[r["step"] for r in rows],
        metric_rows=(sum(1 for _ in metrics.open()) - 1) if metrics.is_file() else None,
        validations=sorted(p.name for p in result_root.glob("validation_*_result.json")),
    )
    write(root / f"pull_receipt_{int(started)}.json", receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k not in ("checkpoints_fetched", "checkpoints_already_present")}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
