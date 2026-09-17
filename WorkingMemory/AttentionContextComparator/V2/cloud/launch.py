"""Provision one RunPod pod and start the AV-context v2 overnight arm.

Usage: python WorkingMemory/AttentionContextComparator/V2/cloud/launch.py --hours 20 --updates 30000
"""
import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from WorkingMemory.AttentionContextComparator.V2.cloud.prepare_bundle import prepare
from WorkingMemory.cloud_shutdown import api_key, provider_request, read, sha, write

DATASET = ROOT / "WorkingMemory/ProspectiveQuery/runs/prospective_20260914_175602/bsds500_runtime.tar.gz"
DATASET_SHA = "990e2bd405f6bcc4c940299f437685fc08765dcb56923ab0ab639f5f37c1883e"
SSH_KEY = Path.home() / ".runpod/ssh/runpodctl-ssh-key"
GPU_PREFERENCE = ("NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 4090")


def validation_targets(updates):
    targets = [800, 1600, 2400]
    step = 4000
    while step < updates:
        targets.append(step)
        step += 2000
    targets.append(updates)
    return sorted(set(t for t in targets if t <= updates))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=20.0)
    parser.add_argument("--updates", type=int, default=30000)
    args = parser.parse_args()
    if sha(DATASET) != DATASET_SHA:
        raise RuntimeError("Dataset bundle hash mismatch")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = HERE.parent / "runs" / f"cloud_{stamp}"
    run_root.mkdir(parents=True)
    started = time.time()
    deadline = started + args.hours * 3600
    targets = validation_targets(args.updates)
    manifest, archive, bundle_sha = prepare(run_root / "bundle", deadline, args.updates, targets)
    write(run_root / "bundle_manifest.json", manifest)
    key = api_key()
    public_key = (SSH_KEY.with_suffix(".pub")).read_text().strip()
    pod = None
    for gpu in GPU_PREFERENCE:
        request = dict(
            name=f"vawm-av-context-v2-{stamp}", imageName="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
            computeType="GPU", cloudType="COMMUNITY", gpuTypeIds=[gpu], gpuTypePriority="custom", gpuCount=1,
            containerDiskInGb=30, volumeInGb=10, volumeMountPath="/workspace", ports=["22/tcp"], supportPublicIp=True,
            interruptible=False, minRAMPerGPU=16, minVCPUPerGPU=4, env=dict(PUBLIC_KEY=public_key),
        )
        write(run_root / "pod_create_request.json", dict(request, env={"PUBLIC_KEY": "[local public key]"}))
        created = provider_request("POST", "https://rest.runpod.io/v1/pods", key, request)
        write(run_root / "pod_create.json", created)
        if created["status"] in (200, 201) and isinstance(created["body"], dict) and created["body"].get("id"):
            pod = created["body"]
            break
        print("creation failed for", gpu, created["status"], str(created["body"])[:300], flush=True)
    if pod is None:
        raise RuntimeError("No pod could be created")
    pod_id = pod["id"]
    print("pod", pod_id, flush=True)
    ready = None
    for _ in range(120):
        current = provider_request("GET", "https://rest.runpod.io/v1/pods/" + pod_id, key)
        body = current["body"] if isinstance(current["body"], dict) else {}
        if body.get("publicIp") and body.get("portMappings", {}).get("22"):
            ready = body
            break
        time.sleep(10)
    if ready is None:
        provider_request("POST", f"https://rest.runpod.io/v1/pods/{pod_id}/stop", key)
        provider_request("DELETE", "https://rest.runpod.io/v1/pods/" + pod_id, key)
        raise RuntimeError("Pod never exposed SSH; deleted")
    write(run_root / "pod_ready.json", ready)
    host, port = ready["publicIp"], int(ready["portMappings"]["22"])
    known_hosts = run_root / "known_hosts"
    for _ in range(30):
        scan = subprocess.run(["ssh-keyscan", "-p", str(port), "-T", "10", host], capture_output=True, text=True)
        if scan.stdout.strip():
            known_hosts.write_text(scan.stdout)
            break
        time.sleep(10)
    else:
        raise RuntimeError("ssh-keyscan failed")
    config = dict(
        status="provisioned", account_email="jonathan@palladio.ai", pod_id=pod_id, pod_name=pod["name"],
        created_unix=started, deadline_unix=deadline, cloud="COMMUNITY", gpu_requested=ready.get("machine", {}).get("gpuTypeId") or GPU_PREFERENCE[0],
        gpu_usd_per_hour=ready.get("costPerHr"), remote_root="/workspace/vawm_av_context_v2",
        remote_results="/workspace/vawm_av_context_v2/av_context_v2_results", remote_bundle="/workspace/av_context_v2_bundle.tar.gz",
        bundle_sha256=bundle_sha, remote_dataset_bundle="/workspace/bsds500_runtime.tar.gz", dataset_bundle_sha256=DATASET_SHA,
        venv_root="/workspace/av-context-v2-venv", ssh_host=host, ssh_port=port, ssh_user="root",
        ssh_identity=str(SSH_KEY).replace("\\", "/"), known_hosts=str(known_hosts).replace("\\", "/"),
        local_run=str(run_root).replace("\\", "/"), retrieval_grace_seconds=900,
        expected_runtime=manifest["expected_runtime"], target_updates=args.updates, validation_targets=targets,
        target_episodes=args.updates * 40, authorized_arms=manifest["authorized_arms"], arm="attention_context_comparator_v2",
        initialization_kind="full_model_from_scratch", loaded_parent=False,
        cleanup_owner="Detached v2 lifecycle retrieves/verifies this pod only, then stops and deletes it at completion, failure or hard deadline.",
    )
    config_path = run_root / "cloud_provisioning.json"
    write(config_path, config)
    common = ["-i", str(SSH_KEY), "-o", "UserKnownHostsFile=" + str(known_hosts), "-o", "StrictHostKeyChecking=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "KexAlgorithms=curve25519-sha256"]
    destination = f"root@{host}"
    for _ in range(30):
        probe = subprocess.run(["ssh", "-n", *common, "-p", str(port), destination, "echo ready"], capture_output=True, text=True, timeout=60)
        if probe.returncode == 0:
            break
        time.sleep(10)
    else:
        raise RuntimeError("SSH never became ready: " + probe.stderr[-300:])
    for local, remote in ((archive, config["remote_bundle"]), (DATASET, config["remote_dataset_bundle"]), (config_path, "/workspace/av_context_v2_cloud.json"), (HERE / "setup_remote.sh", "/workspace/setup_av_context_v2.sh")):
        subprocess.run(["scp", *common, "-P", str(port), str(local), destination + ":" + remote], check=True, timeout=1800, capture_output=True, text=True)
    print("uploaded bundle, dataset, config, setup", flush=True)
    setup = subprocess.run(["ssh", "-n", *common, "-p", str(port), destination, "sed -i 's/\\r$//' /workspace/setup_av_context_v2.sh && bash /workspace/setup_av_context_v2.sh /workspace/av_context_v2_cloud.json"], capture_output=True, text=True, timeout=2400)
    (run_root / "setup_output.log").write_text(setup.stdout + "\n--- stderr ---\n" + setup.stderr)
    if setup.returncode:
        raise RuntimeError("Remote setup failed; see setup_output.log")
    supervisor_pid = setup.stdout.strip().rsplit("REMOTE_SUPERVISOR_PID=", 1)[-1].strip()
    config.update(status="launched", remote_supervisor_pid=supervisor_pid, launched_unix=time.time())
    write(config_path, config)
    lifecycle_log = (run_root / "lifecycle_detached.log").open("a")
    flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    lifecycle = subprocess.Popen([sys.executable, "-B", str(HERE / "lifecycle_monitor.py"), str(config_path)], cwd=ROOT, stdout=lifecycle_log, stderr=subprocess.STDOUT, creationflags=flags, close_fds=True)
    write(run_root / "launch_receipt.json", dict(
        status="launched", pod_id=pod_id, gpu=config["gpu_requested"], cost_per_hour=config["gpu_usd_per_hour"], deadline_unix=deadline,
        max_cost_usd=(config["gpu_usd_per_hour"] or 0) * args.hours, updates=args.updates, validation_targets=targets,
        bundle_sha256=bundle_sha, remote_supervisor_pid=supervisor_pid, lifecycle_pid=lifecycle.pid, launched_unix=time.time(),
    ))
    write(HERE.parent / "cloud_run.json", dict(run=str(run_root), pod_id=pod_id, deadline_unix=deadline))
    print(json.dumps(dict(pod_id=pod_id, host=host, port=port, gpu=config["gpu_requested"], supervisor_pid=supervisor_pid, lifecycle_pid=lifecycle.pid, run=str(run_root)), indent=1))


if __name__ == "__main__":
    main()
