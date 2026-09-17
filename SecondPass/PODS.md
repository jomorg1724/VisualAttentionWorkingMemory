# How we work on pods

A runbook for RunPod runs in this project, written from the runs of 2026-09-13 to 2026-09-17. It records what actually happened, the scripts that exist, the failure modes we hit, and the rules that follow from them. Read this before provisioning anything.

## 1. Rules

1. **No pod without an explicit instruction from the user.** Laptop first: a run goes to a pod only after a laptop-scale version has left chance, or when the user says so.
2. **One finite budget per pod, stated before launch**: hours, GPU, price per hour, and the maximum cost. The remote program must stop starting new work before the deadline, and something on our side must stop and delete the pod after it.
3. **Every input is hash-pinned**: the source bundle, the dataset bundle, and the config. The remote setup verifies both bundles with `sha256sum -c` before doing anything else.
4. **Everything that comes back is verified**: archives are checked against a remote `sha256sum`, checkpoints against per-file digests. A pod is deleted only after a verified retrieval, unless the user says `--force`.
5. **Receipts for every step**: provisioning, launch, pulls, finalisation. If a step happened by hand (it did, twice), the receipt says so and how it was verified.
6. **Fetch during the run, not only at the end.** Losing a pod loses whatever was only on it.

## 2. Accounts, keys and tools

- RunPod account: the API key is read from `C:\Users\jomor\.runpod\config-palladio.toml` (`apikey = ...`) by `WorkingMemory/cloud_shutdown.py: api_key()`. Never print it, never put it in a receipt.
- SSH: key pair `~/.runpod/ssh/runpodctl-ssh-key(.pub)`. The public key is passed to the pod as the `PUBLIC_KEY` env at creation.
- REST API: `https://rest.runpod.io/v1/pods` (POST create, GET status, POST `/stop`, DELETE). `provider_request()` in `cloud_shutdown.py` wraps it; a deleted pod answers 404 on GET, which is our proof of deletion.
- Image: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`; we build our own venv inside it with the project's pinned stack (torch 1.13.1+cu117, numpy 1.23.1, scipy 1.8.1, Pillow 9.1.1).
- GPU preference: 4090, then 3090, then A5000, community cloud, non-interruptible, 1 GPU, 30 GB container disk, 10 GB volume at `/workspace`, port 22/tcp. On 2026-09-17 no 4090 was available and the 3090 cost $0.22/h.

## 3. Windows-specific gotchas (these cost us hours)

- **Run launchers, watchers and finalisers from PowerShell**, not Git Bash. Git's MSYS `ssh` rewrites backslashes and paths in remote arguments and broke two retrievals on 2026-09-15.
- **Windows `ssh-keyscan` returns nothing for RunPod hosts** (2026-09-17). Git's `ssh-keyscan` works. Write `known_hosts` with Git's keyscan (drop the `#` comment lines), then let the Windows `ssh`/`scp` use it with `-o UserKnownHostsFile=... -o StrictHostKeyChecking=yes`. `cloud/launch.py` reuses an existing non-empty `known_hosts` and skips the scan.
- **Always pass `-o KexAlgorithms=curve25519-sha256 -o BatchMode=yes -o ConnectTimeout=20`**; without BatchMode a prompt can hang a detached process forever.
- **The remote setup script must be CRLF-cleaned** before bash runs it: `sed -i 's/\r$//'`. The launcher does this.
- **Detached local processes die with the Claude session.** The 2026-09-17 watcher (started with `Start-Process -WindowStyle Hidden`) stopped at 02:51 when the session ended; the pod finished at 03:46 and idled about 8 hours. Two fixes, use at least one: make the remote side self-terminating (a remote finaliser that packs results and calls stop/delete itself), or register the watcher as a Windows scheduled task or service. Check it is alive before assuming it is.

## 4. The scripts (accumulator program, `WorkingMemory/PlainBaseline/cloud/`)

| Script | What it does |
|---|---|
| `launch.py --hours H --seeds 1,2 [--attach <run_dir>]` | Builds a hash-pinned bundle of only the needed sources (self-checks that it imports), creates the pod, waits for IP and port, keyscans (or reuses `known_hosts`), writes `cloud_provisioning.json`, uploads bundle + dataset + config + `setup_remote.sh`, runs the remote setup, records the remote pids. `--attach` continues from an already-created pod. |
| `setup_remote.sh` | On the pod: verify bundle and dataset hashes, build the venv, unpack the dataset, assert CUDA, profile 3 updates of one accumulator model, then start two `program.py` lanes with `nohup` and write `program.pids`. |
| `watch.py <config>` | Every 10 minutes: pack results on the pod excluding `*.pt`, scp, verify, extract to `<run>/pulled/`. When both lanes report `completed`/`deadline` or the deadline plus grace has passed: final pull, checkpoint pull with digests, stop, delete, confirm 404, write `watch_receipt.json`. Leaves the pod running if the final retrieval fails. |
| `finalize.py <config>` | One-shot version of the watcher's end game, for when the watcher died: stop remote processes, verified pull, checkpoint pull, stop/delete/404, `finalize_receipt.json`. |
| `status.py`, `tabulate.py` | Print the pulled validation looks and the test tables from the program receipts. |
| `../../cloud_shutdown.py <config> [--force]` | The older generic stop-with-retrieval for the v2 lineage's layout; refuses to delete on failed retrieval unless `--force`. |

The older v2 lifecycle scripts (`WorkingMemory/AttentionContextComparator/V2/cloud/`) do the same job for that experiment's worker and are kept for reference; their remote layout differs (`checkpoint_index.jsonl`).

## 5. The remote program

`WorkingMemory/PlainBaseline/program.py` runs, per (arm, seed), the sequence ring gate -> D0 curriculum -> delay ladder, each stage a `baseline.py` subprocess with its own `receipt.json`, and skips finished stages on restart, so a lane can be resumed. It refuses to start a stage within 20 minutes of the deadline and marks it `not_started_deadline`. Two lanes share one GPU (plain+convgru, opponent+kda); on the 3090 a full lane took 87 and 126 minutes.

Memory: with three stacked frames and 28-frame trials the accumulator arms need 2.9 to 5.5 GB per chunk of 32 on a 24 GB card; on the laptop's 8 GB use chunk 16 and never more than two GPU processes.

## 6. Procedure, start to finish

1. Laptop smoke test of the exact program at tiny scale (`--scale 0.004 --gate 0 --val-n 16 --test-n 16`); every stage of every arm must complete.
2. `python -B WorkingMemory/PlainBaseline/cloud/launch.py --hours 6 --seeds 1,2` from PowerShell. If it fails after the pod exists, do not create another pod: fix and `--attach <run_dir>`.
3. Verify by ssh that the lanes are running (`ps -eo pid,etime,cmd | grep PlainBaseline`, `nvidia-smi`). If the launcher crashed after remote setup, write the launch receipt by hand and say so in it.
4. Start the watcher from PowerShell, then confirm the first line of `watch.log` says `pulled`. Prefer a scheduled task.
5. Read progress with `python -m WorkingMemory.PlainBaseline.cloud.status`; never ssh in to look at metrics by hand while the watcher owns the pod.
6. At the end, check `watch_receipt.json` (or run `finalize.py`): `stop 200, delete 204, lookup 404`, `final_retrieval: verified`, and a non-zero checkpoint count if checkpoints matter. Then tabulate, write the journal page, commit.
7. Confirm on the RunPod dashboard that the account has no pods.

## 7. What went wrong, dated

- 2026-09-15: retrieval from two pods failed because Git's ssh mangled remote paths; the shutdown script deleted anyway; all remote checkpoints and full metrics of both arms were lost. Fix: `cloud_shutdown.py` now refuses to delete on failed retrieval without `--force`; run from PowerShell.
- 2026-09-16: overnight v2 pod stopped by the user as a negative result; retrieval of 49 checkpoints verified before deletion. This is the reference for a clean shutdown.
- 2026-09-17: `ssh-keyscan` stall, launcher crash after successful setup, watcher death with the session, about 8 idle hours, and a checkpoint fetch that returned zero files (the remote `find ... | xargs sha256sum` listing came back empty; cause not reproducible because the pod is gone). Results were verified and are local; the terminal weights are not. Fixes: `--attach`, `finalize.py`, this runbook, and the two rules above about self-termination and fetching during the run.

## 8. Cost log

| Date | Pod | GPU | $/h | Hours | Note |
|---|---|---|---:|---:|---|
| 2026-09-17 | `hy2m2tjf1awuhf` | RTX 3090 | 0.22 | ~10.8 | ~2.8 h of work, ~8 h idle after the watcher died |

Earlier pods (2026-09-13 to 09-16) are recorded in their own run directories under `WorkingMemory/*/runs/cloud_*/` and in the chronology.
