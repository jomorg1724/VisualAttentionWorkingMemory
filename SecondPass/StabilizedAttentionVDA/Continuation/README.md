# Explicit VDA resumption — October 9, 2026

The user requested continuing the paused VDA training. Resume run01/latest1280
with exact model, all86 Adam states, task/condition scheduler, native streams,
Python/NumPy/CPU/CUDA random state. Keep the16,000 cumulative VDA-update target;
14,720 updates remain. Model, loss, learning rate, clipping, batch size, pixels,
labels, sampling and validation cadence are unchanged. No baseline re-evaluation
or initialization from the older13-task checkpoint. Next validation1600, then
every320 and terminal; selected final test256 per condition.

Original run01 best/latest1280 and its source files remain read-only. New runtime
is `LocalRuntime/continuation01`. Best1280 remains eligible and is copied into
the new runtime; new best replaces that copy only on improved validation AUC.
All checkpoint state needed for a future continuation is saved every16updates
and at graceful stop. The old stop request remains with run01; a new request
belongs in continuation01/stop_requested.json.

The launcher carries forward only the unused original wall-time allowance:
floor(original supervisor deadline minus its exit time),26,607seconds
(7h23m27s), starting at this explicit activation. Thirty seconds are reserved
for supervisor shutdown. No automatic extension/restart/cloud. Doom remains
prepared only and is not launched by this continuation.

Two CPU tests passed: exact weight/optimizer/scheduler/stream/RNG restoration,
next16 native task batches, an identical disposable Adam update after reload,
unchanged source optimizer state, and remaining-allocation accounting. Original
VDA and shared dependency source hashes are checked before activation. CUDA RNG
restore and persisted production updates require runtime receipts; a launched
supervisor alone is not proof that training has advanced.

Explicit launch command (do not launch a duplicate):

```powershell
C:/Python310/python.exe -X utf8 -u -m SecondPass.StabilizedAttentionVDA.Continuation.launch --source SecondPass/StabilizedAttentionVDA/LocalRuntime/run01 --run SecondPass/StabilizedAttentionVDA/LocalRuntime/continuation01
```
