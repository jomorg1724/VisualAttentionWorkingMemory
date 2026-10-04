# Training — single-stimulus conv RViT

**CANCELLED BY USER — cloud pod stopped and deleted.** Last saved checkpoint 1,700 CPU reloaded with all92 Adam states; 66 artifacts retrieved/hash verified. Last observed live update 1726; last100-update mean loss 0.68365. No final held-out evaluation was run. Exact podjxbmb44y9wamhl deleted; account pod list verified empty. Earlier training notes below are historical.

Fresh production is verified on one A40 pod `jxbmb44y9wamhl` at $0.49/hour.
Production checkpoint **3 / 96 presentations** was downloaded, hash verified and
CPU reloaded: all **92 Adam states and learned parameter tensors advanced** from
a direct fresh initialization. No predecessor or profile state was inherited.

Same original conv RViT architecture, full-sequence BPTT, FP32/TF32 off,
Adam1e-4/no clipping, effective batch32/micro4. One patch at the native center
(20,50) or(80,50), cue ring and other patch removed; all **29/37/45 frames**,
retained target-patch pixels and direction-change timing/properties preserved.
Labels remain57%change/43%nochange. This is a modified task, not the native benchmark.

CUDA profile pins the full matched target: **2,310 updates /70,000 presentations /
7,000 unique movies**. Every1,000-trial pool gets ten shuffled epochs.
First production snapshot **update6 /192 presentations**, loss **0.68173**.
First validation at100; no performance result yet.

New8h/$5 creation cap: 2026-10-03T03:28:02.439784+00:00 to2026-10-03T11:28:02.439784+00:00,
**October3,4:28:02 AM Pacific**. Science deadline2026-10-03T11:18:02.439784+00:00; final600seconds
reserved for retrieval. Independent authenticated guard/off-pod mirror active.
The random-frame cloud and structured-motion local runs continue within their
original caps. No extra arm or automatic cap renewal.

[Design](README.md) · [Stimulus preview](stimulus_preview.png)
· [Focused pixel/timing/label check](check_results.json)
· [Production evidence](CloudRuntime/production_verified.json)
· [CPU checkpoint reload](CloudRuntime/downloaded_checkpoint_verified.json)
· [Mirrored artifacts](CloudRuntime/artifacts/)
