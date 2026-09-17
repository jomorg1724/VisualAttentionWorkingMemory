# Cloud motion-allocation arm

This folder owns packaging, deployment and artifact retrieval for the50%-motion continuation. The common implementation and local10%-motion arm are owned in the parent TrainingExposure directory. No architecture, objective, calibration or residual change is included.

Both arms start from the selected attention8400 checkpoint, preserving compatible model/Adam/task-local stream/RNG progress, with4000 added updates × batch8 =32,000 episodes. The focused80-update cycle contains five visits to each of eight single/binding delay cells and20 visits to each of two motion delay cells. All shared protocol values, validation selection and fresh paired evaluation seeds come from the common worker's pinned configuration.

Cloud provisioning waits for actual local GPU launch evidence. One Palladio RTX3090 is allowed, with a four-hour absolute deadline from creation including setup, profile, training, evaluation and retrieval. The scientific environment is Python3.10, Torch1.13.1+cu117, NumPy1.23.1, SciPy1.8.1 and Pillow9.1.1. Training is fp32 with TF32 disabled. This matches package versions; different hardware/platforms are not bitwise-equivalent training.

The private SSH key remains local. Only its existing public key is supplied to the pod. Source/checkpoint bundles and returned archives are checksummed. The remote worker has a finite absolute cutoff; the connected-API coordinator heartbeat separately stops billing on completion or deadline. A worker exit is not a billing stop.

`watch_remote.py` retrieves completed or failed run evidence, checks the archive and every indexed file, and writes retrieval_receipt.json. Cloud cleanup proceeds as soon as retrieval is verified, even if local training continues. Comparative analysis belongs to the common researcher and waits for both arms. No paid volume or pod is retained after successful retrieval.
