# Five-task training with the original attention biases

The user cancelled the bias-free five-task run, then asked to restore the bias terms and launch training again. This separate run starts from the **original intact attention8400 checkpoint**, not the cancelled run's damaged or partially trained weights. The cancelled run and all completed old-task results remain preserved.

Attention logits retain the original learned source preference and distance penalty: `QK^T/sqrt(32) + source_bias - softplus(raw_locality)*distance_squared`. Both bias parameters remain trainable. All original model tensors and123 compatible Adam states are copied exactly, including the six bias scalars. The five fresh128-to-class heads are exactly the same initial tensors as in the cancelled arm. Total parameters:557,676. A CPU check verified exact equality to the original attention computation.

Each optimizer update averages five batch8 cross-entropies, then performs one gradient clip and Adam step. Tasks, task-local streams, seeds, constant learning rates, activation checkpointing, fp32/full BPTT and selection are unchanged from the five-task protocol. The target is4,000updates /160,000episodes. This is new-task acquisition, not a rerun of the old three-task experiment.

The existing healthy L40 pod is reused. No additional pod or budget is created. The original absolute deadline remains2026-09-14 09:36:25UTC. A fresh profile confirms the complete target fits the remaining time. Runtime sources and output directory are versioned separately; remote results are in `/workspace/vawm_unbiased/biased_results`. The old `/workspace/vawm_unbiased/remote_results` is cancelled and immutable.

The new watcher incrementally retrieves hash-indexed checkpoints and verifies the complete reconstructed final artifact set. Root's connected-API heartbeat owns pod stop/delete after this new run finishes or reaches the original deadline. The older cancellation receipt must not trigger deletion while this run remains active.
