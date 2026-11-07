# Weighted-mean CNN–GRU cloud queue

This is a one-shot automatic cloud queue, not an active training run.
The predecessor is the single-stimulus RViT pod `jxbmb44y9wamhl`.
`queue_after.py` waits for its completed final retrieval and verified deletion,
then provisions one fresh A40 and launches prepare → profile → pin → production.
The existing random-frame pod `5us0rp5jwmg5bu` may continue under its original cap.
Waiting creates no rental and starts no training allowance.

The next run uses the same 8-hour/$5 maximum as the last diagnostic; its clock
starts at pod creation, including setup, profiling, evaluation and retrieval.
Exposure targets 2,310 updates / 70,000 presentations / 7,000 unique movies and
is pinned before production from measured throughput, in complete replay pools.
Weights, optimizer, RNG and stream counters start fresh; profile state is discarded.
The independent pod guard stops at completion/deadline; the parent mirror retrieves
and verifies artifacts before deleting this exact ephemeral pod.

`queue_status.json` records waiting/activation. Training is claimed only after
`production_verified.json` and a downloaded checkpoint establish actual Adam progress.
The claim files prevent retries or duplicate rentals. An unresolved predecessor
cleanup or failed deployment ends this attempt without a new retry or cap extension.
