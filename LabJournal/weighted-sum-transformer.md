# Direct weighted-image transformer change classifier

October 4, 2026. After the completed weighted-image VAE experiment, the user requested a direct method: patchify the same weighted representation, process it with one transformer layer, a CLS token and four heads, then feed the first and second clips' CLS outputs to an FFN that predicts change/no change. This new request supersedes the preceding end-of-night closure for this one experiment.

## Final result — completed October 4, 9:45 PM PDT

All 25,024 updates finished in 44 min 46 sec: 391,000 unique movies, 782,000 pair presentations. Selected checkpoint 5,000 by minimum validation BCE; latest 25,024 preserved. Final 3,072 presentations / 1,536 paired contexts: **BA 49.25%, AUC 0.492292, BCE 0.694917**; zero-logit BA 49.77%. All six condition cells remain near chance (48.05–50.59% BA). This direct model did not solve the task at this exposure.

Root independently verified best/latest weights and all 24 optimizer states aligned to their saved steps, with finite tensors. Worker16652, supervisor32432 and guard39176 exited normally, well within the original four-hour cap. No new run is queued. This is a final held-out result; launch/live statements below are historical. It does not establish that the weighted image itself is fundamentally insufficient or that every transformer design would fail.

See [final report](../SecondPass/WeightedSumTransformer/LocalRuntime/run01/FINAL_REPORT.md), [full results](../SecondPass/WeightedSumTransformer/LocalRuntime/run01/final_test.json) and [root verification](../SecondPass/WeightedSumTransformer/LocalRuntime/run01/root_final_verification.json).

## Architecture and learning signal

### Post-run skeptical audit

A read-only CPU audit identified severe CLS attention concentration: at selected checkpoint 5000, one head assigns 92.69% mean mass to CLS itself and another picks the same patch in 175/192 sampled clip images. At latest 25024, one head attends effectively only to CLS. This is a measured failure signature, not a demonstrated cause or remedy. Saved scores reproduce, all parameter tensors changed, checked stimuli/labels agree, and train/eval numerical differences are small. The carried-over small 26/28-degree changes and missing prelaunch tiny-dataset learning test are explicit methodological limitations. No training occurred during this audit; chance performance does not establish that the weighted representation is insufficient. See [audit report](../SecondPass/WeightedSumTransformer/LocalRuntime/run01/Audit/REPORT.md).

Each chronological three-frame clip becomes `(oldest + 2 middle + 3 newest)/6`. This is the same reversible weighted RGB representation. A learned linear patch embedding uses nonoverlapping 10 by 10 patches, giving 100 image tokens of width128. Add a learned CLS token and learned positional embeddings, then apply exactly one shared transformer encoder layer with four attention heads, a512-dimensional internal feedforward network, pre-normalization, GELU and zero dropout. Final normalization supplies the128-dimensional CLS output.

Both clips use identical encoder weights. They are separate examples in the encoder batch and cannot attend to one another. Ordered concatenation `[CLS_before, CLS_after]` feeds `256 -> 256 GELU -> 128 GELU -> 1`. The entire model trains from scratch using binary cross-entropy on balanced change/no-change labels. No reconstruction, KL, explicit direction/angle target, contrastive objective, frozen learned weights or historical checkpoint transfer is included.

Fresh191101/191102/191103 namespaces preserve the original continuous six-frame dot movie, speeds0.375/1/2 and changes26/28degrees. Metadata is analysis-only. Each training pair has independently drawn nuisance variables; evaluation pairs unchanged/changed movies with identical nuisance context and before clip. All six speed/angle cells remain separately reported.

## Execution and evaluation

One local GPU worker, FP32, TF32 disabled, at most two CPU threads, one Adam learning rate3e-4 and epsilon1e-8, no weight decay or gradient clipping. Requested25,024 updates correspond to391 pools of1,000 fresh pairs reused for two shuffled epochs:391,000 unique movies/782,000 presentations. A new finite four-hour allowance starts at this model's first GPU profile, covers training/evaluation/reporting, and cannot extend automatically. Native profiling can reduce exposure in complete64-update pools before production. There is no cloud or extra comparison arm.

Validation checkpoints at64/5,000/10,000/15,000/20,000/terminal select minimum BCE. A validation-only positive-sign threshold is fixed for fresh final evaluation; zero-logit decisions are also reported. Final test has3,072 presentations in1,536 paired nuisance contexts. Checkpoints preserve optimizer/RNG/data progress, with independent owned-process deadline enforcement.

CPU checks and an independent actual-source review cover the requested architecture, all trainable gradients, independent shared branches and the actual production path through validation/checkpoint saving/final evaluation. Evidence of software correctness is not evidence of task success.

## Actual execution

**Training locally:** supervisor32432 launched at9:00:16PM PDT; production worker16652 and independent guard39176. New shared budget starts2026-10-04T21:00:19.489855-07:00 and ends2026-10-05T01:00:19.489855-07:00 (October5,1:00:19AM PDT). Native profiling pinned the full25,024updates/391,000unique pairs/782,000presentations; profile weights were discarded. Measured update0.04228seconds plus1.88566seconds/1,000-pair generation; evaluation0.003149seconds/pair with622.67seconds reserve and1.25safety margin.

Architecture checks and complete CPU production fixture passed64updates, validation/checkpoint saving, best selection and final paired12-presentation engineering evaluation. Independent actual-source review passed. Actual GPU validation64 and checkpoint saving succeeded; the root independently loaded best64/latest250 with all24 aligned Adam states and finite weights/optimizer values, then observed live training beyond1024. Source hashes still match the frozen snapshot/profile. Startup validation at64 was near chance (BA52.08%, AUC0.50456), a software-health snapshot rather than a trained outcome. Planned next validation5,000; no final score exists yet.

Runtime evidence is in `SecondPass/WeightedSumTransformer/LocalRuntime/run01/`: `profile.json`, `production_config.json`, frozen source snapshot, `root_startup_verification.json` and owned-worker/guard receipts. There is one fresh supervised model, no cloud or extra arm.

[Current status](CURRENT_STATUS.md) · [Previous VAE result](weighted-sum-vae.md)
