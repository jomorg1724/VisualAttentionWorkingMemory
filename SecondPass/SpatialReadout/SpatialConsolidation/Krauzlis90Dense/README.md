# Whole-fresh dense Krauzlis ±90 competitor — dense01

**Prepared and CPU-verified; not launched.** Parent alone owns rental. No prior weights, optimizer state or profile state enters production. Existing model sources/archives/pods are untouched.

## Architecture and tradeoff

Native RGB 100×100, center by −0.5, causal stack3 with centered-zero startup padding; flatten the entire 9×100×100 input to 90,000 values. One ordinary `nn.Linear(90000,128)` + ReLU globally encodes the whole image. Three serial global DenseKDA residual blocks follow. Each uses ordinary `nn.Linear(128,82)` for packed q/k/v/alpha/beta and `nn.Linear(32,128)` for output; residual is `ReLU(x + output(read))`. Two heads, key8, value16; q/k L2 normalized with eps1e−6; alpha/beta weights start zero, biases initialize retention .9/write .5. The exact existing `kda_update` decay/correction/read arithmetic is reused. Its imported source module also defines legacy spatial classes, but none are constructed or called here.

Then a vector128 dense GRU (Linear256→256 write/reset, Linear256→128 candidate, zero gate/candidate biases), Linear128→128/ReLU readout, Linear128→2 task head. Only the native task head exists; there are no inactive multi-task heads. No terminal transformer, convolutions, patch tokens, unfold, pooling, resizing, grayscale, local weight sharing or detached recurrence. All learned projections are `nn.Linear`. The width is a single prespecified manageable design, not a sweep or exact parameter match; live native profile determines update allocation, not architecture choice.

Measured construction: **11,679,992 parameters /22 tensors**. Encoder11,520,128; KDA44,406; GRU98,688; readout16,512; head258. FP32 weights46,719,968 bytes; Adam moments93,439,936 bytes (+88 scalar-step bytes). Global persistent recurrent state896 FP32 values/episode (three2×8×16 KDA states plus128 GRU); batch32 state114,688 bytes, excluding autograd/full-BPTT activations and input stacks. Real CPU production checkpoint3 is140,213,720 bytes. GPU peak memory/throughput not measured.

This is a **package comparison, not isolated convolution causality**: global compression, state size, capacity, residual pathway and absence of terminal spatial transformer differ. No spatial KDA map or localized-neuron interpretation is implied.

## Fixed scientific contract

Exactly the previous native cue, ±90 renderer, timing, target/foil/catch rule, three B12/B20/B28 conditions and cross-entropy are reused. Metadata is scoring-only. Native RNG magnitude draw is consumed before override; post-event RNG can diverge from the unmodified small-magnitude task. Renderer tests retain paired preevent/catch parity.

Adam lr1e−4, betas(.9,.999), eps1e−8, weight_decay0; batch32/micro4, fp32, TF32 off, no clipping, full BPTT. No extra CNN arm, task teaching, auxiliary labels or architecture sweep. Target **7,737 updates /247,584 episodes**, compared descriptively with fresh02's completed exposure. Profile is six disposable native updates (two/cell), eval20/cell; use per-cell maximum times, training×1.25/eval×1.35 and startup/checkpoint/finalization reserves. Pin a lower target only before production if needed; fewer than1,000 updates fails closed. Two validation looks (midpoint and terminal),100/cell; selected and terminal final tests200/cell, new final-only draws; same-step results may be reused after weight equality. Selection: mean three-cell AUC, then BA, earlier ties. Actual allocation is published before production.

Init/CUDA/scheduler seeds match fresh02 numerically but all streams start from zero. Architecture RNG consumption differs. Train/validation independent stream namespaces128492763/128592763 match fresh02; final namespace138692763 is new, so this is not paired identical final-test inference. Model RNG does not seed procedural task draws. No inherited checkpoint input is supported. CPU step-zero and native persisted verifier each compare every model tensor with the seeded direct constructor and require empty initial Adam/streams/selection. Step3 verification requires all22 parameter tensors changed and every named Adam state at3; all seven major groups advance.

## Launch and safety

Parent executable: `/Users/jonathanmorgan/VAWMRuntime/cloud_krauzlis90_dense01/launch.sh`.
It uses the existing explicitly approved local guard credential only in memory and existing SSH key; no secret is packaged. It performs prepare→bounded profile→allocation pin→fresh production, binds startup readiness to actual checkpoint3 digest/full Adam/native verification/timing, downloads and verifies the checkpoint before returning success. Guard independently owns deadline/completion/failure stop and private status publication; successful completion stops compute while retaining disk without waiting for laptop retrieval. No healthy-production stop smoke test is run. Guard authentication/installation must still be verified on the new pod by parent launcher.

NEW eight-hour origin is new creation, scientific deadline10min earlier. Single A40≤$0.49/h, $1 storage reserve, total≤$5; no active parallel rental. The old stopped pod is never restarted/modified. Old retained storage remains a separate pre-existing charge, not new compute; parent must account for it separately. New storage reserve covers running storage and24h retained-volume horizon. Retention beyond that remains billable; no automatic deletion is authorized. Provider API failure can delay verified stop; independent guard is not a guarantee against provider failure.

## CPU evidence

Actual native six-update profile→fresh reinitialization→three production updates at batch32/micro4, two CPU threads: **1 passed in21.92s**. Production losses .6915817782/.6940824389/.6922373399;96 episodes; all22 named Adam states at3; direct constructor equality; native verify CLI, corruption rejection, immutable checkpoint rejection, parent download verification and real `deploy.production_ready` all passed. Matched profile ratio .984834662. Evidence/checkpoints live only outside archives under `cloud_krauzlis90_dense01/native_cpu_evidence.json` and `cpu_native/`.

Additional scoped tests cover dense forward/backward, runtime-forbidden functional local operations, frame-stack semantics, renderer parity, budget/handoff/no-restart, tiny selected/terminal finalization and mocked guard/deploy contracts. CPU timings/results are software evidence, not accelerator feasibility or acquisition results. No cloud training metrics exist for dense01.
