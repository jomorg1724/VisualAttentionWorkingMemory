# Fresh local weighted-mean CNN–GRU

User authorized local training on October 3. `launch.py` installs one bounded
launchd supervisor and an independent deadline guard. It copies only source files
to a new runtime; no weights, optimizer, RNG or data state is inherited.

The MPS profile itself validates native sequence forward/backward/Adam; its states
are discarded. The measured profile pins complete 1,000-trial / ten-epoch pools
before starting a separate freshly initialized production process.
One worker owns the shared local GPU lock. Effective batch32/micro4, FP32/full BPTT,
Adam1e-4/no clipping, at mosttwoCPUthreads. Same modified single-stimulus task.
The first profile starts an immutable eight-hour cap, including evaluation/reporting.

Runtime: `/Users/jonathanmorgan/VAWMRuntime/weighted_mean_conv_gru_local01/run`.
Deadline: October3,3:40:04PM Pacific. Planned exposure2310updates/70000presentations/
7000unique movies. The old cloud queue remains disabled; no paid compute is running.
