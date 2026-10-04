# Verification execution record

Before accelerator budget began, using the declared runtime Python and all numerical thread limits1:

- RED: `test_control.py` failed on its deliberate assertion that `control.py` did not yet exist (1 failure,0.65s).
- GREEN after implementation:1 passed in2.01s.
- Native batch exactly equals direct renderer replay under810001; label1 iff metadata event is target.
- Batched versus individual predictions within tolerance.
- Ordinary whole-batch versus independent microbatch accumulation maximum parameter-gradient error:1.0728836059570312e−6.
- First-frame absolute input-gradient sum:0.26218298077583313; final-frame:0.25920403003692627.
- Finite connected gradients for every parameter tensor; every parameter tensor changed on an AdamW update.
- Native CPU loss:0.6681342720985413.

Environment discovery: macOS,38,654,705,664 bytes memory; Torch2.8.0,NumPy2.4.6,Python3.11.15; MPS available. Runtime Python `/Users/jonathanmorgan/VAWMRuntime/recurrent_transformer_cpu_env/bin/python`.

These checks demonstrate correct computation paths and accumulation, not task acquisition. MPS-specific profile results are in checks.json; fit/generalization outcomes must be read from result.json rather than inferred from these smoke tests.
