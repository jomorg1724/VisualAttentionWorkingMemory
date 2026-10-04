# Visual memory textbook companion

Start with `output/pdf/visual_memory_textbook.pdf`. Editable prose is in
`visual_memory_textbook.md`.

This is a standalone reading edition of the supplied ten-page architecture note.
It explains the implemented spatial-KDA/vector-GRU learner, the planned final
ConvGRU branch, and all 13 training tasks / 35 primary conditions. The new branch
is deliberately described as a design under construction, per the user's current
clarification. This document does not launch or evaluate models.

## Build and check

From this directory, using Python with reportlab, PyMuPDF and Pillow, plus pandoc
and tectonic on PATH:

```sh
python3 build_reading.py
```

The layout uses installed Times New Roman, Arial and Menlo fonts. `draw_diagrams.py`
creates vector diagrams; `verify_reading.py` renders every page, checks text bounds,
checks all task sections against the catalog, and checks architecture arithmetic.
`verification.json` records those checks; `source_manifest.json` identifies the
local code/design sources. Visual inspection complements programmatic checks.

No existing model, training code, checkpoint, source report or agent-owned work
was edited. All files for this reading reside in this new directory.
