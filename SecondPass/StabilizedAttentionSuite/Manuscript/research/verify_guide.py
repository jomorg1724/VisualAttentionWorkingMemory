"""Read-only integrity checks for the frozen interactive guide assets."""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parents[2]
ASSETS = HERE / "assets/guide"
receipt = json.loads((ASSETS / "provenance.json").read_text())
catalog = json.loads((ASSETS / "catalog.js").read_text(encoding="utf8").removeprefix("window.GUIDE=").removesuffix(";"))
for name, expected in receipt["source_hashes"].items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name

groups = frames = conditions = trials = gifs = 0
recognition = []
for task in catalog["tasks"]:
    text = (ASSETS / task["id"] / "data.js").read_text(encoding="utf8")
    data = json.loads(text.split("\n", 1)[1].split("=", 1)[1].removesuffix(";"))
    assert len(data["cells"]) == task["conditions"]
    assert sum(c["n"] for c in data["cells"]) == task["trials"]
    assert abs(np.mean([c["accuracy"] for c in data["cells"]]) - task["accuracy"]) < 1e-10
    conditions += len(data["cells"])
    trials += task["trials"]
    for g in data["groups"]:
        count = len(g["time"])
        for field in ["phases", "example_map", "mean_map", "mean_mass", "ci_low", "ci_high", "temporal"]:
            assert len(g[field]) == count, (task["id"], field)
        for field in ["example_map", "mean_map"]:
            array = np.asarray(g[field])
            assert array.shape == (count, 10, 10)
            assert np.isfinite(array).all() and (array >= 0).all()
            np.testing.assert_allclose(array.sum(axis=(1, 2)), 1, atol=1e-5)
        with Image.open(HERE / g["sprite"]) as image:
            assert image.size == (count * 100, 100)
        assert g["choice"] == int(np.argmax(g["probabilities"]))
        if g["gif"]:
            with Image.open(HERE / g["gif"]) as image:
                assert image.n_frames == count
                for i in range(count):
                    image.seek(i)
                    assert image.info["duration"] == 200
            gifs += 1
        if task["id"] == "image_recognition":
            recognition.append({"group": g["title"], "frames": count,
                                "study_frames": g["phases"].count("study"),
                                "probe_frames": g["phases"].count("probe")})
        groups += 1
        frames += count
assert (len(catalog["tasks"]), conditions, trials, groups, frames, gifs) == (19, 437, 55936, 63, 888, 6)
assert (HERE / "index.html").read_bytes() == (HERE / "explainer.html").read_bytes()
result = dict(passed=True, source_hashes_verified=len(receipt["source_hashes"]),
              tasks=19, conditions=conditions, trials=trials, groups=groups,
              frames=frames, gifs=gifs, recognition=recognition)
(HERE / "research/guide_integrity_verified.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
