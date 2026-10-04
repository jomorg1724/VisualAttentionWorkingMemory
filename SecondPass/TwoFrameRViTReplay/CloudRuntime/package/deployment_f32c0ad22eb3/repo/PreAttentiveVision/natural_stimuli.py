"""Native BSDS500 photo crops with phase-preserving spectral detail differences.

Data download is explicit (`python -m PreAttentiveVision.natural_stimuli --prepare`).
Sampling is CPU-only and consumes the caller's NumPy Generator, not global RNG.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile
import time
import urllib.request

import numpy as np
from PIL import Image


DATA_ROOT = Path(__file__).resolve().parent / "data" / "bsds500"
SOURCE_URL = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
SOURCE_PAGE = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html"
PROTOCOL = {
    "version": "natural_spectrum_bsds500_v1", "task": "natural_spectrum", "classes": 2,
    "crop_size": 100, "upsampling": False,
    "input_photometry": "sRGB decoded to linear RGB, luminance weights .2126/.7152/.0722",
    "beta_center_range": [-.5, .5], "beta_deltas": [.15, .30, .60],
    "frequency_reference_cycles_per_pixel": .1, "target_rms": .15,
    "mean": .5, "safe_peak_deviation": .45,
    "label": "index of frame with smaller beta / greater relative high-frequency weight",
    "boundary": "periodic FFT of native100 crop; no window or resize",
    "source_splits": "official train200/val100/test200; no photos cross splits",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_dataset(root=DATA_ROOT):
    """One official archive; extract only regex-selected ordinary photo members."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    archive = root / "BSR_bsds500.tgz"
    if not archive.exists():
        partial = root / "BSR_bsds500.tgz.part"
        began = time.monotonic()
        request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "PAV-research/1.0"})
        with urllib.request.urlopen(request, timeout=30) as response, partial.open("wb") as output:
            total = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > 300 * 1024 * 1024 or time.monotonic() - began > 300:
                    raise RuntimeError("BSDS download exceeded finite size/time limit")
                output.write(block)
        partial.replace(archive)
    pattern = re.compile(r"BSR/BSDS500/data/images/(train|val|test)/(\d+\.jpg)")
    records = []
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            match = pattern.fullmatch(member.name)
            if match is None:
                continue
            if not member.isfile() or member.size > 10 * 1024 * 1024:
                raise ValueError("Invalid image archive member")
            split, filename = match.groups()
            destination = (root / "images" / split / filename).resolve()
            destination.relative_to(root)  # resolved path containment, no extractall/symlinks
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(f"Refusing ambiguous partial dataset overwrite: {destination}")
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError("Unreadable archive image")
            with source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)
            with Image.open(destination) as image:
                width, height = image.size
                if min(width, height) < 100:
                    raise ValueError("Source photo too small for native crop")
            records.append({"split": split, "file": destination.relative_to(root).as_posix(),
                            "base_id": f"BSDS500/{split}/{filename}", "width": width, "height": height,
                            "bytes": destination.stat().st_size, "sha256": sha256(destination)})
    counts = {split: sum(row["split"] == split for row in records) for split in ("train", "val", "test")}
    if counts != {"train": 200, "val": 100, "test": 200}:
        raise ValueError(f"Unexpected official split counts: {counts}")
    if len({row["sha256"] for row in records}) != 500:
        raise ValueError("Duplicated image file bytes across BSDS500")
    manifest = {"dataset": "BSDS500", "source_page": SOURCE_PAGE, "download_url": SOURCE_URL,
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "archive_bytes": archive.stat().st_size, "archive_sha256": sha256(archive),
                "checksum_scope": "locally computed identity, not an independently published checksum",
                "split_counts": counts, "image_count": len(records),
                "purpose": "natural photo source only; no segmentation annotations used or benchmark score claimed",
                "citation": "Arbelaez, Maire, Fowlkes, Malik (2011), Contour Detection and Hierarchical Image Segmentation",
                "images": sorted(records, key=lambda row: row["file"])}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


class NaturalSpectrum:
    """A source-photo pool with no mutable sampling state other than caller RNG."""
    def __init__(self, split, root=DATA_ROOT):
        if split not in ("train", "val", "test"):
            raise ValueError(split)
        self.split, self.root = split, Path(root).resolve()
        path = self.root / "manifest.json"
        if not path.exists():
            raise FileNotFoundError("Prepare official BSDS500 first: python -m PreAttentiveVision.natural_stimuli --prepare")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.records = [record for record in manifest["images"] if record["split"] == split]
        expected = {"train": 200, "val": 100, "test": 200}[split]
        if len(self.records) != expected:
            raise ValueError("Dataset source split mismatch")
        self.identity = {"dataset_manifest_sha256": sha256(path), "split": split, "protocol": PROTOCOL}
        self._cache = {}
        freq = np.fft.fftfreq(100)
        fy, fx = np.meshgrid(freq, freq, indexing="ij")
        radius = np.sqrt(fx**2 + fy**2)
        radius[0, 0] = 1.  # DC explicitly zeroed, avoid singular powers
        self.log_frequency = np.log(radius / .1)

    def _image(self, index):
        if index not in self._cache:
            path = self.root / self.records[index]["file"]
            if sha256(path) != self.records[index]["sha256"]:
                raise ValueError("Source photo differs from dataset manifest")
            with Image.open(path) as photo:
                rgb = np.asarray(photo.convert("RGB"), dtype=np.float32) / 255.
            linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055)**2.4)
            self._cache[index] = (linear @ np.array([.2126, .7152, .0722], np.float32)).astype(np.float32)
        return self._cache[index]

    def sample(self, rng, label):
        if label not in (0, 1):
            raise ValueError("Natural spectrum label must be target interval0/1")
        for attempt in range(32):
            index = int(rng.integers(len(self.records)))
            image = self._image(index)
            row = int(rng.integers(image.shape[0] - 99))
            col = int(rng.integers(image.shape[1] - 99))
            crop = image[row:row + 100, col:col + 100].astype(np.float64)
            if crop.std() > 1e-4:
                break
        else:
            raise RuntimeError("32 native crops lacked nonzero contrast")
        rotations = int(rng.integers(4))
        reflected = bool(rng.integers(2))
        crop = np.rot90(crop, rotations)
        if reflected:
            crop = crop[:, ::-1]
        center = float(rng.uniform(-.5, .5))
        delta = float(rng.choice([.15, .30, .60]))
        betas = [center + delta / 2, center + delta / 2]
        betas[label] = center - delta / 2
        spectrum = np.fft.fft2(crop - crop.mean())
        spectrum[0, 0] = 0
        standardized = []
        for beta in betas:
            modified = spectrum * np.exp(-beta * self.log_frequency)
            modified[0, 0] = 0
            raster = np.fft.ifft2(modified).real
            raster -= raster.mean()
            std = raster.std()
            if std < 1e-12:
                raise ValueError("Degenerate filtered crop")
            standardized.append(raster / std)
        z = np.stack(standardized)
        rms = min(.15, .45 / np.abs(z).max())
        pair = (.5 + rms * z).astype(np.float32)
        frames = np.repeat(pair[:, None], 3, axis=1)
        metadata = {"task": "natural_spectrum", "family": "natural_spectrum", "protocol": PROTOCOL["version"],
                    "label": int(label), "difficulty": f"beta_delta_{delta:.2f}",
                    "beta_delta": delta, "betas_by_frame": betas, "target_mean": .5,
                    "actual_common_rms": float(rms), "base_id": self.records[index]["base_id"],
                    "split": self.split, "crop_xywh": [col, row, 100, 100],
                    "quarter_turns": rotations, "horizontal_reflection": reflected,
                    "source_contrast_rejections": attempt, "clipping": False}
        return frames, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        report = prepare_dataset()
        print(json.dumps({key: value for key, value in report.items() if key != "images"}, indent=2))
