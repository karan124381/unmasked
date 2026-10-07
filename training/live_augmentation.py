"""Create live-call style video variants from a folder of MP4 files.

This does not create labels or synthetic fakes. It applies deployment-like
perturbations to existing REAL/FAKE clips so the detector learns robustness
against compression, resize, frame-rate and audio/container changes.
"""
from __future__ import annotations

import argparse
import random
import subprocess
from pathlib import Path


def run(src: Path, dst: Path, mode: str, seed: int):
    random.seed(seed)
    dst.parent.mkdir(parents=True, exist_ok=True)
    scale = random.choice(["426:-2", "640:-2", "854:-2"])
    fps = random.choice([10, 12, 15, 20, 24])
    bitrate = random.choice(["350k", "550k", "800k", "1200k"])
    vf = f"scale={scale},fps={fps}"
    af = "aresample=16000"
    if mode == "noisy":
        vf += ",eq=brightness=0.02:contrast=0.92"
        af += ",highpass=f=80,lowpass=f=7000"
    elif mode == "blur":
        vf += ",gblur=sigma=0.7"
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", vf, "-af", af,
        "-c:v", "libx264", "-preset", "veryfast", "-b:v", bitrate,
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "64k", str(dst),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="Folder containing REAL/FAKE video folders")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--copies", type=int, default=2)
    args = p.parse_args()

    modes = ["normal", "noisy", "blur"]
    videos = list(args.input.rglob("*.mp4"))
    if not videos:
        raise SystemExit("No .mp4 files found")
    for i, src in enumerate(videos):
        label = src.parent.name
        for j in range(args.copies):
            mode = modes[(i + j) % len(modes)]
            dst = args.output / label / f"{src.stem}_live_{j}_{mode}.mp4"
            try:
                run(src, dst, mode, seed=i * 100 + j)
            except subprocess.CalledProcessError:
                print(f"failed: {src}")
    print(f"Created live-style variants under {args.output}")


if __name__ == "__main__":
    main()
