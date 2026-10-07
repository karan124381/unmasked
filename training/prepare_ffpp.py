from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MANIPULATIONS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--frames-per-video", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-videos", type=int, default=0, help="0 = all videos")
    return p.parse_args()


def videos(root: Path, relative: str):
    folder = root / relative
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS) if folder.exists() else []


def group_key(path: Path, fake: bool) -> str:
    # FF++ fake filenames are generally target_source.mp4. Keep all manipulations
    # associated with the same target/source group in one split.
    stem = path.stem
    return stem.split("_")[0] if fake and "_" in stem else stem


def make_splits(items, seed):
    groups = {}
    for item in items:
        groups.setdefault(item[0], []).append(item)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    n_test = max(1, round(n * 0.15))
    n_val = max(1, round(n * 0.15))
    test_keys = set(keys[:n_test])
    val_keys = set(keys[n_test:n_test + n_val])
    out = {"train": [], "val": [], "test": []}
    for key, rows in groups.items():
        split = "test" if key in test_keys else "val" if key in val_keys else "train"
        out[split].extend(rows)
    return out


def extract_frames(video: Path, destination: Path, count: int):
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return 0
    indices = [round(i * (total - 1) / max(1, count - 1)) for i in range(count)]
    written = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        destination.mkdir(parents=True, exist_ok=True)
        out = destination / f"{video.stem}_{idx:06d}.jpg"
        if cv2.imwrite(str(out), frame):
            written += 1
    cap.release()
    return written


def main():
    args = parse_args()
    root = args.dataset_root
    out = args.output_dir

    real = videos(root, "original_sequences/youtube/c23/videos")
    fake = []
    for method in MANIPULATIONS:
        fake.extend(videos(root, f"manipulated_sequences/{method}/c23/videos"))

    items = [(group_key(p, False), "real", p) for p in real]
    items += [(group_key(p, True), "fake", p) for p in fake]
    if args.max_videos:
        random.Random(args.seed).shuffle(items)
        items = items[: args.max_videos]

    splits = make_splits(items, args.seed)
    for split, rows in splits.items():
        counts = {"real": 0, "fake": 0}
        for _, label, video in rows:
            written = extract_frames(video, out / split / label, args.frames_per_video)
            counts[label] += written
        print(f"{split}: {counts}")

    print(f"Prepared frames under {out}")


if __name__ == "__main__":
    main()
