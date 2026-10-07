import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import librosa
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import pipeline

AUDIO_MODEL = os.getenv("AUDIO_MODEL", "Hemgg/Deepfake-audio-detection")
VIDEO_MODEL = os.getenv("VIDEO_MODEL", "Hemg/Deepfake-Detection")
MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))

app = FastAPI(title="Unmaskd Detection API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_audio_pipe = None
_video_pipe = None


def get_audio_pipe():
    global _audio_pipe
    if _audio_pipe is None:
        _audio_pipe = pipeline(
            "audio-classification",
            model=AUDIO_MODEL,
            device=0 if torch.cuda.is_available() else -1,
        )
    return _audio_pipe


def get_video_pipe():
    global _video_pipe
    if _video_pipe is None:
        _video_pipe = pipeline(
            "image-classification",
            model=VIDEO_MODEL,
            device=0 if torch.cuda.is_available() else -1,
        )
    return _video_pipe


def fake_probability(predictions):
    """Convert model labels to a fake probability when labels are descriptive.
    If a model exposes only anonymous LABEL_* names, return None rather than
    pretending we know which class means fake.
    """
    fake_words = ("fake", "spoof", "synthetic", "deepfake", "generated", "manipulated")
    real_words = ("real", "bonafide", "bona fide", "genuine", "authentic")
    fake = 0.0
    real = 0.0
    known = False
    for item in predictions:
        label = str(item.get("label", "")).lower()
        score = float(item.get("score", 0.0))
        if any(w in label for w in fake_words):
            fake += score
            known = True
        elif any(w in label for w in real_words):
            real += score
            known = True
    if not known:
        return None
    total = fake + real
    return fake / total if total else None


def run_ffmpeg_extract_audio(src: str, dst: str):
    cmd = [
        "ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", dst
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError("ffmpeg could not extract an audio track")


def analyze_audio(path: str):
    pipe = get_audio_pipe()
    y, sr = librosa.load(path, sr=16000, mono=True)
    if y.size == 0:
        raise ValueError("Audio track is empty")
    chunk_len = 5 * sr
    chunks = []
    for start in range(0, len(y), chunk_len):
        chunk = y[start:start + chunk_len]
        if len(chunk) >= int(1.0 * sr):
            chunks.append((start / sr, min((start + len(chunk)) / sr, len(y) / sr), chunk))
    if not chunks:
        chunks = [(0.0, len(y) / sr, y)]

    scores = []
    details = []
    for start, end, chunk in chunks[:30]:
        preds = pipe(chunk, sampling_rate=16000, top_k=None)
        fp = fake_probability(preds)
        details.append({"start": round(start, 2), "end": round(end, 2), "predictions": preds})
        if fp is not None:
            scores.append(fp)
    return {
        "available": True,
        "fake_probability": round(float(np.mean(scores)), 4) if scores else None,
        "duration": round(len(y) / sr, 2),
        "chunks": details,
        "model": AUDIO_MODEL,
    }


def analyze_video(path: str):
    pipe = get_video_pipe()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Video could not be opened")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if frame_count else 0.0
    target = min(16, max(4, int(duration) + 1))
    indices = np.linspace(0, max(frame_count - 1, 0), target, dtype=int)
    scores = []
    details = []
    for idx in sorted(set(indices.tolist())):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preds = pipe(Image.fromarray(rgb), top_k=None)
        fp = fake_probability(preds)
        details.append({"timestamp": round(idx / fps, 2), "predictions": preds})
        if fp is not None:
            scores.append(fp)
    cap.release()
    return {
        "available": True,
        "fake_probability": round(float(np.mean(scores)), 4) if scores else None,
        "duration": round(duration, 2),
        "frames_analyzed": len(details),
        "frames": details,
        "model": VIDEO_MODEL,
    }


def classify(overall_fake: Optional[float]):
    if overall_fake is None:
        return "insufficient-model-labels"
    if overall_fake >= 0.75:
        return "high-risk"
    if overall_fake >= 0.45:
        return "suspicious"
    return "low-risk"


@app.get("/")
def root():
    return {"service": "Unmaskd Detection API", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy", "cuda": torch.cuda.is_available()}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()
    allowed = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".mp4", ".mov", ".webm", ".avi", ".mkv"}
    if suffix not in allowed:
        raise HTTPException(400, "Unsupported media type")

    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, f"input{suffix}")
        with open(src, "wb") as out:
            total = 0
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_MB * 1024 * 1024:
                    raise HTTPException(413, f"File exceeds {MAX_MB} MB limit")
                out.write(chunk)

        is_video = suffix in {".mp4", ".mov", ".webm", ".avi", ".mkv"}
        video_result = None
        audio_result = None
        errors = []

        try:
            if is_video:
                video_result = analyze_video(src)
                audio_path = os.path.join(td, "audio.wav")
                try:
                    run_ffmpeg_extract_audio(src, audio_path)
                    audio_result = analyze_audio(audio_path)
                except Exception as exc:
                    errors.append(f"Audio analysis unavailable: {exc}")
            else:
                audio_result = analyze_audio(src)
        except Exception as exc:
            errors.append(f"Analysis failed: {exc}")

        scores = []
        weights = []
        if audio_result and audio_result.get("fake_probability") is not None:
            scores.append(audio_result["fake_probability"]); weights.append(0.4 if video_result else 1.0)
        if video_result and video_result.get("fake_probability") is not None:
            scores.append(video_result["fake_probability"]); weights.append(0.6 if audio_result else 1.0)
        overall = float(np.average(scores, weights=weights)) if scores else None

        return {
            "filename": filename,
            "verdict": classify(overall),
            "fake_probability": round(overall, 4) if overall is not None else None,
            "audio": audio_result,
            "video": video_result,
            "errors": errors,
            "note": "Research prototype. Model performance can vary across codecs, languages, generators and real-world media.",
        }
