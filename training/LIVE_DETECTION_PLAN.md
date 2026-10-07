# Unmaskd Live Detection Plan

Unmaskd is intended to support both uploaded media and live streams.

## 1. Offline uploads

Analyze the complete file with a sliding-window pipeline:

- video: face detection/cropping -> frame/temporal detector -> video-level aggregation
- audio: speech segmentation -> anti-spoof/deepfake detector -> audio-level aggregation
- multimodal fusion: combine calibrated audio/video probabilities

## 2. Live video

The browser captures a MediaStream and sends short rolling WebM windows to the API. The current `live.html` prototype uses 5-second windows. A production version should use overlapping windows (for example 2-4 seconds) and maintain a rolling risk score so the UI updates continuously rather than treating every window as an independent decision.

## 3. Live videocalls

There are two supported product designs:

### A. Unmaskd's own video-call application

Use WebRTC for the call. Keep access to both participants' MediaStreams and send low-resolution analysis windows to the detector. This is the cleanest design because the application owns the media streams.

### B. External calls (Meet/Zoom/Teams/etc.)

A normal web page cannot silently access another application's call media. With explicit user permission, a browser can capture a selected tab/window/screen; however, this may capture a composite call view instead of the individual remote video/audio tracks. A browser extension or native desktop component would be needed for deeper integration, and platform permissions/policies must be respected.

## 4. Training requirements for real-time robustness

The training set should not contain only clean high-quality videos. Add augmentation for:

- H.264/WebM compression
- low bitrate
- 240p/360p/480p resolution
- 10-30 FPS
- frame drops
- motion blur
- resizing/cropping
- webcam noise and lighting changes
- microphone noise, echo and reverberation
- packet-loss-like audio artifacts
- short clips and partial faces

The validation/test split must be by source video/person, not by individual extracted frames, to avoid leakage.

## 5. Target inference behavior

A live detector should output a probability/risk estimate plus a temporal stability measure. Avoid declaring a person fake from one frame. Require evidence over multiple windows and expose `insufficient evidence` when the model is uncertain.

## 6. Datasets

FaceForensics++ is useful for the video baseline and contains original videos and multiple manipulation methods. DeepfakeBench provides a broader standardized framework and supports multiple video detectors and datasets. Dataset licenses and access terms must be followed.

## 7. Research warning

No detector should be marketed as perfectly reliable. Deepfake generators, codecs, languages, cameras and compression conditions change over time. Unmaskd should report calibrated confidence and evidence rather than an absolute authenticity guarantee.
