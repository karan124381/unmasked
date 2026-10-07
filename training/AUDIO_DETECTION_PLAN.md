# Unmaskd Audio Authenticity Detection

Unmaskd should treat audio as a first-class detection signal alongside video.

## Goal

For each rolling live-call audio window, produce:

- `authenticity_score`: 0-100, where higher means the audio is more consistent with authentic speech.
- `deepfake_probability`: 0-1, where higher means stronger evidence of synthetic/manipulated speech.
- `verdict`: `authentic`, `suspicious`, `high-risk`, or `insufficient-signal`.
- model evidence and window timestamps.

These are model estimates, not proof of authenticity.

## Detection pipeline

Microphone / call audio
-> 2-5 second rolling windows
-> resample to model rate
-> loudness / silence handling
-> anti-spoof / synthetic-speech detector
-> calibration
-> temporal smoothing
-> authenticity score
-> multimodal fusion with video

## Training

Use genuine speech and multiple synthetic/manipulated speech families. Include realistic live-call conditions such as:

- WebRTC/VoIP codecs and compression
- low bitrate
- background noise
- microphone noise
- echo and reverberation
- packet-loss-like artifacts
- resampling
- short utterances and silence
- different speakers, accents and languages where licensing permits

Use speaker-disjoint and source-disjoint train/validation/test splits to reduce leakage.

ASVspoof 2021 DeepFake is an important benchmark for spoofed/deepfake speech, but published research shows that generalization to unseen generators and real-world conditions remains difficult. Therefore Unmaskd should validate on out-of-domain audio instead of relying on one benchmark.

## Authenticity score

The UI should not display raw classifier confidence as if it were a guaranteed probability of truth. Calibrate the detector on a held-out validation set, then map the calibrated synthetic probability to:

`authenticity_score = round(100 * (1 - calibrated_deepfake_probability))`

Example presentation:

- 90-100: High authenticity signal
- 70-89: Mostly consistent with authentic speech
- 40-69: Uncertain / suspicious
- 0-39: Strong manipulation signal

Thresholds must be tuned using validation data and should not be presented as universal guarantees.

## Live fusion

For a video call, maintain separate rolling scores:

`audio_authenticity`
`video_authenticity`

Then combine calibrated probabilities using a learned or validated fusion layer. Do not simply average arbitrary model scores in the final production system.

The system should also expose `signal_quality` so poor microphone/network conditions do not get mistaken for deepfake evidence.

## Research references

- ASVspoof 2021 DeepFake task: https://arxiv.org/abs/2210.02437
- Audio Deepfake Detection survey: https://arxiv.org/abs/2308.14970
- Audio deepfake generalization study: https://arxiv.org/abs/2203.16263
