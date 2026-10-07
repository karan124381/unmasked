# Unmaskd dataset setup

This guide sets up the first real training run without committing raw datasets to GitHub.

## 1. Video: Kaggle DeepFake Detection

Recommended first video dataset: Kaggle's DeepFake Detection competition dataset.

1. Open the Kaggle competition page and accept its rules if Kaggle requires it.
2. In Google Colab, install the Kaggle CLI:

```bash
pip install -q kaggle
```

3. Upload your Kaggle API token (`kaggle.json`) when prompted, then run:

```bash
mkdir -p ~/.kaggle
cp /content/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

4. Download the competition data:

```bash
mkdir -p /content/unmaskd/data/video_raw
cd /content/unmaskd/data/video_raw
kaggle competitions download deepfake-detection --unzip
```

The Kaggle CLI officially supports competition downloads with `kaggle competitions download <competition>` and `--unzip`.

## 2. Prepare video frames

Clone the Unmaskd repository in Colab:

```bash
cd /content
rm -rf unmasked
 git clone https://github.com/karan124381/unmasked.git
cd /content/unmasked
```

If the dataset was downloaded into `/content/unmaskd/data/video_raw`, point the preparation script at that location. Example:

```bash
python training/prepare_ffpp.py \
  --dataset-root /content/unmaskd/data/video_raw \
  --output-dir /content/unmaskd/data/video_frames \
  --frames-per-video 8
```

> Note: `prepare_ffpp.py` was originally written around FaceForensics++ directory conventions. The Kaggle competition layout may differ. If its labels/folders do not match the expected structure, adapt the preparation script to the actual Kaggle folder structure instead of guessing labels.

## 3. Audio: ASVspoof 2021 DF

Use the ASVspoof 2021 Speech Deepfake (DF) database as the first dedicated audio benchmark. ASVspoof states that its DF database contains bonafide and spoofed speech and provides the data through Zenodo under an Open Data Commons Attribution licence.

Registration/download instructions are on the official ASVspoof site. After downloading and extracting the DF data, keep it outside GitHub and create a local manifest containing:

- file path
- label (bonafide/spoof)
- speaker/source ID when available
- dataset split

For Unmaskd, split by speaker/source rather than randomly splitting individual audio clips.

## 4. Train the first video baseline

After the prepared ImageFolder data exists:

```bash
cd /content/unmasked
pip install -r training/requirements.txt
python training/train_video.py \
  --data-dir /content/unmaskd/data/video_frames \
  --epochs 5 \
  --batch-size 32
```

Expected outputs include a best checkpoint and evaluation metrics. Do not report accuracy/F1/AUC until the run actually completes.

## 5. Next model stage for live calls

The EfficientNet baseline is a starting point. The production research path is:

- temporal video detector rather than isolated-frame classification
- audio anti-spoof detector
- calibrated probabilities
- signal-quality estimation
- temporal smoothing over rolling 2–5 second windows
- learned/validated audio-video fusion

For live calls, train with deployment-like augmentation: WebRTC/VoIP compression, lower bitrate, frame drops, blur, resizing, lighting changes, microphone noise, echo/reverb, resampling and short speech windows.

## Dataset rules

- Do not commit raw video/audio datasets to GitHub.
- Keep credentials such as `kaggle.json` out of the repository.
- Follow each dataset's license, competition rules and access requirements.
- Store code, manifests, configs, checkpoints and metrics in the repository; store large raw datasets in the training environment or approved object storage.
