# Unmaskd Model Training

This folder contains the reproducible training pipeline for the Unmaskd video detector.

## Dataset

The recommended primary dataset is **FaceForensics++ (FF++)**. The official GitHub repository is:

https://github.com/ondyari/FaceForensics

The full FF++ media is **not stored in that GitHub repository**. The official project requires an access request and then provides a download script. Do not scrape or redistribute the dataset.

Use the C23 videos for the first Unmaskd experiment:

```text
FF_Dataset/
├── original_sequences/youtube/c23/videos/
├── manipulated_sequences/Deepfakes/c23/videos/
├── manipulated_sequences/Face2Face/c23/videos/
├── manipulated_sequences/FaceSwap/c23/videos/
└── manipulated_sequences/NeuralTextures/c23/videos/
```

## 1. Prepare frames

```bash
python training/prepare_ffpp.py --dataset-root /path/to/FF_Dataset --output-dir training/data --frames-per-video 8
```

The script creates video-level train/validation/test splits before extracting frames, which helps prevent frames from the same source video leaking across splits.

## 2. Train

```bash
pip install -r training/requirements.txt
python training/train_video.py --data-dir training/data --epochs 5 --batch-size 32
```

The best checkpoint is written to `training/checkpoints/`.

## 3. Evaluate

The training script reports accuracy, precision, recall, F1 and ROC-AUC and writes a classification report plus confusion matrix.

## Important

This is a research baseline, not a production-grade detector. A frame classifier can learn compression and dataset-specific artifacts. For a stronger Unmaskd model, the next stage should add temporal modeling and cross-dataset testing on an unseen dataset such as Celeb-DF or DFDC.
