# Unmaskd Detection API

FastAPI research prototype for multimodal deepfake analysis.

## Run locally

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The API exposes `POST /analyze` and `GET /health`.

## Docker

```bash
docker build -t unmaskd-api .
docker run -p 8000:8000 unmaskd-api
```

## Important

The backend loads the configured Hugging Face models at first inference. Set `AUDIO_MODEL` and `VIDEO_MODEL` to model repositories whose task/labels match the intended detector. The code intentionally refuses to convert anonymous model labels into a fake/real probability, because doing so without knowing the label semantics would be misleading.

This is a research prototype, not a forensic-grade certification system. Validate model accuracy on representative datasets before making operational claims.
