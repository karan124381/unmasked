import asyncio
import json
import os
import tempfile
from pathlib import Path

from fastapi import WebSocket, WebSocketDisconnect

from main import analyze

# WebSocket protocol for the browser live prototype.
# Client sends JSON metadata followed by binary WebM chunks.
# Each binary chunk is treated as a short rolling observation window.

LIVE_MAX_MB = int(os.getenv("LIVE_MAX_MB", "12"))


async def live_socket(websocket: WebSocket):
    await websocket.accept()
    total_bytes = 0
    try:
        while True:
            message = await websocket.receive()
            if message.get("text") is not None:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                    continue
                if payload.get("type") == "stop":
                    await websocket.send_json({"type": "stopped"})
                    break
                await websocket.send_json({"type": "ready", "window_seconds": payload.get("window_seconds", 5)})
                continue

            chunk = message.get("bytes")
            if not chunk:
                continue
            total_bytes += len(chunk)
            if total_bytes > LIVE_MAX_MB * 1024 * 1024:
                await websocket.send_json({"type": "error", "message": "Live session size limit exceeded"})
                break

            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
                tmp.write(chunk)
                path = tmp.name

            try:
                # Reuse the same analysis implementation used by uploads.
                # This endpoint intentionally analyzes rolling chunks rather than
                # claiming frame-by-frame inference.
                from fastapi import UploadFile
                from starlette.datastructures import Headers

                with open(path, "rb") as fh:
                    upload = UploadFile(file=fh, filename="live.webm", headers=Headers({"content-type": "video/webm"}))
                    result = await analyze(upload)
                await websocket.send_json({"type": "result", "result": result})
            except Exception as exc:
                await websocket.send_json({"type": "error", "message": str(exc)})
            finally:
                Path(path).unlink(missing_ok=True)
    except WebSocketDisconnect:
        return
