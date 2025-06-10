Yes, **FastAPI** supports both **synchronous HTTP endpoints** and **WebSocket** for real-time streaming.

Here’s how you can build a **FastAPI app** that:

### ✅ 1. Accepts a video file (HTTP `POST`)

### ✅ 2. Streams video frames in real-time to the client using WebSocket

### ✅ 3. Performs inference on each frame (e.g., with Triton)

---

## ✅ FastAPI App: Video Upload + WebSocket Stream

### 🔧 Install dependencies:

```bash
pip install fastapi uvicorn python-multipart opencv-python tritonclient[grpc] websockets
```

---

## 🔌 `main.py`

```python
import cv2
import numpy as np
import uuid
import os
import shutil
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput
import torch.nn.functional as F
import torch

app = FastAPI()

UPLOAD_DIR = "./uploaded_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Triton Config
TRITON_CLIENT = InferenceServerClient(url="localhost:8001")
TRITON_MODEL_NAME = "p2pnet_onnx"
INPUT_NAME = "input"
OUTPUT_NAMES = ["pred_logits", "pred_points"]

# Preprocessing
import torchvision.transforms as transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((336, 336)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def preprocess(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
    tensor = transform(img).unsqueeze(0).numpy()
    return tensor

def postprocess(logits, points, frame):
    scores = F.softmax(torch.from_numpy(logits), -1)[:, :, 1][0].numpy()
    points = points[0][scores > 0.4]

    for p in points:
        frame = cv2.circle(frame, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)
    cv2.putText(frame, f"Count: {len(points)}", (10, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

def infer_frame(tensor):
    inputs = [InferInput(INPUT_NAME, tensor.shape, "FP32")]
    inputs[0].set_data_from_numpy(tensor)
    outputs = [InferRequestedOutput(n) for n in OUTPUT_NAMES]
    results = TRITON_CLIENT.infer(model_name=TRITON_MODEL_NAME, inputs=inputs, outputs=outputs)
    return results.as_numpy("pred_logits"), results.as_numpy("pred_points")


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    video_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{video_id}.mp4")

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"video_id": video_id}


@app.websocket("/ws/{video_id}")
async def stream_video(websocket: WebSocket, video_id: str):
    await websocket.accept()
    video_path = os.path.join(UPLOAD_DIR, f"{video_id}.mp4")

    if not os.path.exists(video_path):
        await websocket.send_text("Invalid video_id")
        await websocket.close()
        return

    cap = cv2.VideoCapture(video_path)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            tensor = preprocess(frame)
            logits, points = infer_frame(tensor)
            vis_frame = postprocess(logits, points, frame)

            # Encode to JPEG
            _, jpeg = cv2.imencode('.jpg', vis_frame)
            await websocket.send_bytes(jpeg.tobytes())
    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        cap.release()
```

---

## 🧪 WebSocket Frontend (Test)

```html
<!DOCTYPE html>
<html>
<head>
  <title>Video Stream</title>
</head>
<body>
  <h1>Real-time Inference Stream</h1>
  <img id="video-frame" width="672" height="336"/>

  <script>
    const video_id = prompt("Enter video_id:");
    const ws = new WebSocket(`ws://localhost:8000/ws/${video_id}`);
    const img = document.getElementById("video-frame");

    ws.binaryType = "arraybuffer";

    ws.onmessage = function (event) {
      const blob = new Blob([event.data], {type: 'image/jpeg'});
      img.src = URL.createObjectURL(blob);
    };
  </script>
</body>
</html>
```

---

## ✅ Usage

1. Upload video:

```bash
curl -F "file=@video.mp4" http://localhost:8000/upload
```

2. Use returned `video_id` to test WebSocket via the HTML client.

---

Would you like support for **real-time camera streaming** or **saving the processed video** as well?
