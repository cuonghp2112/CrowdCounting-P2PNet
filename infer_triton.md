To serve your **ONNX model with Triton Inference Server**, you'll need to:

---

### ✅ 1. Organize the Triton model repository structure:

Triton expects a specific directory layout.

```
triton_models/
└── p2pnet_onnx/
    ├── 1/
    │   └── model.onnx
    └── config.pbtxt
```

Place your ONNX model (`model_fixed_bs1.onnx`) as `triton_models/p2pnet_onnx/1/model.onnx`.

Rename:

```bash
cp model_fixed_bs1.onnx triton_models/p2pnet_onnx/1/model.onnx
```

---

### ✅ 2. Write `config.pbtxt` for Triton:

```protobuf
name: "p2pnet_onnx"
platform: "onnxruntime_onnx"
max_batch_size: 0

input [
  {
    name: "input"  # <-- match this to your ONNX model input name
    data_type: TYPE_FP32
    dims: [3, 336, 336]
  }
]

output [
  {
    name: "pred_logits"
    data_type: TYPE_FP32
    dims: [100, 2]
  },
  {
    name: "pred_points"
    data_type: TYPE_FP32
    dims: [100, 2]
  }
]
```

> 🧠 To confirm the ONNX input/output names and shapes, use:

```python
ort_session.get_inputs()[0].name
ort_session.get_outputs()[0].name
```

---

### ✅ 3. Run Triton Server with the model repo:

```bash
docker run --rm --gpus all -p8000:8000 -p8001:8001 -p8002:8002 \
  -v $(pwd)/triton_models:/models nvcr.io/nvidia/tritonserver:24.03-py3 \
  tritonserver --model-repository=/models
```

---

### ✅ 4. Example Python Triton client:

Install client libs:

```bash
pip install tritonclient[http]
```

Then:

```python
import numpy as np
import tritonclient.http as httpclient
from PIL import Image
import torchvision.transforms as standard_transforms
import torch.nn.functional as F
import cv2

transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

client = httpclient.InferenceServerClient(url="localhost:8000")

img_raw = Image.open("test_img/image (3).jfif").convert("RGB")
img_raw = img_raw.resize((336, 336))
img_tensor = transform(img_raw).unsqueeze(0).numpy()

inputs = [
    httpclient.InferInput("input", img_tensor.shape, "FP32")
]
inputs[0].set_data_from_numpy(img_tensor)

outputs = [
    httpclient.InferRequestedOutput("pred_logits"),
    httpclient.InferRequestedOutput("pred_points"),
]

response = client.infer(model_name="p2pnet_onnx", inputs=inputs, outputs=outputs)

logits = response.as_numpy("pred_logits")
points = response.as_numpy("pred_points")

scores = F.softmax(torch.from_numpy(logits), -1)[:, :, 1][0].numpy()
points = points[0][scores > 0.4]

# Draw points
img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
for p in points:
    img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)

cv2.putText(img_to_draw, f"Count: {len(points)}", (10, 320),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

img_to_draw = cv2.cvtColor(img_to_draw, cv2.COLOR_BGR2RGB)
Image.fromarray(img_to_draw).save("triton_output.png")
```

---

### ✅ Result

You’ll have a working Triton-based inference setup with output image saved to `triton_output.png`.

---

Let me know if you want:

* Batch inference support
* gRPC version
* Dockerized deployment script


---

Here's a complete **FastAPI app** that does the following:

* Accepts image uploads (e.g., `.jpg`, `.jfif`, etc.)
* Preprocesses the image (resize to `336x336`, normalize, etc.)
* Sends the tensor to **Triton Inference Server** using **gRPC**
* Processes the outputs (logits + points) to draw detection results
* Returns the resulting image

---

## ✅ Step 1: Install Required Packages

```bash
pip install fastapi uvicorn python-multipart tritonclient[grpc] torchvision pillow opencv-python
```

---

## ✅ Step 2: `main.py` (FastAPI app)

```python
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from io import BytesIO
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import tritonclient.grpc as grpcclient

app = FastAPI()

# Triton gRPC client
triton_client = grpcclient.InferenceServerClient(url="localhost:8001")

# Input and output names from ONNX model
TRITON_MODEL_NAME = "p2pnet_onnx"
INPUT_NAME = "input"
OUTPUT_NAMES = ["pred_logits", "pred_points"]

# Preprocessing: resize, to tensor, normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Load image and convert
    image = Image.open(BytesIO(await file.read())).convert("RGB")
    image = image.resize((336, 336))
    tensor = transform(image).unsqueeze(0).numpy()

    # Prepare gRPC inference input
    inputs = [
        grpcclient.InferInput(INPUT_NAME, tensor.shape, "FP32")
    ]
    inputs[0].set_data_from_numpy(tensor)

    # Request both outputs
    outputs = [
        grpcclient.InferRequestedOutput(name) for name in OUTPUT_NAMES
    ]

    # Inference
    results = triton_client.infer(model_name=TRITON_MODEL_NAME, inputs=inputs, outputs=outputs)

    # Post-process outputs
    logits = results.as_numpy("pred_logits")
    points = results.as_numpy("pred_points")

    scores = F.softmax(torch.from_numpy(logits), -1)[:, :, 1][0].numpy()
    filtered_points = points[0][scores > 0.4]

    # Draw results
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for p in filtered_points:
        img = cv2.circle(img, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)

    cv2.putText(img, f"Count: {len(filtered_points)}", (10, 330),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Convert to RGB PIL image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result_img = Image.fromarray(img)

    # Convert to byte stream for response
    buffer = BytesIO()
    result_img.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")
```

---

## ✅ Step 3: Run the API Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## ✅ Example Request

Using `curl`:

```bash
curl -X POST http://localhost:8000/predict/ \
  -F "file=@test_img/image.jfif" \
  --output output.png
```

---

### ✅ Notes:

* Ensure `tritonserver` is running at `localhost:8001` with model `p2pnet_onnx`.
* Make sure your model input name is actually `"input"` — adjust if needed.

Would you like a version that returns **JSON count and coordinates** instead of the image?

