from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tritonclient.http as httpclient
import torchvision.transforms as standard_transforms
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import io

# FastAPI app
app = FastAPI()

# Connect to Triton server
triton_client = httpclient.InferenceServerClient(url="p2pnet-tritonserver:8000")  # use container hostname if in Docker
model_name = "p2pnet"

# Preprocessing
transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
])

def predict_point_count(input_img: Image.Image) -> int:
    img_raw = input_img.convert('RGB')

    # Resize for model
    target_size = (336, 336)
    img_raw = img_raw.resize(target_size, Image.ANTIALIAS)

    # Convert to NCHW FP32 numpy
    img_tensor = transform(img_raw).unsqueeze(0).numpy()

    # Triton input
    inputs = [httpclient.InferInput("input", img_tensor.shape, "FP32")]
    inputs[0].set_data_from_numpy(img_tensor, binary_data=True)

    # Triton outputs
    outputs = [
        httpclient.InferRequestedOutput("output"),
        httpclient.InferRequestedOutput("263")
    ]

    # Run inference
    results = triton_client.infer(
        model_name=model_name,
        inputs=inputs,
        outputs=outputs
    )

    # Extract results
    pred_logits = results.as_numpy("output")
    pred_points = results.as_numpy("263")

    # Postprocess
    outputs_scores = F.softmax(torch.from_numpy(pred_logits), -1)[:, :, 1][0].numpy()
    outputs_points = pred_points[0]

    threshold = 0.4
    points = outputs_points[outputs_scores > threshold].tolist()
    return len(points)

@app.get("/ping")
async def ping():
    try:
        live = triton_client.is_server_live()
        ready = triton_client.is_server_ready()
        mode_ready = triton_client.is_model_ready(model_name=model_name)
        return {"server_live": live, "server_ready": ready, "mode_ready": mode_ready}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        img_bytes = await file.read()
        input_img = Image.open(io.BytesIO(img_bytes))

        # Run inference
        count = predict_point_count(input_img)

        return {"count": count}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)