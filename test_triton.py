import tritonclient.http as httpclient
import torchvision.transforms as standard_transforms
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
 
# Connect to Triton server
triton_client = httpclient.InferenceServerClient(url="localhost:8014")  # HTTP port
 
model_name = "p2pnet"  # same as Triton repo folder name
 
# Preprocessing
transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
])
 
def predict_and_visualize_triton(input_img):
    print("received image")
    img_raw = input_img.convert('RGB')
 
    # Resize to match ONNX input
    target_size = (336, 336)  # width, height
    # img_raw = img_raw.resize(target_size, Image.ANTIALIAS)
    img_raw = img_raw.resize(target_size, Image.Resampling.LANCZOS)
 
    # Convert to NCHW FP32 numpy
    img_tensor = transform(img_raw).unsqueeze(0).numpy()
 
    print("sending to tritonserver")
    # Create input
    inputs = []
    inputs.append(httpclient.InferInput("input", img_tensor.shape, "FP32"))
    inputs[0].set_data_from_numpy(img_tensor, binary_data=True)
 
    # Define outputs
    outputs = []
    outputs.append(httpclient.InferRequestedOutput("output"))
    outputs.append(httpclient.InferRequestedOutput("263"))
 
    # Run inference
    results = triton_client.infer(model_name=model_name,
                                  inputs=inputs,
                                  outputs=outputs)
 
    # Extract results
    preds = results.as_numpy("output")
    pred_logits = results.as_numpy("output")
    pred_points = results.as_numpy("263")

 
    # Postprocess
    outputs_scores = F.softmax(torch.from_numpy(pred_logits), -1)[:, :, 1][0].numpy()
    print(outputs_scores)
    outputs_points = pred_points[0]
 
    threshold = 0.4
    points = outputs_points[outputs_scores > threshold].tolist()
    print("predict image")
 
    size = max(2, int(0.005 * max(img_raw.size)))
    img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
    for p in points:
        img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
 
    num_points = len(points)
    height, width, _ = img_to_draw.shape
    cv2.putText(img_to_draw, f'Count: {num_points}',
                (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)
 
    img_to_draw = cv2.cvtColor(img_to_draw, cv2.COLOR_BGR2RGB)
    print("visualize")
    return Image.fromarray(img_to_draw)
 
 
if __name__ == "__main__":
    image_path = "/home/cuongph14/Downloads/DATG3540.JPG"
    input_img = Image.open(image_path)
    import time
    tik0 = time.time()
    output_img = predict_and_visualize_triton(input_img)
    latency = time.time() - tik0
    print(f"latency: {latency}")
    output_img.save("visualized_output_triton_http.png")