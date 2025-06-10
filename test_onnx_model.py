import onnxruntime as ort
import torchvision.transforms as standard_transforms
import numpy as np
from PIL import Image
import cv2
import torch

def predict_and_visualize_onnx(input_img):
    print("received image")
    img_raw = input_img.convert('RGB')
    
    # Force resize to fixed ONNX input shape: (3, 896, 672)
    target_size = (336, 336)  # (width, height)
    img_raw = img_raw.resize(target_size, Image.ANTIALIAS)

    # Transform and prepare input tensor
    img_tensor = transform(img_raw).unsqueeze(0).numpy()  # numpy array for ONNX
    print("resized image")

    # Run inference with ONNX Runtime
    ort_inputs = {ort_session.get_inputs()[0].name: img_tensor}
    ort_outs = ort_session.run(None, ort_inputs)

    # Extract outputs based on your ONNX model's output structure
    # Assuming model outputs same dict-like outputs:
    # [pred_logits, pred_points]
    pred_logits = ort_outs[0]  # shape: (batch, num_queries, num_classes)
    pred_points = ort_outs[1]  # shape: (batch, num_queries, 2)

    import torch.nn.functional as F
    outputs_scores = F.softmax(torch.from_numpy(pred_logits), -1)[:, :, 1][0].numpy()
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
    cv2.putText(
        img_to_draw,
        f'Count: {num_points}',
        (10, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    img_to_draw = cv2.cvtColor(img_to_draw, cv2.COLOR_BGR2RGB)
    print("visualize")
    return Image.fromarray(img_to_draw)

# Load ONNX model once
ort_session = ort.InferenceSession("model_fixed_bs1.onnx")

transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

from PIL import Image

# Path to your .jfif image file
image_path = "test_img/image (3).jfif"

# Load image
input_img = Image.open(image_path)

# Run inference + visualize
output_img = predict_and_visualize_onnx(input_img)

output_img.save("visualized_output.png")