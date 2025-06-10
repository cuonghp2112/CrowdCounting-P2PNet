import gradio as gr
import torch
import torchvision.transforms as standard_transforms
import numpy as np
from PIL import Image
import cv2
import argparse
import os
from models import build_model
import warnings
warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='vgg16_bn', type=str)
    parser.add_argument('--row', default=2, type=int)
    parser.add_argument('--line', default=2, type=int)
    parser.add_argument('--weight_path', default='weights/SHTechA.pth', type=str)  # Update with your weight path
    parser.add_argument('--gpu_id', default=0, type=int)
    args = parser.parse_args("")  # empty string to avoid parsing sys.argv
    return args

args = get_args()
device = torch.device('cpu')
model = build_model(args)
model.to(device)
if args.weight_path:
    checkpoint = torch.load(args.weight_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
model.eval()

transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_and_visualize(input_img):
    print("received image")
    img_raw = input_img.convert('RGB')
    width, height = img_raw.size
    scale_w = 800 / width
    scale_h = 600 / height
    scale = min(scale_w, scale_h)
    new_width = int(width * scale)
    new_height = int(height * scale)
    new_width = (new_width // 128) * 128
    new_height = (new_height // 128) * 128

    img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
    img = transform(img_raw).unsqueeze(0).to(device)
    print("resized image")

    with torch.no_grad():
        outputs = model(img)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
        outputs_points = outputs['pred_points'][0]
        threshold = 0.4
        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
    
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

demo = gr.Interface(
    fn=predict_and_visualize,
    inputs=gr.Image(type="pil", label="Input Image", image_mode='RGB', sources=['upload'], format=None),
    outputs=gr.Image(type="pil", label="Output Visualization"),
    title="P2PNet Crowd Counting Demo",
    description="Upload an image (including .jfif) to get the predicted crowd count and visualization.",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860, debug=True)
