import gradio as gr
import torch
import torchvision.transforms as standard_transforms
import numpy as np
from PIL import Image
import cv2
import argparse
import os
import tempfile
from models import build_model
import warnings
import subprocess
warnings.filterwarnings('ignore')


# Load model
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='vgg16_bn', type=str)
    parser.add_argument('--row', default=2, type=int)
    parser.add_argument('--line', default=2, type=int)
    parser.add_argument('--weight_path', default='weights/SHTechA.pth', type=str)
    parser.add_argument('--gpu_id', default=0, type=int)
    args = parser.parse_args("")
    return args

args = get_args()
device = torch.device('cpu')
model = build_model(args).to(device)
checkpoint = torch.load(args.weight_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()

transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Frame-level prediction
def predict_frame(frame, skip=False, pred_points=None):
    img_raw = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')
    width, height = img_raw.size
    scale = min(800 / width, 600 / height)
    new_width = (int(width * scale) // 128) * 128
    new_height = (int(height * scale) // 128) * 128
    img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)

    img = transform(img_raw).unsqueeze(0).to(device)
    if skip:
        img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
        size = max(2, int(0.005 * max(img_raw.size)))
        if pred_points is not None:
            for p in pred_points:
                img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
        return img_to_draw, pred_points
    else:
        with torch.no_grad():
            outputs = model(img)
            scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
            points = outputs['pred_points'][0][scores > 0.4].detach().cpu().numpy()

        img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
        size = max(2, int(0.005 * max(img_raw.size)))
        for p in points:
            img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
        return img_to_draw, points

def finalize_video(input_path):
    output_path = input_path.replace(".mp4", "_final.mp4")
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists
        "-i", input_path,
        "-vcodec", "libx264",  # HTML5-compatible codec
        "-movflags", "faststart",  # Ensures correct metadata is written at beginning
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path

# Video processing (1 FPS)
def process_video(video):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return None, "Could not open video"

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = tempfile.mktemp(suffix=".mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    # print((new_width, new_height))
    current_frame = 0
    frame_interval = int(fps)  # 1 frame per second
    points = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame % frame_interval == 0:
            processed_frame, pred_points = predict_frame(frame)
            points = pred_points
            cv2.putText(
                processed_frame,
                f"Count: {len(pred_points)}",
                (10, processed_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            writer.write(cv2.resize(processed_frame, (w, h)))
        else:
            processed_frame, _ = predict_frame(frame, skip=True, pred_points=points)
            cv2.putText(
                processed_frame,
                f"Count: {len(points)}",
                (10, processed_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            writer.write(cv2.resize(processed_frame, (w, h)))

        current_frame += 1

    cap.release()
    writer.release()
    final_output = finalize_video(out_path)

    return final_output


# Gradio interface
demo = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Video Source"),
    outputs=[
        gr.Video(label="Processed Video"),
    ],
    title="P2PNet Video Crowd Counting",
    description="Detects people once per second in uploaded video. Red dots indicate detected persons. Count shown on each annotated frame.",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860, debug=True)
