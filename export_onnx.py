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


dummy_input = torch.randn(1, 3, 336, 336, device=device)

onnx_path = "model_fixed_bs1.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    # No dynamic axes for fixed batch size
)

print(f"Model exported to {onnx_path} with fixed batch size 1.")