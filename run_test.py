import argparse
import datetime
import random
import time
from pathlib import Path

import torch
import torchvision.transforms as standard_transforms
import numpy as np

from PIL import Image
import cv2
from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
import warnings
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)
    
    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    parser.add_argument('--output_dir', default='',
                        help='path where to save')
    parser.add_argument('--weight_path', default='',
                        help='path where the trained weights saved')

    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')

    return parser

def main(args, debug=False):

    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    print(args)
    device = torch.device('cpu')
    # get the P2PNet
    model = build_model(args)
    # move to GPU
    model.to(device)
    # load trained model
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    # convert to eval mode
    model.eval()
    # create the pre-processing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # set your image path here
    test_img_dir = "./test_img"
    i=3
    for img_name in os.listdir("./test_img")[i:i+1]:
        img_path = os.path.join(test_img_dir,img_name)
        # load the images
        img_raw = Image.open(img_path).convert('RGB')
        # round the size
        width, height = img_raw.size
        # Compute scale ratio to fit within 800x600
        scale_w = 800 / width
        scale_h = 600 / height
        scale = min(scale_w, scale_h)

        # Compute new size while keeping aspect ratio
        new_width = int(width * scale)
        new_height = int(height * scale)

        # Round down to nearest multiple of 128
        new_width = (new_width // 128) * 128
        new_height = (new_height // 128) * 128

        # Resize the image
        img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
        # img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
        # pre-proccessing
        img = transform(img_raw)

        samples = torch.Tensor(img).unsqueeze(0)
        samples = samples.to(device)
        # run inference
        tik0 = time.time()
        outputs = model(samples)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
        latency = time.time() - tik0
        print(f"latency: {latency}")

        outputs_points = outputs['pred_points'][0]

        threshold = 0.4
        # filter the predictions
        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
        predict_cnt = int((outputs_scores > threshold).sum())

        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

        outputs_points = outputs['pred_points'][0]
        # draw the predictions
        size = 2
        size = max(size, int(0.005*max(img_raw.size)))
        img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
        for p in points:
            img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)

        num_points = len(points)
        print(f"Number of points: {num_points}")
        
        # Optionally, put the count text on the image
        cv2.putText(img_to_draw, f'Count: {num_points}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # save the visualized image
        out_file_name = os.path.splitext(os.path.basename(img_name))[0]
        cv2.imwrite(os.path.join(args.output_dir, '{}_pred_{}.jpg'.format(out_file_name,predict_cnt)), img_to_draw)
        print('{}_pred_{}.jpg'.format(out_file_name,predict_cnt))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)