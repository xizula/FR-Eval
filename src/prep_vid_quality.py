import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.environ['PYTHONPATH'] = parent_dir
sys.path.append(parent_dir)
from pathlib import Path
from facenet_pytorch import MTCNN
from PIL import Image
import cv2
import argparse
from tqdm import tqdm
import os
import random
import onnxruntime
import numpy as np
import torch 
from torchvision import transforms
from typing import Union
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torchvision.io import read_image
from torchvision import transforms
from model import load_quality_method
from torch import nn

class HeadPose:
    def __init__(self):
        self.model_paths = [r"D:\Projekt\models\head_pose\fsanet-1x1-iter-688590.onnx", r"D:\Projekt\models\head_pose\fsanet-var-iter-688590.onnx"]
        self.models = [onnxruntime.InferenceSession(model_path) for model_path in self.model_paths]
    
    def __call__(self, image):
        image = [self.transform(image)]
        yaw_pitch_roll_results = [
            model.run(["output"], {"input": image})[0] for model in self.models
        ]
        yaw, pitch, roll = np.mean(np.vstack(yaw_pitch_roll_results), axis=0)
        return yaw, pitch, roll
    
    def transform(self, image):
        trans = transforms.Compose([
            Resize(64),
            Normalize(mean=127.5,std=128),
            ToTensor()
            ])
        image = trans(image)
        return image 
    

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = (img-self.mean)/self.std
        return img


class ToTensor(object):
    def __call__(self, img):
        img = img.transpose((2, 0, 1))

        return torch.from_numpy(img)

class Resize(object):
    def __init__(self, size=64):
        self.size = (size, size)

    def __call__(self, img):
        img = cv2.resize(img, self.size)
        return img


HEAD_POSE = HeadPose()
FACE_DETECTOR = MTCNN(image_size=112, margin=20, post_process=False)
TRANSFORM = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
SUPPORTED_QUALITY_CHECKS = ['sdd_fiqa_mod', 'sdd_fiqa', 'cr_fiqa']

def check_frontal_face(image):
    yaw, pitch, roll = HEAD_POSE(image)
    return abs(yaw) < 20 and abs(pitch) < 20 and abs(roll) < 20


def get_best_quality_frame(images: torch.tensor, estimator: nn.Module):
    with torch.no_grad():
        images = images[::10]
        qualities = estimator(images.cuda())
    best_quality = torch.argmax(qualities).item()
    best_quality_frame = images[best_quality]
    best_quality_frame = best_quality_frame / 255.0  # Normalize to [0, 1] range
    transformed  = TRANSFORM(best_quality_frame)
    return transformed


def select_frontal_frames(frames: list, sample: int=1):
    frontal_frames = []
    for frame in frames:
        if check_frontal_face(frame):
            frontal_frames.append(frame)
    if len(frontal_frames) < sample:
        return frontal_frames
    return frontal_frames


def detect_and_preprocess_face(image):
    cropped = []
    for img in image:
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_cropped = FACE_DETECTOR(img)
        if img_cropped is None:
            continue
        cropped.append(img_cropped)
    if len(cropped) == 0:
        return None

    return torch.stack(cropped)



def process_video(video: Union[str, Path], length: int=4, step: int=1, num_frames: int=1, crop: bool=True, estimator: nn.Module=None):
    
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    usable_frames = (total_frames // fps) * fps
    frames_per_segment = length * fps
    step_frames = step * fps

    all_frames = []
    for _ in range(usable_frames):  # Process only usable frames
        ret, frame = video.read()
        if not ret:
            break
        all_frames.append(frame)
    video.release()

    final_images = []

    for start_frame in range(0, usable_frames - frames_per_segment + 1, step_frames):
        end_frame = start_frame + frames_per_segment
        segment = all_frames[start_frame:end_frame]
        frontal_face = select_frontal_frames(segment, num_frames)
        if not frontal_face:
            return None
        if crop:
            images = detect_and_preprocess_face(frontal_face)
            if images is None:
                return None
        if estimator is not None:
            best_frame = get_best_quality_frame(images, estimator)

        final_images.append(best_frame.numpy())
    
    if not isinstance(final_images, torch.Tensor):
        final_images = torch.tensor(final_images)

    return final_images

    
def get_images_tensor(video: Union[str, Path, cv2.VideoCapture], length: int=4, step: int=1, num_frames: int=1, crop: bool=True, quality_check: str='sdd_fiqa_mod'):
    if isinstance(video, (str,Path)):
        video = cv2.VideoCapture(str(video))
    estimator = load_quality_method(quality_check) if quality_check is not None else None
    images = process_video(video, length, step, num_frames, crop, estimator)
    return images
