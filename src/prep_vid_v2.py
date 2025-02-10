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
import mediapipe as mp
import numpy as np
from torchvision.io import read_image
from torchvision import transforms


class HeadPose:
    def __init__(self):
        self.model_paths = ["models/head_pose/fsanet-1x1-iter-688590.onnx", "models/head_pose/fsanet-var-iter-688590.onnx"]
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
FACE_DETECTOR = MTCNN(image_size=112, margin=20)
TRANSFORM = transforms.Compose([
    transforms.Lambda(lambda x: x.float())
])

# def transform(image):
#     trans = transforms.Compose([
#             transforms.ToTensor(),
#         ])
#     return trans(image)

def check_frontal_face(image):
    yaw, pitch, roll = HEAD_POSE(image)
    return abs(yaw) < 30 and abs(pitch) < 30 and abs(roll) < 30


def select_frontal_frames(frames: list, sample: int=1):
    frontal_frames = []
    for frame in frames:
        if check_frontal_face(frame):
            frontal_frames.append(frame)
    if len(frontal_frames) < sample:
        return frontal_frames
    return random.sample(frontal_frames, sample)


def detect_and_preprocess_face(image):
    cropped = []
    for img in image:
        img_cropped = FACE_DETECTOR(img, save_path='test.jpg')
        img_cropped = TRANSFORM(img_cropped)
        img_cropped = img_cropped / 255.0
        cropped.append(img_cropped)
        if img_cropped is None:
            return None
    if len(cropped) == 0:
        return None

    return torch.stack(cropped)



def process_video(video: Union[str, Path], length: int=4, step: int=1, num_frames: int=1, crop: bool=True):
    
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

    images = []

    for start_frame in range(0, usable_frames - frames_per_segment + 1, step_frames):
        end_frame = start_frame + frames_per_segment
        segment = all_frames[start_frame:end_frame]
        frontal_face = select_frontal_frames(segment, num_frames)
        if not frontal_face:
            return None
        images.extend(frontal_face)
    if crop:
        images = detect_and_preprocess_face(np.array(images))
        if images is None:
            return None
    
    if not isinstance(images, torch.Tensor):
        images = torch.tensor(images)

    return images

    
def get_images_tensor(video: Union[str, Path, cv2.VideoCapture], length: int=4, step: int=1, num_frames: int=1, crop: bool=True):
    if  isinstance(video, (str,Path)):
        video = cv2.VideoCapture(str(video))
    images = process_video(video, length, step, num_frames, crop)
    return images
