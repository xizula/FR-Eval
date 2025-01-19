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
        img_cropped = FACE_DETECTOR(img, save_path='test_w.jpg')
        img_cropped = cv2.imread('test_w.jpg')
        img_cropped = img_cropped.transpose(2, 0, 1)
        # print(img_cropped.shape)
        # print(img_cropped.min(), img_cropped.max())
        # # print("img cropped")
        # # # print(img_cropped.shape)
        # # img_crop = img_cropped.permute(1, 2, 0)
        
        # # img_crop = img_crop.numpy()
        # # print(np.min(img_crop), np.max(img_crop))
        # # img_crop = img_crop.astype(np.uint8)
        # # cv2.imwrite('test.jpg', img_crop)
        # # print("img saved")
        cropped.append(img_cropped)
        if img_cropped is None:
            return None
    if len(cropped) == 0:
        return None
    # print(len(cropped))
    # print(cropped[0].shape)
    return torch.tensor(cropped)


def align_face(image, target_size=(112, 112)):
    """
    Aligns the face by rotating it so the eyes are parallel to the bottom of the image
    and centers the nose tip in the middle of the image.
    Args:
        image: Input image (BGR format).
        target_size: Tuple indicating the desired output dimensions (width, height).
    Returns:
        Aligned image or None if no face is detected.
    """
    # Initialize Mediapipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        # Convert to RGB as Mediapipe works with RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image to detect facial landmarks
        result = face_mesh.process(rgb_image)
        if not result.multi_face_landmarks:
            return None 
        
        # Get the landmarks for the first detected face
        face_landmarks = result.multi_face_landmarks[0]
        
        # Extract key points for alignment
        h, w, _ = image.shape
        landmarks = [(int(landmark.x * w), int(landmark.y * h)) for landmark in face_landmarks.landmark]
        left_eye = np.mean([landmarks[33], landmarks[133]], axis=0)  
        right_eye = np.mean([landmarks[362], landmarks[263]], axis=0)  
        nose_tip = landmarks[1]  # Nose tip
        
        # Step 1: Align the eyes (rotation)
        delta_y = right_eye[1] - left_eye[1]
        delta_x = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(delta_y, delta_x))
        eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        M_rotate = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, M_rotate, (w, h), flags=cv2.INTER_CUBIC)
        
        # Update landmarks after rotation
        landmarks = np.array(landmarks)
        ones = np.ones((landmarks.shape[0], 1))
        landmarks_homo = np.hstack([landmarks, ones])
        rotated_landmarks = M_rotate @ landmarks_homo.T
        rotated_landmarks = rotated_landmarks.T
        nose_tip = rotated_landmarks[1]  
        
        # Step 2: Center the nose tip (translation)
        center_x, center_y = target_size[0] // 2, target_size[1] // 2
        nose_x, nose_y = int(nose_tip[0]), int(nose_tip[1])
        translation_x = center_x - nose_x
        translation_y = center_y - nose_y
        M_translate = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
        translated_image = cv2.warpAffine(rotated_image, M_translate, target_size, flags=cv2.INTER_CUBIC)
        
        return translated_image


def crop_faces(images: np.ndarray):
    imgs = detect_and_preprocess_face(images)
    return torch.tensor(imgs)
  


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
