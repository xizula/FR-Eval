from pathlib import Path
from facenet_pytorch import MTCNN
from PIL import Image
import cv2
import argparse
from tqdm import tqdm
import os
import random
from model import HeadPose
import mediapipe as mp
import numpy as np
from matplotlib import pyplot as plt
os.environ["GLOG_minloglevel"] = "2"

HEAD_POSE = HeadPose()


def delete_small_folders(root_folder, min_videos=5):
    root_path = Path(root_folder)

    for subfolder in root_path.iterdir():
        if subfolder.is_dir():
            videos = list(subfolder.glob('*.*'))
            video_files = [vid for vid in videos if vid.suffix.lower() in ['.mp4', '.avi', '.mov', '.flv', '.wmv', '.webm']]
            
            if len(video_files) < min_videos:
                for file in subfolder.iterdir():
                    file.unlink() 
                subfolder.rmdir()  
                print(f"Deleted folder: {subfolder}")

def create_image_dataset(root_folder, sample, output_folder=None, suffix = "_images"):
    root_path = Path(root_folder)
    if output_folder is None:
        output_folder = root_path.parent / (root_path.name + suffix)
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    for subfolder in tqdm(root_path.iterdir()):
        if subfolder.is_dir():
            for video_path in subfolder.glob('*.*'):
                if video_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.flv', '.wmv', '.webm']:
                    video_output_path = output_path / subfolder.name
                    video_output_path.mkdir(exist_ok=True)
                    video_output_path = output_path / subfolder.name / video_path.stem
                    video_output_path.mkdir(exist_ok=True)
                    cap = cv2.VideoCapture(str(video_path))
                    frame_id = 0
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if total_frames < sample:
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            if check_frontal_face(frame):
                                frame_output_path = video_output_path / f"{frame_id}.jpg"
                                try:
                                    image = detect_and_preprocess_face(frame)
                                    image = align_face(image)
                                    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                                    cv2.imwrite(str(frame_output_path), bgr_image)
                                    frame_id += 1
                                except Exception as e:
                                    print(f"No face detected: {video_path}")
                    else:
                        frontal_frames = []
                        for idx in range(total_frames):
                            ret, frame = cap.read()
                            if not ret:
                                break
                            if check_frontal_face(frame):
                                frontal_frames.append(idx)
                                
                        frame_indices = sorted(random.sample(range(len(frontal_frames)), sample)) if len(frontal_frames) > sample else None
                        frames = [frontal_frames[i] for i in frame_indices] if frame_indices is not None else frontal_frames
                        for id in frames:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, id - 1)
                            ret, frame = cap.read()
                            if ret:
                                frame_path =video_output_path / f"{frame_id}.jpg"
                                try:
                                    image = detect_and_preprocess_face(frame)
                                    # image = align_face(image)
                                    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                                    cv2.imwrite(frame_path, bgr_image)
                                    frame_id += 1
                                except Exception as e:
                                    print(f"No face detected: {video_path}")

                    cap.release()

def detect_and_preprocess_face(image, target_size=(112, 112)):
    """
    Detects a face, crops it, and resizes to the target size using Mediapipe.
    Args:
        image: Input image (BGR format).
        target_size: Tuple indicating the desired output dimensions (width, height).
    Returns:
        Cropped and resized face or None if no face is detected.
    """
    fd = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
    with fd as face_detection:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = face_detection.process(rgb_image)
        # print(result)
        
        if not result.detections:
            return None 
        
        detection = result.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        
        h, w, _ = image.shape
        x_min = int(bboxC.xmin * w)
        y_min = int(bboxC.ymin * h)
        x_max = int((bboxC.xmin + bboxC.width) * w)
        y_max = int((bboxC.ymin + bboxC.height) * h)
        
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)
        
        face = image[y_min:y_max, x_min:x_max]
        
        resized_face = cv2.resize(face, target_size, interpolation=cv2.INTER_AREA)
        return resized_face
    

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
        # rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image to detect facial landmarks
        result = face_mesh.process(image)
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
    


def check_frontal_face(image):
    yaw, pitch, roll = HEAD_POSE(image)
    return abs(yaw) < 30 and abs(pitch) < 30 and abs(roll) < 30

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="path to the image dataset to preprocess")
    parser.add_argument("-m", "--min_videos", help="minimum number of videos to keep a folder", type=int, default=5)
    parser.add_argument("-s", "--sample", help="number of images to sample from the videos", type=int, default=5)
    parser.add_argument("--suffix", help="suffix_to_folder", type=str, default="_images")
    args = parser.parse_args()

    delete_small_folders(args.path, args.min_videos)
    create_image_dataset(args.path, args.sample, suffix=args.suffix)