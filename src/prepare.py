from pathlib import Path
from facenet_pytorch import MTCNN
from PIL import Image
import cv2
import argparse
from tqdm import tqdm
import os
import random

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

def create_image_dataset(root_folder, sample, output_folder=None):
    root_path = Path(root_folder)
    if output_folder is None:
        output_folder = root_path.parent / (root_path.name + "_images")
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
                            frame_output_path = video_output_path / f"{frame_id}.jpg"
                            cv2.imwrite(str(frame_output_path), frame)
                            frame_id += 1
                    else:
                        frame_indices = sorted(random.sample(range(total_frames), sample))
                        for idx in range(total_frames):
                            ret, frame = cap.read()
                            if not ret:
                                break
                            if idx in frame_indices:
                                frame_path =video_output_path / f"{frame_id}.jpg"
                                cv2.imwrite(frame_path, frame)
                                print(f"Saved frame {idx} to {frame_path}")
                                frame_id += 1
                                if frame_id == sample:
                                    break

                    cap.release()
    return output_folder

def crop_faces(folder_path):
    mtcnn = MTCNN(image_size=112, margin=20)
    root_path = Path(folder_path)

    for subfolder in root_path.iterdir():
        if subfolder.is_dir():
            for ssfolder in subfolder.iterdir():
                if ssfolder.is_dir():
                    for image_path in ssfolder.glob('*.*'):
                        if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
                            img = Image.open(image_path)
                            img_cropped = mtcnn(img, save_path=[image_path])
                            if img_cropped is None:
                                os.remove(image_path)



def to_bgr(folder_path):
    root_path = Path(folder_path)

    for subfolder in root_path.iterdir():
        if subfolder.is_dir():
            for ssfolder in subfolder.iterdir():
                if ssfolder.is_dir():
                    for image_path in ssfolder.glob('*.*'):
                        if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
                            img = cv2.imread(str(image_path))
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(str(image_path), img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="path to the image dataset to preprocess")
    parser.add_argument("-m", "--min_videos", help="minimum number of videos to keep a folder", type=int, default=5)
    parser.add_argument("-s", "--sample", help="number of images to sample from the videos", type=int, default=5)
    args = parser.parse_args()

    delete_small_folders(args.path, args.min_videos)
    img_path = create_image_dataset(args.path, args.sample)
    crop_faces(img_path)
    to_bgr(img_path)