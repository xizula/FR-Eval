import os
import random

def sample_and_keep_videos_per_folder(root_folder, num_videos_to_keep=50):
    for subdir, _, files in os.walk(root_folder):
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv') 
        
        video_paths = [
            os.path.join(subdir, file) 
            for file in files 
            if file.lower().endswith(video_extensions)
        ]
        
        total_videos = len(video_paths)
        
        if total_videos <= num_videos_to_keep:
            print(f"Folder '{subdir}' contains {total_videos} videos. No files will be deleted.")
            continue

        videos_to_keep = set(random.sample(video_paths, num_videos_to_keep))
        print(f"Selected {num_videos_to_keep} videos to keep in folder: {subdir}")

        deleted_count = 0
        for video in video_paths:
            if video not in videos_to_keep:
                try:
                    os.remove(video)
                    deleted_count += 1
                    print(f"Deleted: {video}")
                except Exception as e:
                    print(f"Error deleting {video}: {e}")
        
        print(f"Deleted {deleted_count} videos in folder: {subdir}. Retained {num_videos_to_keep} videos.")


root_folder = "../data/datasets/test" 
sample_and_keep_videos_per_folder(root_folder, num_videos_to_keep=50)
