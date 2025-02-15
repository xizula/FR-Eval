import numpy as np
from scipy.optimize import brentq
import os
from moviepy import VideoFileClip

def difference(x_value, x_array, y1_array, y2_array):
    y1_interp = np.interp(x_value, x_array, y1_array)
    y2_interp = np.interp(x_value, x_array, y2_array)
    return y1_interp - y2_interp


def calculate_eer(x, y1, y2):
    intersection_x = brentq(difference, x[0], x[-1], args=(x, y1, y2))
    intersection_y = np.interp(intersection_x, x, y1)

    return intersection_x, intersection_y


def delete_short_videos(root_folder, min_duration=4):
    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
                file_path = os.path.join(foldername, filename)
                try:
                    with VideoFileClip(file_path) as video:
                        duration = video.duration  # czas trwania w sekundach
                        if duration < min_duration:
                            print(f"Usuwanie: {file_path} (czas trwania: {duration:.2f}s)")
                            os.remove(file_path)
                except Exception as e:
                    print(f"Nie udało się przetworzyć pliku {file_path}: {e}")

