import os
from typing import List, Tuple, Dict
import ast
import json
import numpy as np
import cv2
import pandas as pd
from functools import cache
import time

def process_video_names(video_names: List[str]) -> List[str]:
    processed_video_names: List[str] = []

    for item in video_names:
        processed_video_name: str = ast.literal_eval(item)[0]
        processed_video_names.append(processed_video_name)

    return processed_video_names


def process_time_stamps(time_stamps: List[str]) -> List[Tuple[float, float]]:
    processed_time_stamps: List[Tuple[float, float]] = []

    for item in time_stamps:
        processed_time_stamp: Tuple[float, float] = tuple(ast.literal_eval(item))
        processed_time_stamps.append(processed_time_stamp)

    return processed_time_stamps


def process_labels(labels: List[str], label_dict_path: str) -> List[int]:
    processed_labels_str: List[str] = []

    for item in labels:
        processed_label_str: str = (json.loads(item)["1"])
        processed_labels_str.append(processed_label_str)
    
    with open(label_dict_path, "r", encoding="utf-8") as f:
        label_dict: Dict[str, int] = json.load(f) 
    
    processed_labels: List[int] = []

    for label in processed_labels_str:
        if label in label_dict:
            processed_labels.append(label_dict[label])
        else:
            raise Exception(f"Label is not in the dictionary!")
    return processed_labels


def parse_csv(path: str, label_dict_path: str) -> Tuple[List[str], List[Tuple[float, float]], List[int]]:
    
    df = pd.read_csv(path, comment='#', header=None)
    
    video_names: List[str] = list(df[1])
    processed_video_names: List[str] = process_video_names(video_names)

    time_stamps: List[str] = list(df[3])
    processed_time_stamps: List[Tuple[float, float]] = process_time_stamps(time_stamps)

    labels: List[str] = list(df[5])
    processed_labels: List[int] = process_labels(labels, label_dict_path)
    
    # this block is to filter out accidental bounding box labels
    indices_to_keep = [i for i, tup in enumerate(processed_time_stamps) if len(tup) > 1]
    processed_video_names = [processed_video_names[i] for i in indices_to_keep]
    processed_time_stamps = [processed_time_stamps[i] for i in indices_to_keep]
    processed_labels = [processed_labels[i] for i in indices_to_keep]
    
    return processed_video_names, processed_time_stamps, processed_labels


@cache
def video_to_frames(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Could not open video")
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.uint8)
        frames.append(frame)
    cap.release()
    frames = np.stack(frames, dtype=np.uint8)
    
    return frames


def extract_seq(frames: np.ndarray, start_time: float, end_time: float, frame_rate: float, num_samples: int) -> np.ndarray:
    start_frame: int = int(start_time * frame_rate)
    end_frame: int = int(end_time * frame_rate)
    sample_indices: np.ndarray = np.linspace(start_frame, end_frame, num_samples, dtype=np.int32)

    frame_samples: np.ndarray = frames[sample_indices]

    return frame_samples


def show_seq(frame_samples):
    
    delay = 500

    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Video", 100, 100)
    cv2.resizeWindow("Video", 640, 480)
    for _, frame in enumerate(frame_samples):
    
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Video", frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    input()
    cv2.destroyAllWindows()


def extract_sequences(data_path: str, video_names: List[str], time_stamps: List[Tuple[float, float]], frame_rate: float, num_samples: int, folder: str):
    if len(video_names) != len(time_stamps):
        raise Exception("Length of video_names and timestamps need to be the same")
    os.makedirs(folder, exist_ok = True)
    for i in range(len(video_names)):
        video_path: str = os.path.join(data_path, video_names[i])
        
        frames: np.ndarray = video_to_frames(video_path)

        start_time: float = time_stamps[i][0]
        end_time: float = time_stamps[i][1]

        frame_samples: np.ndarray = extract_seq(frames, start_time, end_time, frame_rate, num_samples)
        
        
        show_seq(frame_samples)
        
        save_path: str = os.path.join(folder, str(i))
        np.save(save_path, frame_samples)


        


def main():
    LABEL_ROOT_PATH: str = "/mnt/c/Users/chris/Desktop/Code/Arctic_Labels/"
    DATA_ROOT_PATH: str = "/mnt/d/"
    LABEL_DICT_PATH: str = "/home/chrislx/dev/pyprojects/computervision/paper/seq_extraction/labels.json"
    FRAME_RATE: float = 30.0
    NUM_SAMPLES: int = 8
    FOLDER: str = "S01"

    labels_path: str = "Arctic-S01.csv"
    data_path: str = "s01-20241031T161116Z-001/s01/"

    full_labels_path: str = os.path.join(LABEL_ROOT_PATH, labels_path)
    full_data_path: str = os.path.join(DATA_ROOT_PATH, data_path)
    
    video_names, time_stamps, labels = parse_csv(full_labels_path, LABEL_DICT_PATH)
    extract_sequences(full_data_path, video_names, time_stamps, FRAME_RATE, NUM_SAMPLES, FOLDER)
    
    labels = np.array(labels, dtype=np.int32)
    labels_name: str = "labels_" + FOLDER + ".npy"
    np.save(labels_name, labels)

if __name__ == "__main__":
    main()
