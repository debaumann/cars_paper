import numpy as np
from numpy.typing import NDArray
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

def extract(path: NDArray, start_act: NDArray, end_act: NDArray):
    path = path
    for i in tqdm(range(path.shape[0])):
        sample_frames: NDArray = np.linspace(start_act[i], end_act[i], num=8, dtype=np.int32)
        images = []
        for sample_frame in sample_frames:
            image_path = f"/home/chrislx/dev/pyprojects/computervision/paper/h2odataset/{path[i]}/cam4/rgb/{sample_frame:06d}.png"
            image = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB).astype(np.uint8)
            image = image[:, 136:1144, :]
            image = cv2.resize(image, (720, 720), interpolation=cv2.INTER_AREA)
            images.append(image)
            #plt.imshow(image)
            #plt.show()
            np.save(f"/home/chrislx/dev/pyprojects/computervision/paper/h2odataset/sequences_test/{i}.npy", images)


def main():
    LABELS_FILE: str = "/home/chrislx/dev/pyprojects/computervision/paper/h2odataset/action_test.txt"

    df = pd.read_csv(LABELS_FILE, delimiter=" ")
    path: NDArray = df['path'].to_numpy()
    #action_label: NDArray = df['action_label'].to_numpy()
    start_act: NDArray = df['start_act'].to_numpy()
    end_act: NDArray = df['end_act'].to_numpy()
    
    #np.save("action_test.npy", action_label)

    extract(path, start_act, end_act)
    

if __name__ == "__main__":
    main()
