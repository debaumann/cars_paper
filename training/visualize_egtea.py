import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio


path ='/cluster/home/debaumann/cars_paper/train_visuals_egtea_att'
save_path = '/cluster/home/debaumann/cars_paper/visuals'

def visualize_egtea_att(epoch):
    gif = []
    for i in range(8):
        img = cv2.imread(os.path.join(path, f'{epoch}_input_{i}.png'))
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(np.array(img).shape)
        attention = cv2.imread(os.path.join(path, f'{epoch}_heat_att_{i}.png'))
        img = cv2.addWeighted(img, 0.5, attention, 0.5, 0)
        gif.append(img)
    imageio.mimsave(os.path.join(save_path, f'{epoch}_attention.gif'), gif, duration = 500)

visualize_egtea_att(7)