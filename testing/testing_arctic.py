from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
import wandb
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.vit_train_utils import MultiModalDataset, Cars_Action,preprocess,set_seed, compute_soft_iou, TestDataset, compute_top_3





# Create dataset instances for training and validation.
def main():
    seed = 7
    set_seed(seed)
    # Set data paths and subject splits
    data_root = '/cluster/scratch/debaumann/arctic_data'
    save_batch_dir = '/cluster/home/debaumann/cars_paper/test_visuals_tvt_att'
    os.makedirs(save_batch_dir, exist_ok=True)
    train_subjects = ['S01','S02','S04','S05', 'S08',  'S09']
    val_subjects = ['S07','S10']
    test_subjects = ['S03','S06']
    
    testdata = TestDataset(data_root, test_subjects)
    test_loader = DataLoader(testdata, batch_size=1, shuffle=False)
    

    # Wrap the datasets in DataLoaders.
    
    model = Cars_Action()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Load the best saved_model
    save_dir = f'{data_root}/models_tvt'
    best_model_path = os.path.join(save_dir, "best_cars_action_model_tvt.pth")
    print('loading model')
    model.load_state_dict(torch.load(best_model_path))
    print('model loaded')

    model.to(device)
    preprocessor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    wandb.init(project="cars_action_project_arctic", name="Cars_Action_training_run_att")
    
    

    num_epochs = 1
    alpha = 1.0  # Weight for classification loss
    beta = 20.0
    gamma = 20.0

    for epoch in range(num_epochs):

        # --- Validation Loop ---
        model.eval()
        total = 0
        correct = 0
        top5 = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}")):
                imgs, labels = batch
                
                rgb_imgs= preprocess(imgs,preprocessor)
                rgb_imgs = rgb_imgs.squeeze(0).to(device)
                
                labels = labels.to(device)
                
                pixel_values = rgb_imgs
                
                logits,hand,obj = model(pixel_values)
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                top5 += compute_top_3(logits, labels)

                if batch_idx == 380:
                # Create a figure with 5 rows (Input, Hand GT, Object GT, Hand Attn, Obj Attn) and up to 8 columns.
                    fig, axes = plt.subplots(5, 8, figsize=(20, 15))
                    for j in range(8):
                        # Row 0: Input image
                        print(imgs[0,j].shape)
                        img_np = imgs[0,j].permute(1, 2, 0).cpu().numpy()
                        axes[0, j].imshow(img_np)
                        axes[0, j].set_title(f'Input {j+1}')
                        axes[0, j].axis('off')
                        plt.imsave(os.path.join(save_batch_dir, f'{epoch+1}_input_{j}.png'), img_np / 255.0)

                        
                        # Row 3: Hand attention map (model output)
                        hand_att_np = hand[j].squeeze(0).cpu().numpy()
                        axes[1, j].imshow(hand_att_np, cmap='hot')
                        axes[1, j].set_title(f'Hand Attn {j+1}')
                        axes[1, j].axis('off')
                        plt.imsave(os.path.join(save_batch_dir, f'{epoch+1}_hand_att_{j}.png'), hand_att_np, cmap='hot')
                        
                        # Row 4: Object attention map (model output)
                        obj_att_np = obj[j].squeeze(0).cpu().numpy()
                        axes[2, j].imshow(obj_att_np, cmap='hot')
                        axes[2, j].set_title(f'Obj Attn {j+1}')
                        axes[2, j].axis('off')
                        plt.imsave(os.path.join(save_batch_dir, f'{epoch+1}_obj_att_{j}.png'), obj_att_np, cmap='hot')

                    
                    plt.close(fig)
                
                test_acc = correct / total
                test_top5 = top5 / total
                wandb.log({"Test Accuracy": test_acc, "Test Top-5 Accuracy": test_top5})



                
                # Compute classification loss

if __name__ == "__main__":
    main()