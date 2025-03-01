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

from utils.vit_egtea_utils import MultiModalDataset, Cars_Action,preprocess,set_seed, compute_iou





# Create dataset instances for training and validation.
def main():
    seed = 7
    set_seed(seed)
    # Set data paths and subject splits
    data_root = '/cluster/scratch/debaumann/egtea/egtea'
    save_batch_dir = '/cluster/home/debaumann/cars_paper/train_visuals_egtea_base'
    os.makedirs(save_batch_dir, exist_ok=True)
    train_subjects = 'train_split1'
    val_subjects = 'test_split1'
    
    train_dataset = MultiModalDataset(data_root, train_subjects)
    val_dataset = MultiModalDataset(data_root, val_subjects)

    # Wrap the datasets in DataLoaders.
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = Cars_Action()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    preprocessor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")


    # Initialize wandb (customize project name and run name as needed)
    wandb.init(project="cars_action_project_egtea", name="Cars_Action_training_run_base")

    


    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Set up model saving paths
    save_dir = f'{data_root}/models_egtea'
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')
    best_model_path = os.path.join(save_dir, "best_cars_action_model_egtea_base.pth")

    num_epochs = 22
    alpha = 1.0  # Weight for classification loss
    beta = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_class_loss = 0.0
        running_heat_loss = 0.0

        running_heat_iou = 0.0

        total_train = 0
        correct_train = 0

        # --- Training Loop ---
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            # Unpack the batch: adjust the order if needed
            
            rgb_imgs, heat_heatmap, labels = batch
            heat_heatmap = heat_heatmap.squeeze(0).squeeze(1)
            # Process inputs (no resizing/cropping)
            rgb_imgs= preprocess(rgb_imgs,preprocessor)
            rgb_imgs = rgb_imgs.squeeze(0).to(device)
            heat_heatmap = heat_heatmap.to(device)

            labels = labels.to(device)
            
            # For ViT, we assume the preprocessed rgb images are the pixel_values.
            pixel_values = rgb_imgs

            optimizer.zero_grad()
            # Forward pass; assume the model returns (logits, attentions)
            logits,heat = model(pixel_values)
            
            # Compute classification loss
            loss_class = criterion(logits, labels)
            
            # Process attentions for comparison with heatmaps:
            # Ensure attentions have a channel dimension (if not, add one)
            # and resize to match the heatmap spatial dimensions.
            # Here, we assume 'attentions' is of shape [batch, H_att, W_att].
            
            
            # Compute MSE losses for heat and object heatmaps
            
            heat_loss = F.mse_loss(heat, heat_heatmap)

            
            # Total loss
            loss = alpha * loss_class + beta * heat_loss
            
            running_loss += loss.item()
            running_class_loss += loss_class.item()
            running_heat_loss += heat_loss.item()
            heat_iou = compute_iou(heat, heat_heatmap)

            running_heat_iou += heat_iou

            
            loss.backward()
            optimizer.step()
            
            # Compute training accuracy
            _, predicted = torch.max(logits, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        avg_class_loss = running_class_loss / len(train_loader)
        avg_heat_loss = running_heat_loss / len(train_loader)
        avg_heat_iou = running_heat_iou / len(train_loader)
        train_accuracy = correct_train / total_train
        print(f"Epoch [{epoch+1}/{num_epochs}] Train: Loss: {avg_train_loss:.4f}, Class Loss: {avg_class_loss:.4f}, "
            f"heat Loss: {avg_heat_loss:.4f}, Accuracy: {train_accuracy:.4f}")
        
        wandb.log({
            "train_loss": avg_train_loss,
            "class_loss": avg_class_loss,
            "heat_loss": avg_heat_loss,
            "train_accuracy": train_accuracy,
            "heat_iou": avg_heat_iou,
            "epoch": epoch+1
        })
        
        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        val_class_loss = 0.0
        val_heat_loss = 0.0
        val_heat_iou=0.0

        total_val = 0
        correct_val = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}")):
                imgs, heat_heatmap, labels = batch
                heat_heatmap = heat_heatmap.squeeze(0).squeeze(1)
                rgb_imgs= preprocess(imgs,preprocessor)
                rgb_imgs = rgb_imgs.squeeze(0).to(device)
                heat_heatmap = heat_heatmap.squeeze(1).to(device)

                labels = labels.to(device)
                
                pixel_values = rgb_imgs
                
                logits,heat = model(pixel_values)
                
                # Compute classification loss
                loss_class = criterion(logits, labels)
                
                # Process attentions for comparison with heatmaps:
                # Ensure attentions have a channel dimension (if not, add one)
                # and resize to match the heatmap spatial dimensions.
                # Here, we assume 'attentions' is of shape [batch, H_att, W_att].
                
                
                # Compute MSE losses for heat and object heatmaps

                loss = alpha * loss_class + beta * heat_loss
                
                val_loss += loss.item()
                val_class_loss += loss_class.item()
                val_heat_loss += heat_loss.item()

                
                heat_iou = compute_iou(heat, heat_heatmap)
                val_heat_iou += heat_iou

                _, predicted = torch.max(logits, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
                # Log visualizations for the first batch of the validation epoch.
                if batch_idx == 333:
                # Create a figure with 5 rows (Input, heat GT, Object GT, heat Attn, Obj Attn) and up to 8 columns.
                    fig, axes = plt.subplots(5, 8, figsize=(20, 15))
                    for j in range(8):
                        # Row 0: Input image
                        print(imgs[0,j].shape)
                        img_np = imgs[0,j].permute(1, 2, 0).cpu().numpy()
                        axes[0, j].imshow(img_np)
                        axes[0, j].set_title(f'Input {j+1}')
                        axes[0, j].axis('off')
                        plt.imsave(os.path.join(save_batch_dir, f'{epoch+1}_input_{j}.png'), img_np / 255.0)

                        # Row 1: heat heatmap (ground-truth)
                        heat_np = heat_heatmap[j].squeeze(0).cpu().numpy()
                        axes[1, j].imshow(heat_np, cmap='hot')
                        axes[1, j].set_title(f'heat HM {j+1}')
                        axes[1, j].axis('off')
                        plt.imsave(os.path.join(save_batch_dir, f'{epoch+1}_heat_gt_{j}.png'), heat_np, cmap='hot')
                        
                        # Row 3: heat attention map (model output)
                        heat_att_np = heat[j].squeeze(0).cpu().numpy()
                        axes[3, j].imshow(heat_att_np, cmap='hot')
                        axes[3, j].set_title(f'heat Attn {j+1}')
                        axes[3, j].axis('off')
                        plt.imsave(os.path.join(save_batch_dir, f'{epoch+1}_heat_att_{j}.png'), heat_att_np, cmap='hot')
                    
                    wandb.log({"validation_visuals": wandb.Image(fig)})
                    plt.close(fig)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_class_loss = val_class_loss / len(val_loader)
        avg_val_heat_loss = val_heat_loss / len(val_loader)

        val_accuracy = correct_val / total_val
        avg_val_heat_iou = val_heat_iou / len(val_loader)

        print(f"Validation Epoch [{epoch+1}/{num_epochs}] Loss: {avg_val_loss:.4f}, Class Loss: {avg_val_class_loss:.4f}, "
            f"heat Loss: {avg_val_heat_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        
        wandb.log({
            "val_loss": avg_val_loss,
            "val_class_loss": avg_val_class_loss,
            "val_heat_loss": avg_val_heat_loss,
            "val_accuracy": val_accuracy,
            "val_heat_iou": avg_val_heat_iou,
            "epoch": epoch+1
        })
        
        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with validation loss: {avg_val_loss:.4f}")

    # Save the final trained model
    final_model_path = os.path.join(save_dir, "final_cars_action_model_egtea_base.pth")
    torch.save(model.state_dict(), final_model_path)
    print("Training complete, final model saved.")
    wandb.finish()

if __name__ == "__main__":
    main()