from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import random
import attention_vit_utils
import json
from datetime import datetime



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
seed = 7
attention_vit_utils.set_seed(seed)
batch_size = 1
wandb.init(project="attention_transformer")
dataset_root = "/cluster/scratch/cbennewitz/"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
images_heatmaps_path = f"run_attention_{timestamp}"
os.mkdir(images_heatmaps_path)

def log_images_and_heatmaps(images, hand_attentions, obj_attentions, mode, epoch):
    for i in range(images.shape[1]):
        image = images[0, i, :, :, :].numpy().astype(np.uint8)
        hand_attention = hand_attentions[0, i, :, :].detach().cpu().numpy()
        obj_attention = obj_attentions[0, i, :, :].detach().cpu().numpy()
        plt.figure(figsize=(5, 5))
        plt.imshow(image)
        plt.axis("off")
        np.save(f"/cluster/home/cbennewitz/{images_heatmaps_path}/{mode}_{epoch+1}_{i}Input_Image.npy", image)
        image_plt = wandb.Image(plt, caption=f"{mode}_{epoch+1} Input Image {i+1}")
        wandb.log({f"{mode}_input_image_{i+1}": image_plt})
        plt.close()

        plt.figure(figsize=(5,5))
        plt.imshow(hand_attention, cmap="jet")
        plt.axis("off")
        np.save(f"/cluster/home/cbennewitz/{images_heatmaps_path}/{mode}_{epoch+1}_{i}Hand_Attention_Map.npy", hand_attention)
        hand_attention_plt = wandb.Image(plt, caption=f"{mode}_{epoch+1} Hand Attention Map {i+1}")
        wandb.log({f"{mode}_Hand_Attention_MAP_{i+1}": hand_attention_plt})
        plt.close()

        plt.figure(figsize=(5,5))
        plt.imshow(obj_attention, cmap="jet")
        plt.axis("off")
        np.save(f"/cluster/home/cbennewitz/{images_heatmaps_path}/{mode}_{epoch+1}_{i}Obj_Attention_Map.npy", obj_attention)
        obj_attention_plt = wandb.Image(plt, caption=f"{mode}_{epoch+1} Obj Attention Map {i+1}")
        wandb.log({f"{mode}_Obj_Attention_MAP_{i+1}": obj_attention_plt})
        plt.close()


def main():
    preprocessor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    model = attention_vit_utils.ViTMLPModel()
    model.to(device)

    train_data = attention_vit_utils.TrainData(dataset_root)
    val_data = attention_vit_utils.ValData(dataset_root)
    test_data = attention_vit_utils.TestData(dataset_root)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 30
    current_best = np.inf
    
    # Loss weights
    alpha = 1.0
    beta = 100.0
    gamma = 100.0
    

    for epoch in range(num_epochs):
        model.train()
        running_loss: float = 0.0
        running_class_loss: float = 0.0
        running_hand_loss: float = 0.0
        running_obj_loss: float = 0.0
        running_hand_iou: float = 0.0
        running_obj_iou: float = 0.0
        counter_train: int = 0
        train_correct: int = 0
        train_top3: int = 0
        train_total: int = 0
        for sequences, hand_masks, obj_masks, labels in tqdm(train_loader):
            optimizer.zero_grad()

            processed_sequences = attention_vit_utils.preprocess(preprocessor, sequences)
            processed_sequences = processed_sequences.to(device)
            labels = labels.to(device)
            hand_masks = hand_masks.to(device)
            obj_masks = obj_masks.to(device)

            output, hand_attentions, obj_attentions = model(processed_sequences)

            hand_attentions = F.interpolate(hand_attentions, size=(hand_masks.shape[2], hand_masks.shape[3]), mode="nearest")
            obj_attentions = F.interpolate(obj_attentions, size=(obj_masks.shape[2], obj_masks.shape[3]), mode="nearest")

            class_loss = criterion(output, labels)

            hand_loss = F.mse_loss(hand_attentions, hand_masks)
            obj_loss = F.mse_loss(obj_attentions, obj_masks)

            loss = alpha * class_loss + beta * hand_loss + gamma * obj_loss

            running_hand_iou += attention_vit_utils.soft_iou(hand_attentions, hand_masks)
            running_obj_iou += attention_vit_utils.soft_iou(obj_attentions, obj_masks)
            running_loss += loss.item()
            running_class_loss += class_loss.item()
            running_hand_loss += hand_loss.item()
            running_obj_loss += obj_loss.item()

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(output, 1)
            train_correct += (predicted == labels).sum().item()
            train_top3 += attention_vit_utils.top_3(output, labels)

            train_total += labels.size(0)
            


            if counter_train == 100:
                log_images_and_heatmaps(sequences, hand_attentions, obj_attentions, mode="train",epoch=epoch)

            counter_train += 1

        avg_train_loss = running_loss / counter_train
        avg_class_loss = running_class_loss / counter_train
        avg_hand_loss = running_hand_loss / counter_train
        avg_obj_loss = running_obj_loss / counter_train
        avg_hand_iou = running_hand_iou / counter_train
        avg_obj_iou = running_obj_iou / counter_train
        train_accuracy = train_correct / train_total
        train_top_3_accuracy = train_top3 / train_total
        

        wandb.log({"train_loss": avg_train_loss,
                   "class_loss": avg_class_loss,
                   "hand_loss": avg_hand_loss,
                   "objects_loss": avg_obj_loss,
                   "hand_iou": avg_hand_iou,
                   "obj_iou": avg_obj_iou,
                   "train_accuracy": train_accuracy,
                   "train_top3_accuracy": train_top_3_accuracy,
                   "epoch": epoch + 1})

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Class Loss: {avg_class_loss}, Hand Loss: {avg_hand_loss}, Obj Loss: {avg_obj_loss}, Hand IoU: {avg_hand_iou}, Obj IoU: {avg_obj_iou}, Train Accuracy: {train_accuracy*100:.2f}%, Train Top 3 Accuracy: {train_top_3_accuracy*100:.2f}%")

        model.eval()
        val_running_loss: float = 0.0
        val_class_loss: float = 0.0
        val_hand_loss: float = 0.0
        val_obj_loss: float = 0.0
        val_hand_iou: float = 0.0
        val_obj_iou: float = 0.0
        counter_val: int = 0
        val_correct: int = 0
        val_top3: int = 0
        val_total: int = 0
        with torch.no_grad():
            for sequences, hand_masks, obj_masks, labels in tqdm(val_loader):
                processed_sequences = attention_vit_utils.preprocess(preprocessor, sequences)
                processed_sequences = processed_sequences.to(device)
                
                labels = labels.to(device)
                hand_masks = hand_masks.to(device)
                obj_masks = obj_masks.to(device)

                output, hand_attentions, obj_attentions = model(processed_sequences)

                hand_attentions = F.interpolate(hand_attentions, size=(hand_masks.shape[2], hand_masks.shape[3]), mode="nearest")
                obj_attentions = F.interpolate(obj_attentions, size=(obj_masks.shape[2], obj_masks.shape[3]), mode="nearest")


                class_loss = criterion(output, labels)
                hand_loss = F.mse_loss(hand_attentions, hand_masks)
                obj_loss = F.mse_loss(obj_attentions, obj_masks)

                loss = alpha * class_loss + beta * hand_loss + gamma * obj_loss

                val_hand_iou += attention_vit_utils.soft_iou(hand_attentions, hand_masks)
                val_obj_iou += attention_vit_utils.soft_iou(obj_attentions, obj_masks)

                val_running_loss += loss.item()
                val_class_loss += class_loss.item()
                val_hand_loss += hand_loss.item()
                val_obj_loss += obj_loss.item()

                _, predicted = torch.max(output, 1)
                val_correct += (predicted == labels).sum().item()
                val_top3 += attention_vit_utils.top_3(output, labels)
                val_total += labels.size(0)

                if counter_val == 25:
                    log_images_and_heatmaps(sequences, hand_attentions, obj_attentions, mode="val", epoch=epoch)
                
                counter_val += 1

        avg_val_loss = val_running_loss / counter_val
        avg_val_class_loss = val_class_loss / counter_val
        avg_val_hand_loss = val_hand_loss / counter_val
        avg_val_obj_loss = val_obj_loss / counter_val
        avg_val_hand_iou = val_hand_iou / counter_val
        avg_val_obj_iou = val_obj_iou / counter_val
        val_accuracy = val_correct / val_total
        val_top_3_accuracy = val_top3 / val_total

        wandb.log({"val_loss": avg_val_loss,
                   "val_class_loss": avg_val_class_loss,
                   "val_hand_loss": avg_val_hand_loss,
                   "val_obj_loss": avg_val_obj_loss,
                   "val_hand_iou": avg_val_hand_iou,
                   "val_obj_iou": avg_val_obj_iou,
                   "val_accuracy": val_accuracy,
                   "val_top3_accuracy": val_top_3_accuracy,
                   "epoch": epoch + 1})

        print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}, Val Class Loss: {avg_val_class_loss}, Val Hand Loss: {avg_val_hand_loss}, Val Obj Loss: {avg_val_obj_loss}, Val Hand IoU: {avg_val_hand_iou}, Val Obj IoU: {avg_val_obj_iou}, Val Accuracy: {val_accuracy*100:.2f}%, Val Top 3 Accuracy: {val_top_3_accuracy*100:.2f}%")

        if current_best > avg_val_class_loss:
            current_best = avg_val_class_loss
            counter_test = 0
            predictions = []
            model.eval()
            with torch.no_grad():
                for sequences in tqdm(test_loader):
                    processed_sequences = attention_vit_utils.preprocess(preprocessor,sequences)
                    processed_sequences = processed_sequences.to(device)
                    output, hand_attentions, obj_attentions = model(processed_sequences)

                    _, predicted = torch.max(output, 1)
                    predictions.extend(predicted.cpu().tolist())
                    
                    if counter_test == 1 or counter_test == 100:
                        log_images_and_heatmaps(sequences, hand_attentions, obj_attentions, mode="test", epoch=epoch)
                    
                    counter_test += 1

            json_data = {str(i + 1): predictions[i] for i in range(len(predictions))}
            with open("attention_test_predictions_best.json", "w") as f:
                json.dump({"predictions": json_data}, f)
            print("Test predictions saved to attention_test_predictions_best.json")

            model_save_path = "attention_transformer_model_best.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
