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
import justvit_utils
import json



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
seed = 7
justvit_utils.set_seed(seed)
batch_size = 1
wandb.init(project="base_transformer")
dataset_root = "/cluster/scratch/cbennewitz/"

def log_images_and_heatmaps(images, attentions, mode):
    for i in range(images.shape[1]):
        image = images[0, i, :, :, :].numpy().astype(np.uint8)
        attention = attentions[0, i, :, :].detach().cpu().numpy()
        plt.figure(figsize=(5, 5))
        plt.imshow(image)
        plt.axis("off")
        image_plt = wandb.Image(plt, caption=f"{mode} Input Image {i+1}")
        wandb.log({f"{mode}_input_image_{i+1}": image_plt})
        plt.close()

        plt.figure(figsize=(5,5))
        plt.imshow(attention, cmap="jet")
        plt.axis("off")
        attention_plt = wandb.Image(plt, caption=f"{mode} Attention MAP {i+1}")
        wandb.log({f"{mode}_Attention_MAP_{i+1}": attention_plt})
        plt.close()


def main():
    preprocessor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    model = justvit_utils.ViTMLPModel()
    model.to(device)

    train_data = justvit_utils.TrainData(dataset_root)
    val_data = justvit_utils.ValData(dataset_root)
    test_data = justvit_utils.TestData(dataset_root)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    num_epochs = 30
    current_best = np.inf
    

    for epoch in range(num_epochs):
        model.train()
        running_loss: float = 0.0
        counter_train: int = 0
        train_correct: int = 0
        train_total: int = 0
        for sequences, labels in tqdm(train_loader):
            optimizer.zero_grad()
            processed_sequences = justvit_utils.preprocess(preprocessor, sequences)
            processed_sequences = processed_sequences.to(device)
            labels = labels.to(device)
            output, attentions = model(processed_sequences)
            train_loss = criterion(output, labels)
            train_loss.backward()
            optimizer.step()
            running_loss += train_loss.item()
            _, predicted = torch.max(output, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            if counter_train == 100:
                log_images_and_heatmaps(sequences, attentions, mode="train")

            counter_train += 1

        avg_train_loss = running_loss / counter_train
        train_accuracy = train_correct / train_total
        wandb.log({"train_loss": avg_train_loss, "train_accuracy": train_accuracy, "epoch": epoch + 1})
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy*100:.2f}%")

        model.eval()
        val_running_loss: float = 0.0
        counter_val: int = 0
        val_correct: int = 0
        val_total: int = 0
        with torch.no_grad():
            for sequences, labels in tqdm(val_loader):
                processed_sequences = justvit_utils.preprocess(preprocessor, sequences)
                processed_sequences = processed_sequences.to(device)
                labels = labels.to(device)
                output, attentions = model(processed_sequences)
                val_loss = criterion(output, labels)
                val_running_loss += val_loss.item()
                _, predicted = torch.max(output, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                if counter_val == 25:
                    log_images_and_heatmaps(sequences, attentions, mode="val")
                
                counter_val += 1
        avg_val_loss = val_running_loss / counter_val
        val_accuracy = val_correct / val_total
        wandb.log({"val_loss": avg_val_loss, "val_accuracy": val_accuracy, "epoch": epoch + 1})
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss}, Train Accuracy: {train_accuracy*100:.2f}%, "
              f"Val Loss: {avg_val_loss}, Val Accuracy: {val_accuracy*100:.2f}%")

        if current_best > avg_val_loss:
            current_best = avg_val_loss
            counter_test = 0
            predictions = []
            model.eval()
            with torch.no_grad():
                for sequences in tqdm(test_loader):
                    processed_sequences = justvit_utils.preprocess(preprocessor,sequences)
                    processed_sequences = processed_sequences.to(device)
                    output, attentions = model(processed_sequences)

                    _, predicted = torch.max(output, 1)
                    predictions.extend(predicted.cpu().tolist())
                    
                    if counter_test == 1 or counter_test == 100:
                        log_images_and_heatmaps(sequences, attentions, mode="test")
                    
                    counter_test += 1

            json_data = {str(i + 1): predictions[i] for i in range(len(predictions))}
            with open("test_predictions_best.json", "w") as f:
                json.dump({"predictions": json_data}, f)
            print("Test predictions saved to test_predictions.json")

            model_save_path = "transformer_model_best.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
