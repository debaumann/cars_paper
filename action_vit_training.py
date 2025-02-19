from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from seq_loader import ValData, TrainData
import os


# Define the feature extractor function for images 
def preprocess(images):
    # Apply the feature extractor transformations to the images
    encoding = feature_extractor(images=images, return_tensors="pt")
    pixel_values = encoding.pixel_values
    
    return pixel_values




# Define the action prediction model with the classifier head
class ViTActionPredictionModel(nn.Module):
    def __init__(self, vit_model, num_classes, sequence_length):
        super(ViTActionPredictionModel, self).__init__()
        self.vit_model = vit_model
        self.sequence_length = sequence_length
        self.classifier = nn.Sequential(
            nn.Linear(vit_model.config.hidden_size * sequence_length, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0] // self.sequence_length
        pixel_values = pixel_values.view(batch_size * self.sequence_length, *pixel_values.shape[2:])
        
        # Forward pass through ViT
        outputs = self.vit_model(pixel_values, output_attentions=True)
        last_hidden_state = outputs.last_hidden_state[:, 0, :]  # (batch_size * sequence_length, hidden_size)
        attentions = outputs.attentions[-1]  # Get the attention maps from the last layer

        # Reshape and concatenate embeddings
        concatenated_embeddings = last_hidden_state.view(batch_size, self.sequence_length * self.vit_model.config.hidden_size)
        
        # Pass through the classifier
        logits = self.classifier(concatenated_embeddings)
        
        return logits, attentions

# Initialize the feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224', size=640)

# Load the fine-tuned ViT model
vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224', add_pooling_layer=False)
if os.path.exists("fine_tuned_vit_model.pth"):  
    vit_model.load_state_dict(torch.load("fine_tuned_vit_model.pth"))

# Define the action prediction model
num_classes = 10  # Change this to the number of action classes in your dataset
sequence_length = 8
model = ViTActionPredictionModel(vit_model, num_classes, sequence_length)

# Define the dataset and dataloader for training and validation
train_dataset = TrainData()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataset = ValData()
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Custom attention loss function for validation
def custom_attention_loss(attentions, heatmaps):
    resized_attentions = nn.functional.interpolate(attentions, size=heatmaps.shape[-2:], mode='nearest')
    loss = F.mse_loss(resized_attentions, heatmaps)
    return loss

# Training and validation loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for batch in train_loader:
        img, hand_poses, label, obj_poses, heatmaps = batch
        images = img.view(-1, 3, 640, 640)  # Reshape to (batch_size * sequence_length, 3, H, W)
        
        # Define the crop size and preprocess images
        crop_size = 360
        start = (images.shape[-1] - crop_size) // 2
        cropped_images = images[:, :, start:start+crop_size, start:start+crop_size]
        cropped_images = nn.functional.interpolate(cropped_images, size=(640, 640), mode='nearest')
        
        pixel_values = feature_extractor(images=cropped_images, return_tensors="pt").pixel_values
        
        # Forward pass
        logits, _ = model(pixel_values)
        loss = criterion(logits, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Print training loss for the epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

    # Validation loop
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    total_val_loss = 0.0
    total_attention_mse = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            img, hand_poses, label, obj_poses, heatmaps = batch
            images = img.view(-1, 3, 640, 640)  # Reshape to (batch_size * sequence_length, 3, H, W)
            
            # Define the crop size and preprocess images
            crop_size = 360
            start = (images.shape[-1] - crop_size) // 2
            cropped_images = images[:, :, start:start+crop_size, start:start+crop_size]
            cropped_images = nn.functional.interpolate(cropped_images, size=(640, 640), mode='nearest')
            
            pixel_values = feature_extractor(images=cropped_images, return_tensors="pt").pixel_values
            
            # Forward pass
            logits, attentions = model(pixel_values)
            val_loss = criterion(logits, label)
            total_val_loss += val_loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            total_predictions += label.size(0)
            correct_predictions += (predicted == label).sum().item()
            
            # Calculate attention MSE
            attentions = attentions.mean(dim=1).squeeze(1)  # Average over heads and remove singleton dimension
            resized_heatmaps = nn.functional.interpolate(heatmaps.unsqueeze(1), size=(640, 640), mode='nearest').squeeze(1)
            attention_mse = custom_attention_loss(attentions, resized_heatmaps)
            total_attention_mse += attention_mse.item()
    
    # Print validation metrics
    accuracy = correct_predictions / total_predictions
    avg_val_loss = total_val_loss / len(val_loader)
    avg_attention_mse = total_attention_mse / len(val_loader)
    print(f"Validation - Epoch [{epoch + 1}/{num_epochs}], Accuracy: {accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Attention MSE: {avg_attention_mse:.4f}")

# Save the trained action prediction model
torch.save(model.state_dict(), "action_prediction_model.pth")


    
   

# Define the forward pass function to get attention maps
def forward(pixel_values):
    outputs = model(pixel_values, output_attentions=True, interpolate_pos_encoding=True)

    attentions = outputs.attentions[-1] # we are only interested in the attention maps of the last layer
    nh = attentions.shape[1] # number of head
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    w_featmap = pixel_values.shape[-2] // model.config.patch_size
    h_featmap = pixel_values.shape[-1] // model.config.patch_size
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=model.config.patch_size, mode="nearest")[0].cpu()
    print(attentions.shape)
    mean_attention = torch.mean(attentions, dim=0)

    return mean_attention

# Define the custom attention loss function
def custom_attention_loss(attentions, heatmaps):
    print(attentions.max(), attentions.min())

    print(heatmaps.max(), heatmaps.min())
    assert attentions.shape == heatmaps.shape, "Attention maps and heatmaps must have the same shape."
    loss = F.mse_loss(attentions, heatmaps)
    return loss
# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for batch in train_loader:
        img, hand_poses, label, obj_poses,heatmaps = batch
        images = img[:,:, 7, :, :]
        

        # Define the crop size
        crop_size = 360

        # Calculate the starting point for cropping
        start = (images.shape[-1] - crop_size) // 2

        # Crop the images
        cropped_images = images[:, :, :, start:start+crop_size]
        crop_size = 720
        start = (heatmaps.shape[-1] - crop_size) // 2
        heatmaps =  heatmaps[:,7, :, :]
        cropped_heatmaps = heatmaps[:, :, start:start+crop_size]
        cropped_heatmaps = cropped_heatmaps
        resized_heatmaps = nn.functional.interpolate(cropped_heatmaps.unsqueeze(1), size=(640, 640), mode='nearest').squeeze(1)
        # Convert to float
        resized_heatmaps = resized_heatmaps.float()

        # Normalize to [0, 1]
        resized_heatmaps = (resized_heatmaps - resized_heatmaps.min()) / (resized_heatmaps.max() - resized_heatmaps.min())
        
        pixel_values = preprocess_with_heatmap(cropped_images, resized_heatmaps)
        
        # Get attention maps
        attention= forward(pixel_values)
        attention = attention.float()
        attention = (attention - attention.min()) / (attention.max() - attention.min())
        loss = custom_attention_loss(attention, resized_heatmaps.squeeze(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print loss for the epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

# Save the fine-tuned model
torch.save(model.state_dict(), "fine_tuned_vit_model.pth")
