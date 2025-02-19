import random
import torch
import torch.nn as nn
import numpy as np
import glob
from torch.utils.data import Dataset
import os
from natsort import natsorted
from transformers import ViTModel, ViTImageProcessor
import torch.nn.functional as F

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MultiModalDataset(Dataset):
    def __init__(self, data_root, subjects, transform=None):
        """
        Args:
            data_root (str): Root directory containing subject folders.
            subjects (list): List of subject folder names (e.g., ['S01', 'S02']).
            transform (callable, optional): Optional transform to apply on each modality.
        """
        self.samples = []
        self.transform = transform

        for subject in subjects:
            subject_dir = os.path.join(data_root, subject)
            if not os.path.isdir(subject_dir):
                raise ValueError(f"Subject directory {subject_dir} does not exist.")

            # Retrieve file paths for each modality within the subject folder.
            image_paths = natsorted(glob.glob(os.path.join(subject_dir, "images", "*.npy")))
            hand_paths = natsorted(glob.glob(os.path.join(subject_dir, "hand_heatmaps", "*.npy")))
            object_paths = natsorted(glob.glob(os.path.join(subject_dir, "object_heatmaps", "*.npy")))
            subject_labels = natsorted(glob.glob(os.path.join(subject_dir, "action_labels", "*.npy")))
            
            # Load action labels (assumed to be stored in a single .npy file)
            
            
            
            # Verify that all modalities have the same number of samples.
            if not (len(image_paths) == len(hand_paths) == len(object_paths) == len(subject_labels)):
                raise ValueError(f"Mismatch in number of files for subject: {subject}")

            # Append a tuple for each sample.
            for i in range(len(image_paths)):
                self.samples.append((image_paths[i], hand_paths[i], object_paths[i], subject_labels[i]))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, hand_path, obj_path, label = self.samples[idx]
        # Load each modality assuming the data is stored as npy files.
        image = np.load(img_path).astype(np.float32)
        hand_heatmap = np.load(hand_path).astype(np.float32)
        object_heatmap = np.load(obj_path).astype(np.float32)
        label = np.load(label).astype(np.int64)

        
        # Convert numpy arrays to PyTorch tensors.
        image = torch.from_numpy(image)
        hand_heatmap = torch.from_numpy(hand_heatmap)
        object_heatmap = torch.from_numpy(object_heatmap)
        label = torch.from_numpy(label)
        
        # Optionally apply a transform (could be applied individually per modality if needed)
        if self.transform:
            image = self.transform(image)
            hand_heatmap = self.transform(hand_heatmap)
            object_heatmap = self.transform(object_heatmap)
        
        return image, hand_heatmap, object_heatmap, label

# Example usage:






def preprocess(preprocessor, sequences):
    processed_sequences = []
    for i in range(sequences.shape[0]):
        sequence = sequences[i]
        processed_sequence = preprocessor(images = sequence, return_tensors='pt', size=496)
        processed_sequences.append(processed_sequence['pixel_values'])
    processed_sequences = torch.stack(processed_sequences)
    return processed_sequences


class Cars_Action(nn.Module):
    def __init__(self):
        super(Cars_Action, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.vit.config.image_size = 496
        vit_feature_dim = self.vit.config.hidden_size
        self.mlp_input_dim = vit_feature_dim * 8
        

        self.mlp = nn.Sequential(
            nn.Linear(self.mlp_input_dim, 3000),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(3000, 1000),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1000, 300),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(300, 37)
        )
        

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        sequence_length = self.sequence_length

        # Forward pass through ViT
        outputs = self.vit_model(pixel_values, output_attentions=True, interpolate_pos_encoding=True)
        last_hidden_state = outputs.last_hidden_state[:, 0, :]  # (batch_size * sequence_length, hidden_size)
        attentions = outputs.attentions[-1]  # Get the attention maps from the last layer
        # Process attentions to match the input image resolution
        num_heads = attentions.shape[1]
        num_tokens = attentions.shape[-1] - 1
        attentions = attentions[:, :, 0, 1:].reshape(batch_size, num_heads, num_tokens)

        w_featmap = pixel_values.shape[-2] // self.vit_model.config.patch_size
        h_featmap = pixel_values.shape[-1] // self.vit_model.config.patch_size
        attentions = attentions.reshape(batch_size, num_heads, w_featmap, h_featmap)
        attentions = F.interpolate(attentions, scale_factor=self.vit_model.config.patch_size, mode="nearest")
        attentions = attentions.view(batch_size, num_heads, pixel_values.shape[-2], pixel_values.shape[-1])
        attentions = (attentions - attentions.min()) / (attentions.max() - attentions.min())
        hand_attention = attentions[:,0:6,:,:]
        obj_attention = attentions[:,6:12,:,:]
        #free_attention = attentions[:,8:12,:,:]
        # sum_hand = torch.mean(hand_attention, dim=1)
        # sum_obj = torch.mean(obj_attention, dim=1)
        mean_hand = torch.mean(hand_attention, dim=1)
        mean_obj = torch.mean(obj_attention, dim=1)
        #mean_free =torch.mean(free_attention, dim=1)
        # Reshape and concatenate embeddings

        concatenated_embeddings = last_hidden_state.reshape(batch_size // sequence_length, sequence_length * self.vit_model.config.hidden_size)
        # Pass through the classifier
        logits = self.classifier(concatenated_embeddings)

        return logits, mean_hand,mean_obj#, mean_free



class ViTMLPModel(nn.Module):
    def __init__(self):
        super(ViTMLPModel, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.vit.config.image_size = 496
        vit_feature_dim = self.vit.config.hidden_size
        self.mlp_input_dim = vit_feature_dim * 8
        
        self.mlp = nn.Sequential(
            nn.Linear(self.mlp_input_dim, 3000),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(3000, 1000),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1000, 300),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(300, 37)
        )
    
    def forward(self, sequences):
        
        features = []
        attentions = []
        for i in range(sequences.shape[0]):
            sequence = sequences[i]
            outputs = self.vit(sequence, output_attentions=True, interpolate_pos_encoding=True)
            image_features = outputs['pooler_output']
            attentions_out = outputs['attentions']
            last_layer_attentions = attentions_out[-1]
            cls_attention_all_heads = last_layer_attentions.mean(dim=1)
            cls_to_patches_attention = cls_attention_all_heads[:, 0, 1:]
            grid_size = int(cls_to_patches_attention.shape[1] ** 0.5)
            cls_attention_2d_batch = cls_to_patches_attention.reshape(-1, grid_size, grid_size)

            features.append(image_features)
            attentions.append(cls_attention_2d_batch)
        features = torch.stack(features)
        concatenated_features = features.view(features.shape[0], 8*768)
        attentions = torch.stack(attentions)
        output = self.mlp(concatenated_features)

        return output, attentions
        

