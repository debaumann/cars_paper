import random
import torch
import torch.nn as nn
import numpy as np
import glob
from natsort import natsorted
from transformers import ViTModel

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TrainData(torch.utils.data.DataLoader):
    def __init__(self, data_root):
        self.img_paths = data_root + "sequences_train/*npy"
        self.img_paths = natsorted(glob.glob(self.img_paths))
        self.labels = np.load(data_root + "action_train.npy")

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = np.load(self.img_paths[idx]).astype(np.float32)
        img_square = torch.from_numpy(img).float()
        label = int(self.labels[idx])
        return img_square, label

class ValData(torch.utils.data.DataLoader):
    def __init__(self, data_root):
        self.img_paths = data_root + "sequences_val/*npy"
        self.img_paths = natsorted(glob.glob(self.img_paths))
        self.labels = np.load(data_root + "action_val.npy")
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = np.load(self.img_paths[idx]).astype(np.float32)
        img_square = torch.from_numpy(img).float()
        label = int(self.labels[idx])
        return img_square, label
    
class TestData(torch.utils.data.DataLoader):
    def __init__(self, data_root):
        self.img_paths = data_root + "sequences_test/*npy"
        self.img_paths = natsorted(glob.glob(self.img_paths))
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = np.load(self.img_paths[idx]).astype(np.float32)
        img_square = torch.from_numpy(img).float()
        return img_square

def preprocess(preprocessor, sequences):
    processed_sequences = []
    for i in range(sequences.shape[0]):
        sequence = sequences[i]
        processed_sequence = preprocessor(images = sequence, return_tensors='pt', size=496)
        processed_sequences.append(processed_sequence['pixel_values'])
    processed_sequences = torch.stack(processed_sequences)
    return processed_sequences

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
        

