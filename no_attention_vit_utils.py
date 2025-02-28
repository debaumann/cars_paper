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

def soft_iou(pred, gt, eps=1e-6):
    
    intersection = torch.min(pred, gt).sum(dim=(1,2,3))
    union = torch.max(pred, gt).sum(dim=(1,2,3))
    siou = torch.mean((intersection + eps) / (union + eps))
    return siou.item()

def top_5(logits, gt):
    _, top_5_pred = torch.topk(logits, 5, dim=1)
    gt = gt.view(-1, 1)
    correct = torch.sum(top_5_pred.eq(gt).sum(dim=1)).item()
    return correct

def top_5_labels(logits):
    _, top_5_pred = torch.topk(logits, 5, dim=1)
    return top_5_pred

class TrainData(torch.utils.data.DataLoader):
    def __init__(self, data_root):
        self.img_paths = data_root + "sequences_train/*npy"
        self.img_paths = natsorted(glob.glob(self.img_paths))
        self.hand_paths = data_root + "hand_masks_train_resized/*npy"
        self.hand_paths = natsorted(glob.glob(self.hand_paths))
        self.obj_paths = data_root + "obj_masks_train_resized/*npy"
        self.obj_paths = natsorted(glob.glob(self.obj_paths))
        
        self.labels = np.load(data_root + "action_train.npy")

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = np.load(self.img_paths[idx]).astype(np.float32)
        img_square = torch.from_numpy(img).float()
        hand_mask = np.load(self.hand_paths[idx]).astype(np.float32) / 255.0
        hand_mask = torch.from_numpy(hand_mask).float()
        obj_mask = np.load(self.obj_paths[idx]).astype(np.float32) / 255.0
        obj_mask = torch.from_numpy(obj_mask).float()
        label = int(self.labels[idx])
        return img_square, hand_mask, obj_mask, label

class ValData(torch.utils.data.DataLoader):
    def __init__(self, data_root):
        self.img_paths = data_root + "sequences_val/*npy"
        self.img_paths = natsorted(glob.glob(self.img_paths))
        self.hand_paths = data_root + "hand_masks_val_resized/*npy"
        self.hand_paths = natsorted(glob.glob(self.hand_paths))
        self.obj_paths = data_root + "obj_masks_val_resized/*npy"
        self.obj_paths = natsorted(glob.glob(self.obj_paths))

        self.labels = np.load(data_root + "action_val.npy")
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = np.load(self.img_paths[idx]).astype(np.float32)
        img_square = torch.from_numpy(img).float()
        hand_mask = np.load(self.hand_paths[idx]).astype(np.float32) / 255.0
        hand_mask = torch.from_numpy(hand_mask).float()
        obj_mask = np.load(self.obj_paths[idx]).astype(np.float32) / 255.0
        obj_mask = torch.from_numpy(obj_mask).float() 
        label = int(self.labels[idx])
        return img_square, hand_mask, obj_mask, label
    
class TestData(torch.utils.data.DataLoader):
    def __init__(self, data_root):
        self.img_paths = data_root + "sequences_test/*npy"
        self.img_paths = natsorted(glob.glob(self.img_paths))

        self.labels = np.load(data_root + "action_test.npy")
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = np.load(self.img_paths[idx]).astype(np.float32)
        img_square = torch.from_numpy(img).float()
        label = int(self.labels[idx])
        return img_square, label

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
        hand_attentions = []
        obj_attentions = []

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

            num_heads = last_layer_attentions.shape[1]
            hand_attention = last_layer_attentions[:, 0:6, 0, 1:]
            obj_attention = last_layer_attentions[:, 6:12, 0, 1:]

            hand_attention = hand_attention.reshape(-1, 6, grid_size, grid_size)
            obj_attention = obj_attention.reshape(-1, 6, grid_size, grid_size)

            mean_hand = torch.mean(hand_attention, dim=1)
            mean_obj = torch.mean(obj_attention, dim=1)

            features.append(image_features)
            hand_attentions.append(mean_hand)
            obj_attentions.append(mean_obj)

        features = torch.stack(features)
        concatenated_features = features.view(features.shape[0], 8*768)
        hand_attentions = torch.stack(hand_attentions)
        obj_attentions = torch.stack(obj_attentions)

        output = self.mlp(concatenated_features)

        return output, hand_attentions, obj_attentions
        

