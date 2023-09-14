# Importing Libraries
import torch, os
import numpy as np
import cv2
from torch import nn
import timm
from transformers import DistilBertModel, DistilBertConfig
from torch.utils.data import Dataset, DataLoader
from tqdm.autonotebook import tqdm
import pickle
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# Trained Model Configurations
CFG = {
    "debug": False,
    "captions_path": ".",
    "batch_size": 64,
    "num_workers": 4,
    "head_lr": 1e-3,
    "image_encoder_lr": 1e-4,
    "text_encoder_lr": 1e-5,
    "weight_decay": 1e-3,
    "patience": 1,
    "factor": 0.8,
    "epochs": 12,
    "device": "cpu",
    "model_name": 'resnet50',
    "image_embedding": 2048,
    "text_encoder_model": "distilbert-base-uncased",
    "text_embedding": 768,
    "text_tokenizer": "distilbert-base-uncased",
    "max_length": 200,
    "pretrained": True,
    "trainable": True,
    "temperature": 1.0,
    "size": 224,
    "num_projection_layers": 1,
    "projection_dim": 256,
    "dropout": 0.1
}


# Loading Finetuned Clip Model to the below class format
class CLIPModel(nn.Module):
    def __init__(
            self,
            temperature=CFG["temperature"],
            image_embedding=CFG["image_embedding"],
            text_embedding=CFG["text_embedding"],
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature


# Image Encoder Class to extract features using finetuned clip's Resnet Image Encoder
class ImageEncoder(nn.Module):
    def __init__(self, model_name=CFG["model_name"], pretrained=CFG["pretrained"], trainable=CFG["trainable"]):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained, num_classes=0, global_pool="avg")
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


# Text Encoder - Optional in inference
class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG["text_encoder_model"], pretrained=CFG["pretrained"],
                 trainable=CFG["trainable"]):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())

        for p in self.model.parameters():
            p.requires_grad = trainable

        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


# Projection Class - Optional in inference
class ProjectionHead(nn.Module):
    def __init__(
            self,
            embedding_dim,
            projection_dim=CFG["projection_dim"],
            dropout=CFG["dropout"]
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


# Class to transform image to custom data format
class SkyImage(Dataset):
    def __init__(self, img, label):
        self.img = img
        self.img_label = label

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        image = cv2.resize(self.img[idx], (244, 244))
        # image = cv2.cvtColor(self.img[idx], cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, (244, 244))
        image = np.moveaxis(image, -1, 0)
        label = self.img_label[idx]
        return image, label


# Method to extract features from finetuned clip model
def get_features(clip_model, dataset):
    features, label, embeddings = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=64)):
            image_input = torch.tensor(np.stack(images)).cpu().float()
            image_features = clip_model.image_encoder(image_input)
            features.append(image_features)
            label.append(labels)
    return torch.cat(features), torch.cat(label).cpu()


# Loading Clip and Catboost models
CTBR_model = pickle.load(open("catboost_model.sav", 'rb'))
clip_model = CLIPModel().to(CFG["device"])
clip_model.load_state_dict(torch.load("clip_model.pt", map_location=CFG["device"]))
clip_model.eval()


# Method to calculate cloud coverage
def predict_cloud_coverage(image):
    img, lbl = [image], [0]
    # Transforming Data into custom format
    test_image = SkyImage(img, lbl)
    # Extracting Features from Finetuned CLIP model
    features, label = get_features(clip_model, test_image)
    # Predicting Cloud Coverage based on extracted features
    pred_cloud_coverage = CTBR_model.predict(features.cpu().numpy())
    return(round(max(0.0, min(100.0, pred_cloud_coverage[0])), 1))