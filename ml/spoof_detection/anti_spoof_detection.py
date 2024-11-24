import os
import sys
sys.path.append(os.getcwd())
import threading

import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from ml import model_paths
from app.utils.data_access import DataAccessImage

class SpoofClassifierModel(nn.Module):
    """Image Classifier based on pretrained EVA-02 model."""
    
    def __init__(self, num_classes=2, **kwargs):
        super(SpoofClassifierModel, self).__init__()
        # Load the EVA-02 model
        self.pretrained = timm.create_model(
            'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k',
            pretrained=False,
            num_classes=0  # Remove the classification head
        )
        
        # Freeze the pretrained parameters
        for param in self.pretrained.parameters():
            param.requires_grad = False
            
        # Get the feature dimension from the model
        self.feature_dim = self.pretrained.num_features
        
        # Create a new classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            # nn.Linear(self.feature_dim, num_classes),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Extract features using EVA-02
        features = self.pretrained(x)
        # Pass through our classifier
        output = self.classifier(features)
        return output

    def unfreeze_last_n_layers(self, n=2):
        """Unfreeze the last n transformer blocks for fine-tuning."""
        for name, param in self.pretrained.named_parameters():
            param.requires_grad = False  # First freeze everything
            
        # Unfreeze the last n blocks
        blocks_to_unfreeze = [f'blocks.{i}.' for i in range(-n, 0)]
        for name, param in self.pretrained.named_parameters():
            if any(block in name for block in blocks_to_unfreeze):
                param.requires_grad = True



class AntiSpoofCheck:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(AntiSpoofCheck, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, model_path: str):
        self.model = SpoofClassifierModel()
        self.model.load_state_dict(torch.load(model_path, map_location='cpu')["model_state_dict"])
        self.transformation = A.Compose([
            A.Resize(height=448, width=448),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def preprocess_image(self, data_access_image: DataAccessImage) -> torch.Tensor:
        return self.transformation(image=data_access_image.get_rgb_image())["image"].to('cpu')
    
    def check_spoof(self, data_access_image: DataAccessImage) -> bool:
        preprocessed_image = self.preprocess_image(data_access_image=data_access_image)
        preprocessed_image = preprocessed_image.unsqueeze(0)
        preprocessed_image.to('cpu')
        with torch.no_grad():
            output = self.model(preprocessed_image)
            output = output.cpu().detach().numpy()
        not_spoof_prob = round(output[0][1], 2)
        print(f"output: {output}\nnot_spoof_prob: {not_spoof_prob}")
        if not_spoof_prob > 0.6:
            return True
        return False


if __name__ == "__main__":
    model_path = model_paths.SPOOF_MODEL_PATH
    anti_spoof_check = AntiSpoofCheck(model_path=model_path)

    file_path = "/Users/divyanshnew/Pictures/ankesh_fake.jpg"
    file_path = '/Users/divyanshnew/Desktop/Screenshot 2024-11-24 at 3.27.00 PM.png'
    file_path = '/Users/divyanshnew/Desktop/Screenshot 2024-11-24 at 3.29.07 PM.png'
    file_path = '/Users/divyanshnew/Pictures/Photo on 21-12-22 at 11.22 AM.jpg'
    file_path = '/Users/divyanshnew/Pictures/eyes_closed.jpg'
    data_access_image = DataAccessImage(file_path=file_path)
    anti_spoof_check.check_spoof(data_access_image=data_access_image)