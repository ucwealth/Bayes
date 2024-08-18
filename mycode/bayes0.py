import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from scipy.stats import entropy
import cv2
import matplotlib.pyplot as plt


class FeatureExtractor(nn.Module):
    def __init__(self, backbone='vgg16'):
        super(FeatureExtractor, self).__init__()
        if backbone == 'vgg16':
            self.model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        elif backbone == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.model = nn.Sequential(*list(resnet.children())[:-2])  # Remove last layers to get feature maps
        else:
            raise NotImplementedError("Only VGG16 and ResNet50 are supported")

    def forward(self, x):
        return self.model(x)
    
    
def calculate_bayesian_surprise(prior_features, posterior_features):
    # Flatten the feature maps along the spatial dimensions but keep the batch and channel dimensions
    prior_flat = prior_features.view(prior_features.size(0), prior_features.size(1), -1)
    posterior_flat = posterior_features.view(posterior_features.size(0), posterior_features.size(1), -1)
    print("posterior_flat: ", posterior_flat)
    print("posterior_flat log: ", posterior_flat.log())
    
    # Apply softmax to prior_flat to get probabilities and add epsilon to avoid log(0)
    prior_probs = torch.clamp(F.softmax(prior_flat, dim=2), min=1e-10, max=1.0)
    print("prior_probs: ", prior_probs)
    # prior_probs = F.softmax(prior_flat, dim=2)
    
    # Calculate KL divergence for each spatial position
    # kl_divergence = F.kl_div(posterior_flat.log(), prior_flat, reduction='none')
    kl_divergence = F.kl_div(posterior_flat.log(), prior_probs, reduction='none')
    
    # Sum over the channel dimension to get a per-pixel KL divergence
    kl_divergence = kl_divergence.sum(dim=1)
    
    # Reshape back to the original spatial dimensions
    surprise_map = kl_divergence.view(prior_features.size(0), prior_features.size(2), prior_features.size(3))
    print("surprise_map: ", surprise_map)
    return surprise_map



class SaliencyPredictor(nn.Module):
    def __init__(self, backbone='vgg16'):
        super(SaliencyPredictor, self).__init__()
        self.feature_extractor = FeatureExtractor(backbone=backbone)
    
    def forward(self, current_frame, previous_frame):
        # Extract features for both current and previous frames
        current_features = self.feature_extractor(current_frame)
        previous_features = self.feature_extractor(previous_frame)
        print("self.feature_extractor(current_frame)", self.feature_extractor(current_frame))
        
        # Calculate Bayesian surprise
        surprise_map = calculate_bayesian_surprise(previous_features, current_features)
        
        # Apply softmax over the spatial dimensions
        saliency_map = F.softmax(surprise_map.view(surprise_map.size(0), -1), dim=1).view_as(surprise_map)
        print(saliency_map.shape)
        return saliency_map


def process_video(video_path, model, device):
    cap = cv2.VideoCapture(video_path)
    ret, previous_frame = cap.read()
    
    if not ret:
        raise ValueError("Couldn't open the video")

    # Convert previous frame to tensor and normalize
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    previous_frame = preprocess(previous_frame).unsqueeze(0).to(device)
    
    saliency_maps = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to tensor and normalize
        current_frame = preprocess(frame).unsqueeze(0).to(device)
        print("current_frame: ", current_frame.shape)
        
        # Predict saliency
        with torch.no_grad():
            saliency_map = model(current_frame, previous_frame)
            print("saliency_map", saliency_map)
        saliency_maps.append(saliency_map.squeeze().cpu().numpy())
        
        # Update previous frame
        previous_frame = current_frame
    
    cap.release()
    return saliency_maps


def visualize_saliency_maps(saliency_maps):
    for i, saliency_map in enumerate(saliency_maps):
        plt.figure()
        plt.imshow(saliency_map, cmap='hot')
        plt.title(f'Saliency Map Frame {i+1}')
        plt.axis('off')
        plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SaliencyPredictor(backbone='vgg16').to(device)

video_path = "../videos/vid6.mp4"
saliency_maps = process_video(video_path, model, device)

# Example: display or save the first saliency map
plt.imshow(saliency_maps[14], cmap='hot')
plt.show()

# Visualize all saliency maps
# visualize_saliency_maps(saliency_maps)