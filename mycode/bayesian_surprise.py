"""
To Do:

- A tool that researchers can install on their laptop, run and use
- Use different models with higher brain score to see what happens
- Change how the attention stuff works 
- Original paper uses human eye tracking data to change where the model is sampling data from, from the input
- This was hacky because in the original video, they dont have human eye tracking to do that with
- However, there are famous models such as the famous Idy/Ichy and Boldy 2009 model of images in vision science literature. 
- Its used for: If you show someone an image, what is the salient content of that image.
- Since I'm using videos, I would have to project this through time, and average it out
- This produces the kind of data that they were using the human eye tracking data for. 
- E.g: For a given image, this is where people would look. For the next image, this is where people would look. I would then need to have 
some temporal component of that, 
- The attention model would be a statistical heatmap because technical, a human's eye doesnt just jump to a particular place/location on an image. 
Therefore: the model would show that on an image, the probability of looking somewhere is 10% here, 30% here, 80% there. This would be smeared through time
to do some kind of averaging between frames. So for each frame, you have an estimate which is biased by the previous frame estimate.
- The follow up to Time without clocks paper has a version of this attention experiment amongst other demos.
- Integrate this model into the basic time without clock model, using it to replace human eye tracking.
- An experiment I can do for my report is to build various versions of this and compare them. I have the original human data for the reports of duration, so 
then I can compare which one gets closer to the humans.

"""



import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from scipy.stats import entropy
import cv2

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
    
    # Calculate KL divergence for each spatial position
    kl_divergence = F.kl_div(posterior_flat.log(), prior_flat, reduction='none')
    
    # Sum over the channel dimension to get a per-pixel KL divergence
    kl_divergence = kl_divergence.sum(dim=1)
    
    # Reshape back to the original spatial dimensions
    surprise_map = kl_divergence.view(prior_features.size(0), prior_features.size(2), prior_features.size(3))

    print(prior_features.shape)
    print(posterior_features.shape)
    print(kl_divergence.shape)

    
    return surprise_map



class SaliencyPredictor(nn.Module):
    def __init__(self, backbone='vgg16'):
        super(SaliencyPredictor, self).__init__()
        self.feature_extractor = FeatureExtractor(backbone=backbone)
    
    def forward(self, current_frame, previous_frame):
        # Extract features for both current and previous frames
        current_features = self.feature_extractor(current_frame)
        previous_features = self.feature_extractor(previous_frame)
        
        # Calculate Bayesian surprise
        surprise_map = calculate_bayesian_surprise(previous_features, current_features)
        
        # Apply some post-processing (e.g., softmax to get probabilities)
        saliency_map = F.softmax(surprise_map, dim=1)
        
        return saliency_map


def process_video(video_path, model, device):
    cap = cv2.VideoCapture(video_path)
    ret, previous_frame = cap.read()
    
    if not ret:
        raise ValueError("Couldn't open the video")

    # Convert previous frame to tensor
    previous_frame = torch.tensor(previous_frame).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    saliency_maps = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to tensor
        current_frame = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        # Predict saliency
        with torch.no_grad():
            saliency_map = model(current_frame, previous_frame)
        
        saliency_maps.append(saliency_map.squeeze().cpu().numpy())
        
        # Update previous frame
        previous_frame = current_frame
    
    cap.release()
    return saliency_maps


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SaliencyPredictor(backbone='vgg16').to(device)

video_path = "../videos/vid0.mp4"
saliency_maps = process_video(video_path, model, device)

# Example: display or save the first saliency map
import matplotlib.pyplot as plt
plt.imshow(saliency_maps[0], cmap='hot')
plt.show()
