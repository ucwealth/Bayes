import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
import os
import numpy as np
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
    

class SaliencyPredictor(nn.Module):
    def __init__(self, backbone='vgg16'):
        super(SaliencyPredictor, self).__init__()
        self.feature_extractor = FeatureExtractor(backbone=backbone)
    
    def forward(self, current_frame, previous_frame):
        # Extract features for both current and previous frames
        current_features = self.feature_extractor(current_frame)
        previous_features = self.feature_extractor(previous_frame)
        
        # Calculate Bayesian surprise
        surprise_map = self.calculate_bayesian_surprise(previous_features, current_features)
        
        # Apply softmax to get probabilities
        saliency_map = F.softmax(surprise_map.view(surprise_map.size(0), -1), dim=1).view_as(surprise_map)
        
        return saliency_map
    
    
    def calculate_bayesian_surprise(self, prior_features, posterior_features):
        # Flatten the feature maps along the spatial dimensions but keep the batch and channel dimensions
        prior_flat = prior_features.view(prior_features.size(0), prior_features.size(1), -1)
        posterior_flat = posterior_features.view(posterior_features.size(0), prior_features.size(1), -1)

        # Apply softmax to prior_flat to get probabilities and add epsilon to avoid zeros
        prior_probs = torch.clamp(F.softmax(prior_flat, dim=2), min=1e-10, max=1.0)
        
        # Ensure no NaN values after softmax
        if torch.isnan(prior_probs).any():
            print("NaN detected in prior_probs after softmax")
        
        # Add a small epsilon to posterior_flat to avoid log(0)
        posterior_flat = torch.clamp(posterior_flat, min=1e-10, max=1.0)
        posterior_log = posterior_flat.log()

        # Calculate KL divergence for each spatial position
        kl_divergence = F.kl_div(posterior_log, prior_probs, reduction='none')
        
        # Ensure no NaN values in KL divergence
        if torch.isnan(kl_divergence).any():
            print("NaN detected in kl_divergence")

        # Sum over the channel dimension to get a per-pixel KL divergence
        kl_divergence = kl_divergence.sum(dim=1)
        
        # Reshape back to the original spatial dimensions
        surprise_map = kl_divergence.view(prior_features.size(0), prior_features.size(2), prior_features.size(3))

        return surprise_map


# Function to apply temporal smoothing
def smooth_saliency_map(current_saliency, previous_saliency, alpha=0.7):
    return alpha * current_saliency + (1 - alpha) * previous_saliency


def process_video_with_temporal_smoothing(video_path, model, device, output_video_path):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    ret, previous_frame = cap.read()
    
    if not ret:
        raise ValueError("Couldn't open the video")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Preprocess the first frame
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    previous_frame = preprocess(previous_frame).unsqueeze(0).to(device)
    
    # Initialize the first saliency map
    previous_saliency = None
    frame_count = 0

    # Process each frame of the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess current frame
        current_frame = preprocess(frame).unsqueeze(0).to(device)
        
        # Predict saliency for the current frame
        with torch.no_grad():
            current_saliency = model(current_frame, previous_frame).squeeze().cpu().numpy()

        # Resize the saliency map to match the frame size
        current_saliency = cv2.resize(current_saliency, (frame.shape[1], frame.shape[0]))

        # If this is the first frame, initialize previous_saliency
        if previous_saliency is None:
            previous_saliency = current_saliency

        # Apply temporal smoothing
        smoothed_saliency = 0.7 * current_saliency + 0.3 * previous_saliency

        # Check for NaN or Inf values
        if np.isnan(smoothed_saliency).any() or np.isinf(smoothed_saliency).any():
            # print(f"Warning: NaN or Inf values encountered in saliency map at frame {frame_count}.")
            smoothed_saliency = np.nan_to_num(smoothed_saliency, nan=0.0, posinf=1.0, neginf=0.0)

        # Clamp the values to ensure they are in the range [0, 1]
        smoothed_saliency = np.clip(smoothed_saliency, 0, 1)

        # Normalize smoothed saliency map to range [0, 255] for visualization
        smoothed_saliency = (smoothed_saliency * 255).astype(np.uint8)

        # Apply color map to convert grayscale saliency to color
        smoothed_saliency_color = cv2.applyColorMap(smoothed_saliency, cv2.COLORMAP_JET)
        
        # Ensure the frame is in the correct format (BGR)
        if len(frame.shape) == 2 or frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Overlay the saliency map on the original frame
        overlay = cv2.addWeighted(frame, 0.5, smoothed_saliency_color, 0.5, 0)
        
        # Write the frame to the output video
        out.write(overlay)
        
        # Update for the next iteration
        previous_saliency = current_saliency
        previous_frame = current_frame
        frame_count += 1
    
    # Release the video objects
    cap.release()
    out.release()



# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SaliencyPredictor(backbone='vgg16').to(device)

    video_path = "../videos/vid6.mp4"  # Path to your input video file
    output_video_path = "../output/saliency_output.mp4"  # Path to save the output video

    process_video_with_temporal_smoothing(video_path, model, device, output_video_path)
    print(f"Saliency video saved at {output_video_path}")
