import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from .luminance_flicker import LuminanceFlickerDetector
from .motion_detector import MotionDetector
from .gabor import GaborFilters
import sys, os 
from scipy.special import gammaln, psi  # Gamma functions needed for surprise calculation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utils.visualizer import Visualizer 


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        # Luminance filter just averages RGB channels
        self.intensity = nn.Conv2d(3, 1, kernel_size=(1, 1), bias=False) # luminance
        self.intensity.weight.data.fill_(1.0 / 3.0)

        # Color opponency filters are handcrafted
        self.color_opponency_filters = nn.Conv2d(3, 4, kernel_size=(1, 1), bias=False)
        self.init_color_opponency_filters()

        # Example Gabor filter initialization for oriented edge detection
        self.gabor_filters = GaborFilters()

        # Luminance flicker
        self.luminance_flicker = LuminanceFlickerDetector()

        # Motion Detection
        self.motion_detector = MotionDetector()


    def init_color_opponency_filters(self): # seperate these 2 to get 12 each
        # Red-Green and Blue-Yellow opponency
        # Red - Green
        self.color_opponency_filters.weight.data[0, 0, 0, 0] = 1.0  # Red
        self.color_opponency_filters.weight.data[0, 1, 0, 0] = -1.0 # Green
        # Blue - Yellow (Yellow is approximated as Red + Green)
        self.color_opponency_filters.weight.data[1, 2, 0, 0] = 1.0  # Blue
        self.color_opponency_filters.weight.data[1, 0, 0, 0] = -0.5 # Red
        self.color_opponency_filters.weight.data[1, 1, 0, 0] = -0.5 # Green

    def forward(self, current_frame, previous_frame=None):
        # Assume x is a batch of frames with dimension N,C,H,W [batch_size, channels, height, width]

        # Compute luminance
        current_intensity = self.intensity(current_frame)

        # Compute color opponency
        color_opponency = self.color_opponency_filters(current_frame)

        # Compute edges using the Gabor filters on the current frame's luminance
        edges = self.gabor_filters(current_intensity)
        # print("edges: ", edges.shape)

        # Initialize flicker and motion variables
        flicker = None
        motion = None

        # Compute luminance flicker and motion if previous_frame is provided
        if previous_frame is not None:
            previous_intensity = self.intensity(previous_frame)

            # Luminance flicker calculated as the difference between current and previous luminance
            flicker = self.luminance_flicker(current_intensity, previous_intensity)

            # Motion detection based on the current and previous frames
            motion = self.motion_detector(current_frame, previous_frame)
            # print("motion: ", motion.shape)
            # print("previous_intensity: ", previous_intensity.shape)

        # Return a dictionary of the computed features for better accessibility
        features = {
            "luminance": current_intensity,
            "color_opponency": color_opponency,
            "edges": edges,
            "flicker": flicker,
            "motion": motion
        }

        return features


class MultiScalePyramid(nn.Module):
    def __init__(self):
        super(MultiScalePyramid, self).__init__()
        self.base_extractor = FeatureExtractor()
        self.num_scales = 9  # Scales from 0 (original) to 8 (reduced by factor of 256)
        self.scale_pairs = [(2, 5), (2, 6), (3, 6), (3, 7), (4, 7), (4, 8)]

    def forward(self, current_frame, previous_frame=None):
        # Extract features at all scales
        scale_features = {}
        for scale in range(self.num_scales):
            scale_factor = 1 / (2 ** scale)
            scaled_current = F.interpolate(current_frame, scale_factor=scale_factor, mode='bilinear', align_corners=False)
            scaled_previous = F.interpolate(previous_frame, scale_factor=scale_factor, mode='bilinear', align_corners=False) if previous_frame is not None else None

            features = self.base_extractor(scaled_current, scaled_previous)
            scale_features[scale] = features

        # Compute across-scale differences and resample to the common size
        final_features = self.compute_differences_resample(scale_features)

        return final_features
    
    def compute_differences_resample(self, scale_features):
        differences = {key: [] for key in scale_features[0].keys()}  # Dictionary to store differences for each feature
        target_size = (40, 30)  # Resample size as specified

        for pair in self.scale_pairs:
            scale_a, scale_b = pair
            for key in scale_features[0].keys():
                feature_a = scale_features[scale_a][key]
                feature_b = scale_features[scale_b][key]
                
                if feature_a is None or feature_b is None:
                    continue  # Skip if feature is missing

                # Ensure feature_b is resized to the size of feature_a before subtraction
                feature_b_resized = F.interpolate(feature_b, size=feature_a.shape[2:], mode='bilinear', align_corners=False)
                diff = feature_a - feature_b_resized
                diff_resampled = F.interpolate(diff, size=target_size, mode='bilinear', align_corners=False)
                differences[key].append(diff_resampled)

        # Concatenate differences for each feature separately
        final_features = {key: torch.cat(diffs, dim=1) for key, diffs in differences.items()}
        return final_features 


if __name__ == "__main__":
    pyramid = MultiScalePyramid()
    current_frame = torch.randn(1, 3, 640, 480)  # Dummy data for the current frame
    previous_frame = torch.randn(1, 3, 640, 480)  # Dummy data for the previous frame

    # Process the input through the pyramid
    output_features = pyramid(current_frame, previous_frame)
    # print("Output features shape:", output_features.shape)
    
    # plot_feature_maps(output_features) 
    for feature_name, feature_maps in output_features.items():
        Visualizer().visualize_feature_maps(feature_maps, feature_name)
