import cv2
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import os 
from original_time_model import AlexNetModel, FeatureAccumulator, get_transform, plot_durations, predict_time_for_video, train_time_estimator
import numpy as np
from DeepGaze import deepgaze_pytorch

"""

- THIS MODEL USES A PRETRAINED SALIENCY MODEL(DEEPGAZE) INSTEAD OF THE BAYESIAN SUPRISE MODEL 
- THE GOAL IS TO COMPARE THE RESULTS WITH THAT OF THE BAYESIAN SUPRISE MODEL AND SEE WHICH PERFORMS BETTER

"""
class SaliencyModel:
    def __init__(self):
        DEVICE = 'cuda'
        self.model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(DEVICE)
        self.model.eval()

    def compute_saliency(self, frame):
        # Assume the frame is already a tensor with shape (C, H, W)
        with torch.no_grad():
            # Forward pass through the model
            saliency_map = self.model(frame.unsqueeze(0))['out'].squeeze(0)
        
        # Post-process the saliency map to get the attention points
        saliency_map = torch.sigmoid(saliency_map)  # Sigmoid to get values between 0 and 1
        return saliency_map


def extract_attention_point(saliency_map):
    # Find the point with the highest saliency score
    _, _, h, w = saliency_map.shape
    max_idx = torch.argmax(saliency_map)
    y, x = divmod(max_idx.item(), w)
    return x, y  # Return the (x, y) coordinates of the most salient point


def apply_spotlight_filter(frame, saliency_map, spotlight_size=400):
    frame_height, frame_width = frame.shape[1], frame.shape[2]
    half_size = spotlight_size // 2

    # Extract the attention point from the saliency map
    gaze_point = extract_attention_point(saliency_map)
    gaze_x = gaze_point[0]
    gaze_y = gaze_point[1]

    # Clamp the gaze point to ensure it stays within the image boundaries
    gaze_x = min(max(gaze_x, half_size), frame_width - half_size)
    gaze_y = min(max(gaze_y, half_size), frame_height - half_size)

    x_min = max(gaze_x - half_size, 0)
    x_max = min(gaze_x + half_size, frame_width)
    y_min = max(gaze_y - half_size, 0)
    y_max = min(gaze_y + half_size, frame_height)

    # Ensure the crop does not result in zero dimensions
    if x_max - x_min == 0 or y_max - y_min == 0:
        raise ValueError("Crop resulted in an invalid size: zero height or width.")

    spotlight_frame = frame[:, y_min:y_max, x_min:x_max]

    # Resize the cropped region back to the original frame size
    return F.interpolate(spotlight_frame.unsqueeze(0), size=(frame_height, frame_width), mode='bilinear').squeeze(0)


def extract_features_from_video(video_path, model, transform):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return None

    features_conv2_list = []
    features_pool5_list = []
    features_fc1_list = []
    features_fc2_list = []
    output_list = []

    saliency_model = SaliencyModel()
    model.eval()  # Set the model to evaluation mode
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame)

        # Compute the saliency map
        saliency_map = saliency_model.compute_saliency(frame)

        # Apply the saliency-based spotlight filter
        frame = apply_spotlight_filter(frame, saliency_map)

        frame = frame.unsqueeze(0)  # Add batch dimension

        # Extract features
        with torch.no_grad():
            features_conv2, features_pool5, features_fc1, features_fc2, output = model(frame)

        # Append features to lists
        features_conv2_list.append(features_conv2.squeeze(0))
        features_pool5_list.append(features_pool5.squeeze(0))
        features_fc1_list.append(features_fc1.squeeze(0))
        features_fc2_list.append(features_fc2.squeeze(0))
        output_list.append(output.squeeze(0))
    
    cap.release()

    # Stack all features into tensors
    return {
        'conv2': torch.stack(features_conv2_list),
        'pool5': torch.stack(features_pool5_list),
        'fc1': torch.stack(features_fc1_list),
        'fc2': torch.stack(features_fc2_list),
        'output': torch.stack(output_list),
        'frame_rate': frame_rate
    }


if __name__ == "__main__":
    model = AlexNetModel()
    transform = get_transform()
    accumulator = FeatureAccumulator()
    actual_time_list = []
    predicted_durations = []
    video_dir = '../videos' 
    # List all video files in the directory
    all_videos = [os.path.join(video_dir, f) 
                  for f in os.listdir(video_dir) 
                  if f.endswith(('.mp4', '.avi', '.mkv'))]
    # Split into training and testing sets
    train_videos = all_videos[int(0.7 * len(all_videos)):]  # 70% for training
    test_videos = all_videos[:int(0.3 * len(all_videos))]   # 30% for testing
    # train 
    regressor = train_time_estimator(train_videos, model, transform, accumulator)
    # test/predict
    for video in test_videos:
        predicted_duration, actual_duration = predict_time_for_video(video, model, transform, accumulator, regressor)
        actual_time_list.append(actual_duration)
        predicted_durations.append(predicted_duration)
        print(f"Predicted times for {video}: {predicted_duration}")

    plot_durations(test_videos, actual_time_list, predicted_durations)
