import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import os
from bayesian_surprise import SaliencyPredictor
from original_time_model import AlexNetModel, FeatureAccumulator, get_transform, plot_durations, predict_time_for_video, train_time_estimator

""" This code integrates the bayesian surprise model into the original time model """

# Function to apply a spotlight filter based on the computed saliency map
def apply_saliency_spotlight(frame, saliency_map, spotlight_size=400):
    """
    Applies a spotlight filter on the frame focused on the most salient point.

    Params:
    frame (Tensor): The video frame to process.
    saliency_map (Tensor): The saliency map where higher values indicate greater saliency.
    spotlight_size (int): The diameter of the spotlight in pixels.

    Returns:
    Tensor: The cropped frame resized back to the original frame dimensions.
    """
    frame_height, frame_width = frame.shape[1], frame.shape[2]
    half_size = spotlight_size // 2

    # Find the coordinates of the maximum saliency point
    max_val_idx = saliency_map.argmax()
    y, x = divmod(max_val_idx, frame_width)

    # Ensure coordinates are within frame boundaries
    x_min = max(x - half_size, 0)
    x_max = min(x + half_size, frame_width)
    y_min = max(y - half_size, 0)
    y_max = min(y + half_size, frame_height)

    # Check and prevent crop dimensions from being zero
    if x_max - x_min == 0 or y_max - y_min == 0:
        raise ValueError("Crop resulted in an invalid size: zero height or width.")

    # Crop and resize the spotlighted area
    spotlight_frame = frame[:, y_min:y_max, x_min:x_max]
    return F.interpolate(spotlight_frame.unsqueeze(0), size=(frame_height, frame_width), mode='bilinear').squeeze(0)

# Function to extract features from video frames using saliency-based filtering
def extract_features_with_saliency(video_path, model, saliency_model, transform, device):
    """
    Processes a video to extract features using a saliency-based spotlight filter.

    Params:
    video_path (str): Path to the video file.
    model (nn.Module): The neural network model for feature extraction.
    saliency_model (nn.Module): The saliency prediction model.
    transform (callable): Transformations to apply to video frames.
    device (torch.device): The device to run the computations on.

    Returns:
    dict: A dictionary containing stacks of feature tensors and the frame rate.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return None

    features_conv2_list = []
    features_pool5_list = []
    features_fc1_list = []
    features_fc2_list = []
    output_list = []

    model.eval()  # Set model to evaluation mode
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    ret, previous_frame = cap.read()
    if not ret:
        raise ValueError("Couldn't read the first frame")

    previous_frame = transform(previous_frame).unsqueeze(0).to(device)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess and predict saliency
        current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_frame = transform(current_frame).unsqueeze(0).to(device)
        with torch.no_grad():
            saliency_map = saliency_model(current_frame, previous_frame).squeeze().cpu().numpy()

        # Filter frame based on saliency and extract features
        frame_with_spotlight = apply_saliency_spotlight(current_frame[0], saliency_map)
        with torch.no_grad():
            features = model(frame_with_spotlight.unsqueeze(0))
            features_conv2_list.append(features[0].squeeze(0))
            features_pool5_list.append(features[1].squeeze(0))
            features_fc1_list.append(features[2].squeeze(0))
            features_fc2_list.append(features[3].squeeze(0))
            output_list.append(features[4].squeeze(0))
        
        previous_frame = current_frame
    
    cap.release()

    return {
        'conv2': torch.stack(features_conv2_list),
        'pool5': torch.stack(features_pool5_list),
        'fc1': torch.stack(features_fc1_list),
        'fc2': torch.stack(features_fc2_list),
        'output': torch.stack(output_list),
        'frame_rate': frame_rate
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNetModel().to(device)
    saliency_model = SaliencyPredictor(backbone='vgg16').to(device)
    transform = get_transform()
    accumulator = FeatureAccumulator()

    # Define video directories and process them
    video_dir = '../videos'
    all_videos = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mkv'))]

    # Split videos for training and testing
    train_videos = all_videos[int(0.7 * len(all_videos)):]
    test_videos = all_videos[:int(0.3 * len(all_videos))]

    # Train and test the model
    regressor = train_time_estimator(train_videos, model, transform, accumulator)
    actual_time_list = []
    predicted_durations = []

    for video in test_videos:
        result = extract_features_with_saliency(video, model, saliency_model, transform, device)
        predicted_duration, actual_duration = predict_time_for_video(video, model, transform, accumulator, regressor)
        actual_time_list.append(actual_duration)
        predicted_durations.append(predicted_duration)
        print(f"Predicted times for {video}: {predicted_duration}")

    # Visualize results 
    plot_durations(test_videos, actual_time_list, predicted_durations)
