import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import os 
from bayesian_surprise import SaliencyPredictor
from original_time_model import AlexNetModel, FeatureAccumulator, get_transform, plot_durations, predict_time_for_video, train_time_estimator


""" Integration of Bayesian Suprise into the original model """
def apply_saliency_spotlight(frame, saliency_map, spotlight_size=400):
    frame_height, frame_width = frame.shape[1], frame.shape[2]
    half_size = spotlight_size // 2

    # Find the most salient point
    max_val_idx = saliency_map.argmax()
    y, x = divmod(max_val_idx, frame_width)  # Get the x, y coordinates

    # Clamp the coordinates to avoid out-of-bounds cropping
    x_min = max(x - half_size, 0)
    x_max = min(x + half_size, frame_width)
    y_min = max(y - half_size, 0)
    y_max = min(y + half_size, frame_height)

    # Ensure the crop does not result in zero dimensions
    if x_max - x_min == 0 or y_max - y_min == 0:
        raise ValueError("Crop resulted in an invalid size: zero height or width.")

    spotlight_frame = frame[:, y_min:y_max, x_min:x_max]

    # Resize the cropped region back to the original frame size
    return F.interpolate(spotlight_frame.unsqueeze(0), size=(frame_height, frame_width), mode='bilinear').squeeze(0)


def extract_features_with_saliency(video_path, model, saliency_model, transform, device):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return None

    features_conv2_list = []
    features_pool5_list = []
    features_fc1_list = []
    features_fc2_list = []
    output_list = []

    model.eval()  # Set the model to evaluation mode
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    ret, previous_frame = cap.read()
    if not ret:
        raise ValueError("Couldn't read the first frame")

    previous_frame = transform(previous_frame).unsqueeze(0).to(device)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the current frame
        current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_frame = transform(current_frame).unsqueeze(0).to(device)
        
        # Predict saliency for the current frame
        with torch.no_grad():
            saliency_map = saliency_model(current_frame, previous_frame).squeeze().cpu().numpy()

        # Apply the saliency-based spotlight filter
        frame_with_spotlight = apply_saliency_spotlight(current_frame[0], saliency_map)

        # Extract features from the spotlighted frame
        with torch.no_grad():
            features_conv2, features_pool5, features_fc1, features_fc2, output = model(frame_with_spotlight.unsqueeze(0))

        # Append features to lists
        features_conv2_list.append(features_conv2.squeeze(0))
        features_pool5_list.append(features_pool5.squeeze(0))
        features_fc1_list.append(features_fc1.squeeze(0))
        features_fc2_list.append(features_fc2.squeeze(0))
        output_list.append(output.squeeze(0))
        
        previous_frame = current_frame
    
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNetModel().to(device)
    saliency_model = SaliencyPredictor(backbone='vgg16').to(device)
    transform = get_transform()
    accumulator = FeatureAccumulator()

    video_dir = '../videos' 
    all_videos = [os.path.join(video_dir, f) 
                  for f in os.listdir(video_dir) 
                  if f.endswith(('.mp4', '.avi', '.mkv'))]

    train_videos = all_videos[int(0.7 * len(all_videos)):]  # Training videos
    test_videos = all_videos[:int(0.3 * len(all_videos))]   # Testing videos

    regressor = train_time_estimator(train_videos, model, transform, accumulator)

    actual_time_list = []
    predicted_durations = []

    for video in test_videos:
        result = extract_features_with_saliency(video, model, saliency_model, transform, device)
        # replace previous extract with current extract
        # process video

        predicted_duration, actual_duration = predict_time_for_video(video, model, transform, accumulator, regressor)
        actual_time_list.append(actual_duration)
        predicted_durations.append(predicted_duration)
        print(f"Predicted times for {video}: {predicted_duration}")

    plot_durations(test_videos, actual_time_list, predicted_durations)
