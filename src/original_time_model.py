import os
import torch
import torch.nn as nn
from torchvision.models import alexnet, AlexNet_Weights
import torchvision.transforms as transforms
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

"""

    Code Implementation of the perceptual time model outlined in the paper titled:
    "Activity in perceptual classification networks as a basis for human subjective time perception" 
    By Roseboom et al., 2019

"""
class AlexNetModel(nn.Module):
    def __init__(self):
        super(AlexNetModel, self).__init__()
        self.alexnet = alexnet(weights=AlexNet_Weights.DEFAULT)
        
        # Extract specific layers from AlexNet
        self.conv2 = nn.Sequential(*list(self.alexnet.features.children())[:5])  # Up to and including conv2
        self.pool5 = nn.Sequential(*list(self.alexnet.features.children())[5:])  # From after conv2 to pool5
        self.fc7 = nn.Sequential(*list(self.alexnet.classifier.children())[:5])  # Up to and including fc7
        self.output = nn.Sequential(*list(self.alexnet.classifier.children())[5:])  # Output layer

        # Adding optional Dropout and BatchNorm for better regularization
        self.dropout = nn.Dropout(0.5)
        self.batchnorm1 = nn.BatchNorm1d(4096)
        self.batchnorm2 = nn.BatchNorm1d(4096)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Forward pass through each layer, capturing outputs
        # print(f"Input shape: {x.shape}")
        features_conv2 = self.conv2(x)
        # print(f"After conv2: {features_conv2.shape}")
        features_pool5 = self.pool5(features_conv2)
        # print(f"After pool5: {features_pool5.shape}")
        
        # Flatten before passing through fully connected layers
        flattened = torch.flatten(features_pool5, 1)

        classifier_fc7 = self.relu(self.fc7(flattened))
        classifier_fc7 = self.batchnorm1(classifier_fc7)
        classifier_fc7 = self.dropout(classifier_fc7)

        output = self.output(classifier_fc7)
        output = F.softmax(output, dim=1)
        
        return features_conv2, features_pool5, classifier_fc7, output


class FeatureAccumulator:
    def __init__(self, thresholds=None, decay_rate=100.0, random_seed=None):
        if thresholds is None:
            thresholds = [1.0, 1.5, 2.0, 2.5]

        self.thresholds = torch.tensor(thresholds, dtype=torch.float32)
        self.decay_rate = torch.tensor(decay_rate, dtype=torch.float32)
        self.accumulated_changes = torch.zeros(len(thresholds), dtype=torch.int32)

        if random_seed is not None:
            torch.manual_seed(random_seed)

    def update_thresholds(self):
        # Decay thresholds over time
        self.thresholds *= torch.exp(torch.tensor(-1.0) / self.decay_rate)

    def process_features(self, features):
        changes_detected = []
        for i, feature in enumerate(features):
            magnitude = torch.norm(feature)
            change_detected = magnitude > self.thresholds[i]
            self.accumulated_changes[i] += change_detected.int()
            changes_detected.append(change_detected.item())
            if change_detected:
                self.thresholds[i] = 1.0  # Reset threshold for this layer
        self.update_thresholds()
        return changes_detected


class TimeEstimator:
    def __init__(self):
        # Initialize SVR with a radial basis function kernel
        self.regressor = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)

    def train(self, features, times):
        self.regressor.fit(features, times)

    def predict(self, features):
        return self.regressor.predict(features)
    
    
def plot_results(predicted_times: np.ndarray, actual: np.ndarray):
        try:
            plt.plot(actual, predicted_times, 'ro', label='Predicted vs Actual')
            plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'k--', lw=2)
            plt.xlabel('Actual Time (s)')
            plt.ylabel('Predicted Time (s)')
            plt.title('Actual vs Predicted Time')
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"An error occurred during plotting: {e}")


def get_transform():
    """Returns the transformation pipeline for video frames."""
    return transforms.Compose([
        transforms.ToPILImage(),  # Convert ndarray to PIL Image
        transforms.Resize(256),  # Resize the image
        transforms.CenterCrop(224),  # Crop a central part of the image
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def train_time_estimator(training_videos: list, model: AlexNetModel, 
                         transform, accumulator: FeatureAccumulator) -> TimeEstimator:
    """
    Train the TimeEstimator using a list of training videos.
    :param training_videos: List of paths to training video files.
    :param model: Pre-trained AlexNet model for feature extraction.
    :param transform: Transformation to be applied on video frames.
    :param accumulator: FeatureAccumulator to process the extracted features.
    :return: Trained TimeEstimator.
    """
    all_changes_detected = []
    all_times = []
    for video_path in training_videos:
        result = process_video_features(video_path, model, transform, accumulator)
        if result is None:
            continue
        changes_detected = np.array(result['changes_detected'])
        frame_rate = result['frame_rate']
        times = np.array([i / frame_rate for i in range(changes_detected.shape[0])])
        print("Accumulated Changes:", result['accumulated_changes'])
        # print("Frame rate: ", frame_rate)
        # print("Times: ", times)
        all_changes_detected.append(changes_detected)
        all_times.append(times)
    # Make sure there is data to concatenate
    if not all_changes_detected or not all_times:
        raise ValueError("No valid data to train the TimeEstimator.")
    # Concatenate all changes and times
    all_changes_detected = np.concatenate(all_changes_detected, axis=0)
    all_times = np.concatenate(all_times, axis=0)
    # Initialize and train the regressor
    estimator = TimeEstimator()
    estimator.train(all_changes_detected, all_times)
    return estimator


def predict_time_for_video(video_path: str, 
                           model: AlexNetModel, transform, 
                           accumulator: FeatureAccumulator, 
                           regressor: TimeEstimator) -> np.ndarray:
    result = process_video_features(video_path, model, transform, accumulator)
    actual_time = result['duration_seconds']

    if result is None:
        print(f"Error processing video: {video_path}")
        return None
    changes_detected = np.array(result['changes_detected'])
    predicted_times = regressor.predict(changes_detected)
    # Aggregate the predicted times to get the final predicted duration
    predicted_duration = predicted_times[-1] if len(predicted_times) > 0 else None
    print("predicted_duration", predicted_duration)
    return predicted_duration, actual_time


def predict_times_for_videos_in_dir(dir_path: str | list, 
                                          model: AlexNetModel, transform, 
                                          accumulator: FeatureAccumulator, 
                                          regressor: TimeEstimator) -> dict:
    predicted_times_dict = {}
    for filename in os.listdir(dir_path):
        if filename.endswith('.mp4') or filename.endswith('.avi') or filename.endswith('.mkv'):
            video_path = os.path.join(dir_path, filename)
            predicted_times, _ = predict_time_for_video(video_path, model, transform, accumulator, regressor)
            if predicted_times is not None:
                predicted_times_dict[filename] = predicted_times
    for video, times in predicted_times_dict.items():
        print(f"Predicted times for {video}: {times}")
    return predicted_times_dict


def plot_durations(video_names, actual_durations, predicted_durations):

    # Check if the lengths of the lists are the same
    if len(video_names) != len(actual_durations) or len(video_names) != len(predicted_durations):
        raise ValueError("Mismatch in list lengths: video_names, actual_durations, and predicted_durations must have the same length.")
    
    print(len(video_names))
    print(len(actual_durations))
    # print(len(predicted_durations))
    
    x = np.arange(len(video_names))
    width = 0.35
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, actual_durations, width, label='Actual')
    rects2 = ax.bar(x + width/2, predicted_durations, width, label='Predicted')
    
    ax.set_xlabel('Videos')
    ax.set_ylabel('Duration (s)')
    ax.set_title('Actual vs Predicted Video Durations')
    ax.set_xticks(x)
    ax.set_xticklabels(video_names, rotation=45, ha="right")
    ax.legend()
    
    fig.tight_layout()
    plt.show()


def process_video_features(video_path, model, transform, accumulator, gaze_data=None, saliency=False):
    features = extract_features_from_video(video_path, model, transform, gaze_data)

    if features is None:
        return None
    
    all_changes_detected = []
    frame_rate = features['frame_rate']

    # Process each set of features
    for i in range(features['fc7'].size(0)):
        layer_features = [features['conv2'][i], features['pool5'][i], 
                          features['fc7'][i],
                          features['output'][i]]
        changes_detected = accumulator.process_features(layer_features)
        all_changes_detected.append(changes_detected)
    
    # Estimate duration based on accumulated changes
    num_frames = len(features['fc7'])
    duration_seconds = num_frames / frame_rate

    return {
        'accumulated_changes': accumulator.accumulated_changes,
        'changes_detected': all_changes_detected,
        'duration_seconds': duration_seconds,
        'frame_rate': frame_rate
    }


def simulate_gaze_data(num_frames, frame_width, frame_height):
    """Simulate gaze data centered around the middle of the frame."""
    center_x = frame_width // 2
    center_y = frame_height // 2
    gaze_data = []

    # Random movement around the center
    max_shift = 50  # Max shift in pixels from the center

    for _ in range(num_frames):
        shift_x = np.random.randint(-max_shift, max_shift)
        shift_y = np.random.randint(-max_shift, max_shift)
        gaze_x = center_x + shift_x
        gaze_y = center_y + shift_y
        gaze_data.append((gaze_x, gaze_y))
    
    return gaze_data


def apply_spotlight_filter(frame, gaze_point, spotlight_size=400): # make this resusable
    """Apply a spotlight filter based on simulated gaze data."""
    frame_height, frame_width = frame.shape[1], frame.shape[2]
    half_size = spotlight_size // 2

    # Clamp the gaze point to ensure it stays within the image boundaries
    gaze_x = min(max(gaze_point[0], half_size), frame_width - half_size)
    gaze_y = min(max(gaze_point[1], half_size), frame_height - half_size)

    x_min = max(gaze_x - half_size, 0)
    x_max = min(gaze_x + half_size, frame_width)
    y_min = max(gaze_y - half_size, 0)
    y_max = min(gaze_y + half_size, frame_height)

    # Ensure the crop does not result in zero dimensions
    if x_max - x_min == 0 or y_max - y_min == 0:
        raise ValueError("Crop resulted in an invalid size: zero height or width.")

    spotlight_frame = frame[:, y_min:y_max, x_min:x_max]

    # Resize the cropped region back to the original frame size
    if spotlight_frame.size(1) > 0 and spotlight_frame.size(2) > 0:
        return F.interpolate(spotlight_frame.unsqueeze(0), size=(frame_height, frame_width), mode='bilinear').squeeze(0)
    else:
        raise ValueError("Spotlight frame has invalid dimensions after cropping.")


def extract_features_from_video(video_path, model, transform, gaze_data=None): 
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return None

    features_conv2_list = []
    features_pool5_list = []
    classifier_fc7_list = []
    output_list = []

    model.eval()  # Set the model to evaluation mode
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame)

        frame_index = 0
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Simulate gaze data
        if gaze_data is None:
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            gaze_data = simulate_gaze_data(num_frames, frame_width, frame_height)

        # Apply the simulated gaze-based spotlight filter
        if frame_index < len(gaze_data):
            gaze_point = gaze_data[frame_index]
            frame = apply_spotlight_filter(frame, gaze_point)

        frame = frame.unsqueeze(0)  # Add batch dimension

        # Extract features
        with torch.no_grad():
            features_conv2, features_pool5, classifier_fc7, output = model(frame)

        # Append features to lists
        features_conv2_list.append(features_conv2.squeeze(0))
        features_pool5_list.append(features_pool5.squeeze(0)) 
        classifier_fc7_list.append(classifier_fc7.squeeze(0))
        # features_fc2_list.append(features_fc2.squeeze(0))
        output_list.append(output.squeeze(0))

        frame_index += 1
    
    cap.release()

    # Stack all features into tensors
    return {
        'conv2': torch.stack(features_conv2_list),
        'pool5': torch.stack(features_pool5_list),
        'fc7': torch.stack(classifier_fc7_list),
        'output': torch.stack(output_list),
        'frame_rate': frame_rate
    }


if __name__ == "__main__":
    model = AlexNetModel()
    transform = get_transform()
    accumulator = FeatureAccumulator()
    actual_time_list = []
    predicted_durations = []
    # video_path = '../videos/vid0.mp4'
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
    
    # Plot results
    plot_durations(test_videos, actual_time_list, predicted_durations)

