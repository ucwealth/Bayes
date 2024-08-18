import numpy as np
import cv2
import torch
import torch.nn.functional as F
from sklearn.svm import SVR
from bayesian_surprise import SaliencyPredictor


# Assuming `process_video` is the function to generate saliency maps for each frame
def process_video(video_path, model, device):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    saliency_maps = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame and generate saliency map
        frame_tensor = preprocess_frame(frame).unsqueeze(0).to(device)
        saliency_map = model(frame_tensor).squeeze().cpu().numpy()

        # Resize saliency map to match original frame size
        saliency_map = cv2.resize(saliency_map, (frame.shape[1], frame.shape[0]))
        saliency_maps.append(saliency_map)

    cap.release()
    return saliency_maps

# Assuming `compute_time_estimate` is the function to compute time perception
def compute_time_estimate(saliency_maps):
    # Use the accumulated saliency maps to estimate time
    # You can use Euclidean distance between maps, thresholding, or other techniques
    # Similar to the approach in the paper
    accumulated_changes = []

    for i in range(1, len(saliency_maps)):
        diff = np.linalg.norm(saliency_maps[i] - saliency_maps[i-1])
        accumulated_changes.append(diff)

    # Convert accumulated changes to time estimate using regression
    time_estimate = regress_time(accumulated_changes)
    return time_estimate

def regress_time(accumulated_changes):
    # Use SVR or another regression model to map changes to time
    svr = SVR(kernel='rbf')
    X = np.arange(len(accumulated_changes)).reshape(-1, 1)
    y = np.array(accumulated_changes)
    svr.fit(X, y)
    time_estimate = svr.predict(X)
    return time_estimate

# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
saliency_model = SaliencyPredictor(backbone='vgg16').to(device)

video_path = "../videos/vid6.mp4"
saliency_maps = process_video(video_path, saliency_model, device)
time_estimate = compute_time_estimate(saliency_maps)
print(f"Estimated time perception: {time_estimate}")
