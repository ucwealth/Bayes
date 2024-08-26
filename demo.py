import cv2
import os
import numpy as np
import torch
from src.Utils.process_video import VideoProcessing
from src.Utils.visualizer import Visualizer
from src.Bayesian.extract_features import FeatureExtractor, MultiScalePyramid
from src.Bayesian.poisson2 import PoissonGammaSurpriseModel
from src.Bayesian.gaussian2 import GaussianSurpriseModel
from src.compute_surprise import FinalSurprise 

# loop through video folder, extract frames from each, feed to extractor, 
# return feature map, break each map into pixels, feed into surprise models 

def demo_model(video_folder_path, num_time_scales=5):
    """
    Process each video in a folder, extract frames, compute feature maps, and calculate surprise.

    Args:
        video_folder_path (str): Path to the folder containing video files.
        output_folder_path (str): Path to the folder where output should be saved.
        num_time_scales (int): Number of different time scales to compute surprise over.
    """
    # Ensure the output folder exists
    # os.makedirs(output_folder_path, exist_ok=True)

    # Initialize feature extractor
    pyramid_generator = MultiScalePyramid()
    aggregated_surprise_per_video = []

    # Loop through all video files in the folder
    for video_file in os.listdir(video_folder_path):
        if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')): 
            video_path = os.path.join(video_folder_path, video_file)

            video_processor = VideoProcessing(video_path)
            frames = video_processor.extract_all_frames() # returns list of frames in the video

            # Iterate through frames to get both the current and previous frame
            average_surprise_per_frame = []
            for i in range(1, len(frames)):
                current_frame = frames[i]
                previous_frame = frames[i - 1]

                # Convert feature_map to a PyTorch tensor
                current_frame_tensor = torch.from_numpy(current_frame).permute(2, 0, 1).unsqueeze(0).float()  # Convert NumPy array to PyTorch tensor
                previous_frame_tensor = torch.from_numpy(previous_frame).permute(2, 0, 1).unsqueeze(0).float() 

                # Extract features from the current frame
                feature_maps = pyramid_generator(current_frame_tensor, previous_frame_tensor) # dict 

                # Process each map into pixels and feed into surprise model
                average_surprise_per_feature_map = []
                for feature_name, feature_tensor in feature_maps.items():
                    # print(f"Processing feature map: {feature_name}")
                    # print("Feature Tensor Shape: ", feature_tensor.shape)

                    # Convert feature tensor to NumPy
                    feature_tensor_np = feature_tensor.detach().numpy()

                    # Iterate over the height and width of the feature map
                    aggregated_surprise_per_pixel_list = []
                    for y in range(feature_tensor_np.shape[1]):  # Assuming shape is (C, H, W)
                        for x in range(feature_tensor_np.shape[2]):
                            pixel_value = feature_tensor_np[:, y, x]  # Get pixel value (all channels at this location)
                            # Feed pixel_value into surprise model
                            surprise_map = FinalSurprise(pixel_value, num_time_scales=num_time_scales)

                            # Compute surprise
                            surprise_per_pixel = surprise_map.compute_surprise_map() # 2d array

                            # Get mean of all surprises per pixel
                            aggregated_surprise_per_pixel_list.append(np.mean(surprise_per_pixel, axis=0))
                    
                    print(f"aggregated_surprise for feature map {feature_name} : {aggregated_surprise_per_pixel_list}")

                # average of surprise for each feature map 
                average_surprise_per_feature_map.append(np.mean(aggregated_surprise_per_pixel_list, axis=0))

            average_surprise_per_frame.append(np.mean(average_surprise_per_feature_map, axis=0))
            max_surprise_per_frame = np.max(average_surprise_per_feature_map, axis=0)

            normalized_surprise_map = (average_surprise_per_frame - np.min(average_surprise_per_frame)
                                           ) / (np.max(average_surprise_per_frame) - np.min(average_surprise_per_frame))
            print("normalized_surprise_map: ", normalized_surprise_map)
            threshold = np.mean(normalized_surprise_map) + 2 * np.std(normalized_surprise_map)
            salient_regions = normalized_surprise_map > threshold
            blobs = cv2.findContours((salient_regions * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            smoothed_saliency_map = cv2.GaussianBlur(normalized_surprise_map, (5, 5), 0)
            # visualize
            # Feed into model 

    print("Processing completed for all videos in the folder.")
    # aggregated_surprise_per_video.append(np.mean(average_surprise_per_frame, axis=0))


# Example usage
if __name__ == "__main__":
    demo_model('sampleVideos')
