import torch
import torch.nn as nn
import torch.nn.functional as F

""" Optical Flow For Motion Detection """

class MotionDetector(nn.Module):
    def __init__(self):
        super(MotionDetector, self).__init__()
        # Sobel operators for gradient calculation
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

        # Set up the fixed weights for the Sobel filters
        sobel_x_weights = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y_weights = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_x.weight = nn.Parameter(sobel_x_weights, requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_y_weights, requires_grad=False)


    def forward(self, prev_frame, curr_frame):
        # Convert frames to grayscale if they are RGB
        if prev_frame.size(1) == 3:
            prev_frame = torch.mean(prev_frame, dim=1, keepdim=True)
        if curr_frame.size(1) == 3:
            curr_frame = torch.mean(curr_frame, dim=1, keepdim=True)

        # Calculate gradients
        prev_grad_x = self.sobel_x(prev_frame)
        prev_grad_y = self.sobel_y(prev_frame)
        curr_grad_x = self.sobel_x(curr_frame)
        curr_grad_y = self.sobel_y(curr_frame)

        # Estimate flow vectors
        flow_x = curr_grad_x - prev_grad_x
        flow_y = curr_grad_y - prev_grad_y

        # Thresholding to detect significant motion in each direction
        motion_east = F.relu(flow_x)  # Positive x-direction (Right/East)
        motion_west = F.relu(-flow_x) # Negative x-direction (Left/West)
        motion_south = F.relu(flow_y) # Positive y-direction (Down/South)
        motion_north = F.relu(-flow_y) # Negative y-direction (Up/North)

        # Combine all motion directions into a single tensor with 4 channels
        return torch.cat([motion_east, motion_west, motion_south, motion_north], dim=1)
  

# Example usage
if __name__ == "__main__":
    # Create dummy video frames
    prev_frame = torch.randn(1, 1, 100, 100)  # Previous frame
    curr_frame = torch.randn(1, 1, 100, 100)  # Current frame

    # Initialize the optical flow module
    optical_flow = MotionDetector()

    # Compute motion vectors
    motion_vectors = optical_flow(prev_frame, curr_frame)
    
    # Print output tensor sizes
    print("Motion East Shape:", motion_vectors[0].shape)
    print("Motion South Shape:", motion_vectors[1].shape)
    print("Motion West Shape:", motion_vectors[2].shape)
    print("Motion North Shape:", motion_vectors[3].shape)
