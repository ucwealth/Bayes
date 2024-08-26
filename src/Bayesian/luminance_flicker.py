import torch
import torch.nn as nn
import torch.nn.functional as F

class LuminanceCalculator(nn.Module):
    """ Calculate the luminance of an image given in RGB format. """
    def __init__(self):
        super(LuminanceCalculator, self).__init__()
        # Weights for converting RGB to luminance (using the NTSC formula for simplicity)
        self.register_buffer('luminance_weights', torch.tensor([0.2989, 0.5870, 0.1140]))

    def forward(self, x):
        # x is expected to be in shape [batch, channels, height, width] and in RGB format
        # Apply the weights across the color channels to get the luminance image
        luminance = torch.sum(x * self.luminance_weights[None, :, None, None], dim=1, keepdim=True)
        return luminance

class LuminanceFlickerDetector(nn.Module):
    """ Detect luminance flicker between two consecutive frames. """
    def __init__(self):
        super(LuminanceFlickerDetector, self).__init__()
        self.luminance_calculator = LuminanceCalculator()

    def forward(self, current_frame, previous_frame):
        # Compute luminance for both the current and the previous frame
        current_luminance = self.luminance_calculator(current_frame)
        previous_luminance = self.luminance_calculator(previous_frame)

        # Calculate the flicker by finding the absolute difference between the two luminance images
        flicker = torch.abs(current_luminance - previous_luminance)
        return flicker

# Example usage
if __name__ == "__main__":
    # Initialize the flicker detection module
    flicker_detector = LuminanceFlickerDetector()
    
    # Create dummy data simulating two consecutive video frames
    current_frame = torch.rand(1, 3, 480, 640)  # Random image
    previous_frame = torch.rand(1, 3, 480, 640)  # Random image
    
    # Detect flicker between the two frames
    flicker = flicker_detector(current_frame, previous_frame)
    
    print(flicker.shape)  # Should print the shape of the flicker map, e.g., [1, 1, 480, 640]
