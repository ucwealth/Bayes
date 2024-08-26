import numpy as np
from Bayesian.gaussian2 import GaussianSurpriseModel
from Bayesian.poisson2 import PoissonGammaSurpriseModel
from Utils.visualizer import Visualizer

"""
This is the Implementation of the Neural Network outlined in the paper titled 
Of bits and wows: A Bayesian theory of surprise with applications to attention
By Laurent Itti and Pierre Baldi 

======================================

Input video frames are analyzed in dyadic image pyramids with 9 scales (from scale 0 corresponding to the original image, to scale 8 
corresponding to the image reduced by a factor of 256 horizontally and vertically). 
The pyramids are constructed by iteratively filtering and decimating the input image. 
In this implementation, pyramids are computed for the following low-level visual features thought to guide human attention:

(1) luminance or intensity 
(2) red–green color opponency 
(3) blue–yellow color opponency; 
(4) four oriented edge filters (using Gabor kernels) spanning 180∘ 
(5) luminance flicker (as computed from the difference between the previous image and the current one) and
(6) four directions of motion spanning 360∘. 

- The output of the pyramids above is DATA for below model 
- Current beliefs = prior
- Apply bayes theorem to compute posterior
- Posterior in frame 1 becomes prior for frame 2
- Using conjugate priors facilitates this process by ensuring that the posterior has the same functional form as the prior. 
- 2 model classes are used to implement surprise
- Gaussian with a guassian prior
- Poisson with a gamma prior 
- To accommodate for changing data and events at multiple temporal scales, we employ a chained cascade of surprise detectors 
at every pixel in every feature map, where the output of one surprise detector serves as input to the next detector in the cascade. 

- Our implementation uses 5 such cascaded feature detectors at every pixel and for every feature. 
- The first (fastest) is updated with feature map data from the low-level feature computations, and detector i+1 samples from i, 
so that time constants increase exponentially with i. In total, the system thus comprises 72[maps] * (40*30)[pixels] * 5[timescales]
surprise detectors 
- temporally surprise is accounted for by or local temporal surprise, a single neuron in one of the feature maps is considered, 
and the prior is established over time from the observations received to date in the receptive field of that one neuron
- spatial surprise is by 
- Temporal and spatial are combined to give final surprise

** TO DO **
Feature extractor returns 72 feature maps, currently as a dict of about 6 features(keys). Each feature's value is a tensor
Extract each feature map, convert to 2D array, before input into gaussian or poisson
Surprise is computed for every pixel in each of the 72 center-surround feature maps

- section 2 has the math definition of surprise 
- process video and extract frames 

"""

class FinalSurprise:
    def __init__(self, data, num_time_scales=5):
        """
        Initialize the FinalSurprise class with the input data and the number of time scales.

        Args:
            data (np.ndarray): A 2D array of feature detector outputs for the current video frame.
            num_time_scales (int): Number of different time scales to compute surprise over.
        """
        self.data = data
        self.num_time_scales = num_time_scales
        self.models = [np.ones_like(data) for _ in range(num_time_scales)]  # Initialize models to ones
        self.gaussian_model = GaussianSurpriseModel(data, num_time_scales=num_time_scales)
        self.poisson_model = PoissonGammaSurpriseModel(data, num_time_scales=num_time_scales)
        self.S = np.ones_like(data)  # Combined surprise array initialized to ones

    def compute_surprise_map(self):
        """
        Compute the combined surprise values over space and time at multiple scales.

        Returns:
            np.ndarray: A 2D array of combined surprise values.
        """
        # Loop over each time scale to compute surprises
        for i in range(self.num_time_scales):
            # Compute Gaussian surprise for the current time scale
            SL = self.gaussian_model.compute_surprise(self.data, time_scale=i)

            # Compute Poisson surprise for the current time scale
            SS = self.poisson_model.compute_surprise(self.data, time_scale=i)

            # Update combined surprise with a weighted sum of SL and SS, raised to the power of 1/3
            self.S *= (SL + SS / 20) ** (1/3)

            # Update the model for the next time scale based on the current surprise map
            self.data = self.S  # Update self.data with the current combined surprise

        return self.S



# Example usage:
if __name__ == "__main__":
    # Example input data (2D array representing feature detector outputs)
    input_data = np.random.rand(32, 32)  # Example: 32x32 feature map with random values

    # Initialize the SurpriseMapCalculator with input data and number of time scales
    calculator = FinalSurprise(input_data)

    # Compute surprise map
    surprise_map = calculator.compute_surprise_map()

    # Display the computed surprise map
    print("Surprise Map:")
    print(surprise_map)

    # apply poisson dist to input before feeding to pisson model ??

    visualizer = Visualizer(title='Final Surprise Map', xlabel='Value', ylabel='Frequency', figsize=(8, 6))
    visualizer.visualize_histogram(input_data, title="Data for poisson")
    visualizer.visualize_image(surprise_map)
