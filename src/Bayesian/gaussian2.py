import numpy as np
import sys, os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utils.visualizer import Visualizer 

class GaussianSurpriseModel:
    def __init__(self, data, num_time_scales=5, prior_mean=0.0, prior_variance=1.0, data_variance=5.0, zeta=0.7):
        """
        Initialize the GaussianSurpriseModel with priors and parameters.

        Args:
            data (np.ndarray): A 2D array of feature detector outputs for the current video frame.
            num_time_scales (int): Number of different time scales to compute surprise over.
            prior_mean (float): Initial prior mean for the Gaussian distribution.
            prior_variance (float): Initial prior variance for the Gaussian distribution.
            data_variance (float): Fixed variance reflecting sensor noise.
            zeta (float): Relaxation term for the prior variance (0 < ζ < 1).
        """
        self.data = data
        self.num_time_scales = num_time_scales
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.data_variance = data_variance
        self.zeta = zeta
        self.models_mean = [np.full_like(data, prior_mean) for _ in range(num_time_scales)]
        self.models_variance = [np.full_like(data, prior_variance) for _ in range(num_time_scales)]

    def compute_surprise(self, data, time_scale):
        """
        Compute the Gaussian surprise for each pixel in the feature map at a given time scale.

        Args:
            data (np.ndarray): A 2D array of feature detector outputs for the current video frame.
            time_scale (int): The current time scale index.

        Returns:
            np.ndarray: A 2D array of Gaussian surprise values.
        """
        mean = self.models_mean[time_scale]
        variance = self.models_variance[time_scale]
        surprise_map = np.zeros_like(data, dtype=float)

        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                data_sample = data[y, x]
                posterior_variance = 1 / ((1 / (self.zeta * variance[y, x])) + (1 / self.data_variance))
                posterior_mean = posterior_variance * ((mean[y, x] / (self.zeta * variance[y, x])) + (data_sample / self.data_variance))
                
                kl_divergence = 0.5 * (np.log(posterior_variance / variance[y, x]) + 
                                       (variance[y, x] + (mean[y, x] - posterior_mean) ** 2) / posterior_variance - 1)
                surprise_map[y, x] = kl_divergence

                mean[y, x] = posterior_mean
                variance[y, x] = posterior_variance
        
        return surprise_map
