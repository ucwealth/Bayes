import numpy as np
import sys, os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utils.visualizer import Visualizer 

class GaussianSurpriseModel:
    def __init__(self, data, prior_mean=0.0, prior_variance=1.0, data_variance=5.0, zeta=0.7, num_time_scales=5):
        """
        Initialize the GaussianSurpriseModel with input data, priors, and parameters.

        Args:
            data (np.ndarray): A 2D array of feature detector outputs for the current video frame.
            prior_mean (float): Initial prior mean for the Gaussian distribution.
            prior_variance (float): Initial prior variance for the Gaussian distribution.
            data_variance (float): Fixed variance reflecting sensor noise.
            zeta (float): Relaxation term for the prior variance (0 < ζ < 1).
            num_time_scales (int): Number of different time scales to compute surprise over.
        """
        self.data = data
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.data_variance = data_variance
        self.zeta = zeta
        self.num_time_scales = num_time_scales
        self.models_mean = [np.full_like(data, prior_mean) for _ in range(num_time_scales)]  # Prior means
        self.models_variance = [np.full_like(data, prior_variance) for _ in range(num_time_scales)]  # Prior variances
        self.S = np.ones_like(data)  # Combined surprise array initialized to ones

    def update_gaussian_parameters(self, prior_mean, prior_variance, data_sample, zeta):
        """
        Update Gaussian parameters using Bayesian update rule with prior relaxation.

        Args:
            prior_mean (np.ndarray): The prior mean for the Gaussian distribution.
            prior_variance (np.ndarray): The prior variance for the Gaussian distribution.
            data_sample (np.ndarray): The new data sample (feature detector response).
            zeta (float): Relaxation term for the prior variance.

        Returns:
            tuple: Updated posterior mean and variance as numpy arrays.
        """
        # Bayesian update rule with prior relaxation
        posterior_variance = 1 / ((1 / (zeta * prior_variance)) + (1 / self.data_variance))
        posterior_mean = posterior_variance * ((prior_mean / (zeta * prior_variance)) + (data_sample / self.data_variance))
        
        return posterior_mean, posterior_variance

    def compute_surprise(self):
        """
        Compute the combined surprise values over space and time at multiple scales using Gaussian model.

        Returns:
            np.ndarray: A 2D array of combined surprise values.
        """
        for i in range(self.num_time_scales):
            # Update model mean and variance for current time scale
            for y in range(self.data.shape[0]):
                for x in range(self.data.shape[1]):
                    # Retrieve current prior mean and variance
                    prior_mean = self.models_mean[i][y, x]
                    prior_variance = self.models_variance[i][y, x]

                    # Data sample from the current frame
                    data_sample = self.data[y, x]

                    # Update the mean and variance using the Bayesian update rule
                    updated_mean, updated_variance = self.update_gaussian_parameters(prior_mean, prior_variance, data_sample, self.zeta)

                    # Store updated mean and variance
                    self.models_mean[i][y, x] = updated_mean
                    self.models_variance[i][y, x] = updated_variance

                    # Compute surprise using the KL divergence for Gaussian distributions
                    kl_divergence = 0.5 * (np.log(updated_variance / prior_variance) + 
                                           (prior_variance + (prior_mean - updated_mean) ** 2) / updated_variance - 1)
                    
                    # Update combined surprise map
                    self.S[y, x] *= (kl_divergence ** (1/3))

            # Update data for the next time scale (using updated model mean as input)
            self.data = self.models_mean[i]

        return self.S
    
