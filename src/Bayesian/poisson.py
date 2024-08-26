import numpy as np
import sys, os 
from scipy.special import gammaln, psi  # Gamma functions needed for surprise calculation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utils.visualizer import Visualizer 

class PoissonGammaSurpriseModel:
    def __init__(self, data, a1=1.0, b1=1.0, zeta=0.7, num_time_scales=5):
        """
        Initialize the PoissonGammaSurpriseModel with input data and parameters.

        Args:
            data (np.ndarray): A 2D array of feature detector outputs (Poisson rates λ) for the current video frame.
            a1 (float): Shape parameter of the initial Gamma prior.
            b1 (float): Rate parameter of the initial Gamma prior.
            zeta (float): Relaxation term for the prior variance (0 < ζ < 1).
            num_time_scales (int): Number of different time scales to compute surprise over.
        """
        self.data = data
        self.a1 = a1
        self.b1 = b1
        self.zeta = zeta
        self.num_time_scales = num_time_scales
        self.models_a = [np.full_like(data, a1, dtype=float) for _ in range(num_time_scales)]  # Prior 'a' parameters
        self.models_b = [np.full_like(data, b1, dtype=float) for _ in range(num_time_scales)]  # Prior 'b' parameters
        self.S = np.ones_like(data, dtype=float)  # Combined surprise array initialized to ones

    def compute_neighborhood_models(self, i):
        """
        Compute an array of neighborhood models from the previous model or from the data.

        Args:
            i (int): The current time scale index.

        Returns:
            tuple: Two 2D arrays representing the neighborhood model's parameters (a, b).
        """
        if i == 0:
            # For the first time scale, use the initial data directly.
            return self.data, np.ones_like(self.data)  # b2 initially all ones
        else:
            # For subsequent time scales, use the posterior parameters from the previous model.
            return self.models_a[i - 1], self.models_b[i - 1]

    def compute_surprise(self):
        """
        Compute the combined surprise values over space and time at multiple scales using Poisson-Gamma model.

        Returns:
            np.ndarray: A 2D array of combined surprise values.
        """
        for i in range(self.num_time_scales):
            # Compute neighborhood model from previous scale's model or data
            a1, b1 = self.compute_neighborhood_models(i)

            # Update model parameters 'a' and 'b' for the current time scale
            for y in range(self.data.shape[0]):
                for x in range(self.data.shape[1]):
                    # Data sample from the current frame (λ value from feature detector)
                    data_sample = self.data[y, x]

                    # Update Gamma parameters using Bayesian update rule with prior relaxation.
                    a2 = self.zeta * a1[y, x] + data_sample  # N = 1, so m̄ = data_sample
                    b2 = self.zeta * b1[y, x] + 1 # N = 1

                    # Store updated parameters
                    self.models_a[i][y, x] = a2
                    self.models_b[i][y, x] = b2

                    # Compute surprise using the exact formula for KL divergence for Gamma distributions
                    if b1[y, x] > 0 and b2 > 0 and a1[y, x] > 0 and a2 > 0:
                        kl_divergence = (a1[y, x] * np.log(b1[y, x] / b2) - gammaln(a1[y, x]) + gammaln(a2)
                                         + a2 * (psi(a2) - np.log(b2)) + a1[y, x] * (np.log(b1[y, x]) - psi(a1[y, x])))
                    else:
                        kl_divergence = 0  # Avoid invalid calculations

                    # Update combined surprise map, ensuring it stays in the valid range
                    self.S[y, x] *= max(kl_divergence, 1e-8) ** (1/3)  # Avoid multiplying by NaN or inf

            # Update data for the next time scale (using updated model mean as input)
            self.data = np.divide(self.models_a[i], self.models_b[i], out=np.zeros_like(self.models_a[i]), 
                                  where=self.models_b[i] != 0)  # Use the mean of the Gamma distribution as the new data

        return self.S
