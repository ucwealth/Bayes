from scipy.special import gammaln, psi
import numpy as np

class PoissonGammaSurpriseModel:
    def __init__(self, data, num_time_scales=5, a1=1.0, b1=1.0, zeta=0.7):
        """
        Initialize the PoissonGammaSurpriseModel with parameters.

        Args:
            data (np.ndarray): A 2D array of feature detector outputs (Poisson rates λ) for the current video frame.
            num_time_scales (int): Number of different time scales to compute surprise over.
            a1 (float): Shape parameter of the initial Gamma prior.
            b1 (float): Rate parameter of the initial Gamma prior.
            zeta (float): Relaxation term for the prior variance (0 < ζ < 1).
        """
        self.data = data
        self.num_time_scales = num_time_scales
        self.a1 = a1
        self.b1 = b1
        self.zeta = zeta
        self.models_a = [np.full_like(data, a1, dtype=float) for _ in range(num_time_scales)]
        self.models_b = [np.full_like(data, b1, dtype=float) for _ in range(num_time_scales)]

    def compute_surprise(self, data, time_scale):
        """
        Compute the Poisson-Gamma surprise for each pixel in the feature map at a given time scale.

        Args:
            data (np.ndarray): A 2D array of feature detector outputs (Poisson rates λ) for the current video frame.
            time_scale (int): The current time scale index.

        Returns:
            np.ndarray: A 2D array of Poisson-Gamma surprise values.
        """
        a = self.models_a[time_scale]
        b = self.models_b[time_scale]
        surprise_map = np.zeros_like(data, dtype=float)

        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                data_sample = data[y, x]
                a2 = self.zeta * a[y, x] + data_sample
                b2 = self.zeta * b[y, x] + 1

                if b[y, x] > 0 and b2 > 0 and a[y, x] > 0 and a2 > 0:
                    kl_divergence = (a[y, x] * np.log(b[y, x] / b2) - gammaln(a[y, x]) + gammaln(a2)
                                     + a2 * (psi(a2) - np.log(b2)) + a[y, x] * (np.log(b[y, x]) - psi(a[y, x])))
                else:
                    kl_divergence = 0

                surprise_map[y, x] = kl_divergence

                a[y, x] = a2
                b[y, x] = b2
        
        return surprise_map
