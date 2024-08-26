import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""  Edge Class For Feature Extraction """
def gabor_kernel(frequency, theta, sigma_x, sigma_y, nstds=3, grid_size=15):
    """ Create a Gabor filter with the specified parameters """
    t = theta + np.pi / 2
    xmax = grid_size // 2
    ymax = grid_size // 2
    xmin = -xmax
    ymin = -ymax
    (x, y) = np.meshgrid(np.arange(xmin, xmax+1), np.arange(ymin, ymax+1))
    x_theta = x * np.cos(t) + y * np.sin(t)
    y_theta = -x * np.sin(t) + y * np.cos(t)
    gb = np.exp(-.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)) * np.cos(2 * np.pi * frequency * x_theta)
    return torch.tensor(gb, dtype=torch.float32)

class GaborFilters(nn.Module):
    def __init__(self):
        super(GaborFilters, self).__init__()
        # Initialize the convolutional layer with 4 filters
        self.conv = nn.Conv2d(1, 4, kernel_size=15, padding=7, bias=False)

        # Manually set the weights for Gabor filters
        self.conv.weight.data[0, 0] = gabor_kernel(frequency=0.4, theta=0, sigma_x=4, sigma_y=4)
        self.conv.weight.data[1, 0] = gabor_kernel(frequency=0.4, theta=np.pi/4, sigma_x=4, sigma_y=4)
        self.conv.weight.data[2, 0] = gabor_kernel(frequency=0.4, theta=np.pi/2, sigma_x=4, sigma_y=4)
        self.conv.weight.data[3, 0] = gabor_kernel(frequency=0.4, theta=3*np.pi/4, sigma_x=4, sigma_y=4)

        # Freeze the parameters since we don't want them to change during training
        self.conv.weight.requires_grad = False

    def forward(self, x):
        # Apply the Gabor filters
        x = self.conv(x)
        return x

# Example of usage
if __name__ == "__main__":
    # Initialize the Gabor filter module
    gabor_filter = GaborFilters()
    
    # Generate a simple test image, e.g., a white square on a black background
    test_image = torch.zeros((1, 1, 32, 32))
    test_image[0, 0, 8:24, 8:24] = 1
    
    # Apply the Gabor filters
    filtered_images = gabor_filter(test_image)
    
    # Plot the results
    fig, ax = plt.subplots(1, 5)
    ax[0].imshow(test_image[0, 0], cmap='gray')
    ax[0].set_title('Original Image')
    for i in range(4):
        ax[i+1].imshow(filtered_images[0, i].detach().numpy(), cmap='gray')
        ax[i+1].set_title(f'Filter {i+1}')
    plt.show()
