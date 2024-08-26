import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    def __init__(self, title='Visualization', xlabel='X Position', ylabel='Y Position', figsize=(8, 6)):
        """
        Initialize the Visualizer class with common settings for visualization.

        Args:
            title (str): Default title for the visualizations.
            xlabel (str): Default label for the x-axis.
            ylabel (str): Default label for the y-axis.
            figsize (tuple): Default size of the figure.
        """
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.figsize = figsize

    def visualize_image(self, input_map, title=None, cmap='hot', figsize=None, xlabel=None, ylabel=None):
        """
        Visualize the computed surprise map using matplotlib.

        Args:
            surprise_map (np.ndarray): A 2D array of surprise values to visualize.
            title (str): Title of the plot (optional, overrides default).
            cmap (str): Colormap to use for visualization.
            figsize (tuple): Size of the figure (optional, overrides default).
            xlabel (str): Label for the x-axis (optional, overrides default).
            ylabel (str): Label for the y-axis (optional, overrides default).

        Returns:
            None
        """
        plt.figure(figsize=figsize or self.figsize)
        plt.imshow(input_map, cmap=cmap, interpolation='nearest')
        plt.colorbar(label='Surprise Level')
        plt.title(title or self.title)
        plt.xlabel(xlabel or self.xlabel)
        plt.ylabel(ylabel or self.ylabel)
        plt.show()

    def visualize_histogram(self, data, title=None, bins=30, color='blue', alpha=0.7, 
                            figsize=None, xlabel=None, ylabel=None, edgecolor='black'):
        """
        Visualize a histogram of the given data using matplotlib.

        Args:
            data (np.ndarray): A 1D or flattened 2D array of data values to visualize in a histogram.
            title (str): Title of the histogram (optional, overrides default).
            bins (int): Number of bins in the histogram.
            color (str): Color of the histogram bars.
            alpha (float): Transparency level of the histogram bars.
            figsize (tuple): Size of the figure (optional, overrides default).
            xlabel (str): Label for the x-axis (optional, overrides default).
            ylabel (str): Label for the y-axis (optional, overrides default).
            edgecolor (str): Color of the edge of the bars.

        Returns:
            None
        """
        plt.figure(figsize=figsize or self.figsize)
        plt.hist(data.flatten(), bins=bins, density=True, alpha=alpha, color=color, edgecolor=edgecolor)
        plt.title(title or self.title)
        plt.xlabel(xlabel or self.xlabel)
        plt.ylabel(ylabel or self.ylabel)
        plt.show()


    def visualize_feature_maps(self, feature_maps, feature_name):
        # num_maps = feature_tensor.size(0)  # Number of channels
        num_maps = feature_maps.shape[1]  # Number of channels
        cols = int(np.ceil(np.sqrt(num_maps)))  # Calculate columns for a square layout
        rows = (num_maps + cols - 1) // cols  # Calculate rows needed

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        for i in range(rows * cols):
            ax = axes.flat[i] if rows * cols != 1 else axes 
            if i < num_maps:
                ax.imshow(feature_maps[0, i].detach().cpu().numpy(), 
                          cmap='cool', aspect='auto') # other options: 'hot', 'cool', 'cividis', 'gray'
                # ax.set_title(f'{feature_name} {i+1}')
                ax.set_title(f'Channel {i + 1}')
                ax.axis('off')
            else:
                ax.axis('off')  # Hide unused subplots
        plt.title(feature_name)
        plt.tight_layout()
        plt.show()  
