import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter1d
from matplotlib.patches import Patch

# Create a custom colormap for the states
colors = ['#c7cdd6',  # RESTING (grey), 0
          '#FF0000',  # DEPOLARIZED (red), 1
          '#fcba03',  # REPOLARIZED (yellow), 2
          '#a68fdb']  # REFRACTORY (purple), 3

custom_cmap = ListedColormap(colors)



