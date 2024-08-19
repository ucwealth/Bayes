import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from PIL import Image
import matplotlib

print("Matplotlib version:", matplotlib.__version__)
print("PyTorch version:", torch.__version__)
print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)
# print("Matplotlib version:", plt.__version__)
print("Scikit-Learn version:", sklearn.__version__)
print("PIL version:", Image.__version__)

# Metal
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")