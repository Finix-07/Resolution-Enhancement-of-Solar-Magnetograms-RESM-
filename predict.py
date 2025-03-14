import torch
from PIL import Image
from torchvision import transforms as transform
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


# Predict function
def prediction(model, image_path):
    model.eval()
    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0).to('cuda')
    with torch.no_grad():
        output = model(image)
    return output.squeeze(0).cpu()


def scale_to_target_range(data, target_min=0, target_max=0.91):
    min_val, max_val = np.min(data), np.max(data)
    scaled_data = (data - min_val) / (max_val - min_val) * (target_max - target_min) + target_min
    return scaled_data