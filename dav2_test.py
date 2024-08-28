import cv2
import torch

from DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vits' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'DepthAnythingV2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

raw_img = cv2.imread('data/dnerf/jumpingjacks/test/r_000.png')
print(raw_img.shape)
exit()
depth = model.infer_image(raw_img) # HxW raw depth map in numpy
import numpy as np
depth = 1.-(depth - np.min(depth)) / (np.max(depth) - np.min(depth))

import matplotlib.pyplot as plt
plt.imshow(depth, cmap='gray')
plt.colorbar()  # Optional: Add a color bar to the side
plt.show()