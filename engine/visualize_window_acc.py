import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# Add project root to Python path for model loading
script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
results_dir = os.path.join(project_root, 'results')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

image_size = 1000
center_x = image_size // 2
center_y = image_size // 2
window_color = (0, 255, 0)

# New accuracy definition: a prediction is accurate iff it lies inside a disk
# centered at the target with radius r = (window_height + window_width) / 2.
# With a square image (width == height), r simplifies to window_size * image_size.
def draw_window_disk(window_size: float) -> np.ndarray:
    window_width = image_size * window_size
    window_height = image_size * window_size
    radius = int(round((window_width + window_height) / 2.0))
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    cv2.circle(image, (center_x, center_y), radius, window_color, thickness=-1)
    return image


image_window_acc_5 = draw_window_disk(0.05)
image_window_acc_2 = draw_window_disk(0.02)
image_window_acc_1 = draw_window_disk(0.01)
image_window_acc_0_5 = draw_window_disk(0.005)

# save the images
window_acc_dir = os.path.join(results_dir, 'window_acc')
os.makedirs(window_acc_dir, exist_ok=True)
cv2.imwrite(os.path.join(window_acc_dir, 'image_window_acc_5.png'), image_window_acc_5)
cv2.imwrite(os.path.join(window_acc_dir, 'image_window_acc_2.png'), image_window_acc_2)
cv2.imwrite(os.path.join(window_acc_dir, 'image_window_acc_1.png'), image_window_acc_1)
cv2.imwrite(os.path.join(window_acc_dir, 'image_window_acc_0_5.png'), image_window_acc_0_5)
