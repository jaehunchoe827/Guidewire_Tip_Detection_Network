import os
import sys
import yaml
import torch
import numpy as np
from typing import Optional, Dict, List

# Add project root to Python path for model loading
script_dir = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(script_dir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gwtd.nets import nn
from gwtd.utils import util
from gwtd.utils.standardization import IMAGE_MEAN, IMAGE_STD

class GuidewireTipDetector:
    def __init__(
        self,
        config_file_name: str,
        max_batch_size: int = 20,
        sigma_xray_noise: Optional[float] = None,
    ):
        self.config_file_name = config_file_name
        self.sigma_xray_noise = sigma_xray_noise
        self.max_batch_size = max_batch_size
        self.config = self.load_config()
        self.model = None
        self.initialize()

    def load_config(self):
        config_file_path = os.path.join(PROJECT_ROOT, 'results', self.config_file_name, 'config.yaml')
        if not os.path.exists(config_file_path):
            raise FileNotFoundError(f"Config file not found: {config_file_path}")
        with open(config_file_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def initialize(self):
        # get gaussian noise if not provided
        if self.sigma_xray_noise is None:
            self.sigma_xray_noise = self.config['dataset']['sigma_xray_noise']

        # create the model
        name_backbone = self.config['backbone']
        backbone_weights_path = os.path.join(PROJECT_ROOT, 'weights', f'{name_backbone}.pt')
        if not os.path.exists(backbone_weights_path):
            raise FileNotFoundError(f"Backbone weights not found: {backbone_weights_path}")
        self.input_image_shape = self.config['network']['input_image_shape']
        model = nn.YOLOwithCustomHead(
            name_backbone,
            backbone_weights_path,
            self.config['network']['input_image_shape'],
            self.config['network']['head'],
            from_logits=self.config['from_logits']
        )
        model.cuda()
        model.eval() # to evaluation mode
        # load weights
        weights_path = os.path.join(
            PROJECT_ROOT, 'results', self.config['config_name'], 'best.pt'
        )
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"gwtd weights not found: {weights_path}")
        util.load_weight(model, weights_path)
        self.model = model

    def predict(self, images: List[np.ndarray], gray_conversion_method: str = 'opencv') -> Dict[str, np.ndarray]:
        """
        Predict the guidewire tip from a list of images.
        Args:
            images: List[np.ndarray] -> list of BGR color images (as produced by
                cv2.VideoCapture / cv2.imread). Values may be uint8 or float32
                in [0, 1]; either is accepted.
            gray_conversion_method: str -> how to collapse BGR to a single channel
                before feeding the model.
                - 'opencv': BT.601 luma, Y = 0.114*B + 0.587*G + 0.299*R
                  (matches cv2.cvtColor(..., cv2.COLOR_BGR2GRAY)).
                - 'mean':   simple arithmetic mean across B, G, R channels.
        Returns:
            Dict[str, np.ndarray] -> dictionary of predictions
            'heatmaps': predicted probability heatmaps
            'tip_positions': peak pixel positions of the predicted probability heatmaps
            'gray_noised': per-image gray + gaussian-noise arrays at the INPUT
                resolution, float32 in [0, 1], shape (B, H, W). Useful for
                visualizing what was fed to the network.
        """
        # normalize the image and convert to float32 if not already
        images = np.array(images, dtype=np.float32)
        if images.max() > 1.0:
            images /= 255.0
        if images.ndim == 3:
            # (H, W, C) — color axis is last
            if not images.shape[-1] == 3:
                raise ValueError(f"Expected color image with 3 channels, got {images.shape[-1]} channels")
            # convert to (1, C, H, W)
            images = images.transpose(2, 0, 1)[np.newaxis, :, :, :]
        elif images.ndim == 4:
            # (B, H, W, C) — color axis is last
            if not images.shape[-1] == 3:
                raise ValueError(f"Expected color image with 3 channels, got {images.shape[-1]} channels")
            # convert to (B, C, H, W)
            images = images.transpose(0, 3, 1, 2)
        else:
            raise ValueError(f"Invalid image dimensions: {images.ndim}")

        # images is now (B, 3, H, W), BGR ordering, float32 in [0, 1]

        # convert to gray -> (B, 1, H, W)
        if gray_conversion_method == 'opencv':
            # BT.601 luma for BGR input: 0.114 B + 0.587 G + 0.299 R
            bgr_weights = np.array([0.114, 0.587, 0.299], dtype=np.float32).reshape(1, 3, 1, 1)
            gray = (images * bgr_weights).sum(axis=1, keepdims=True)
        elif gray_conversion_method == 'mean':
            gray = images.mean(axis=1, keepdims=True)
        else:
            raise ValueError(
                f"Unknown gray_conversion_method: {gray_conversion_method!r} "
                f"(expected 'opencv' or 'mean')"
            )

        # apply gaussian noise
        noise = np.random.normal(0, self.sigma_xray_noise, gray.shape).astype(np.float32)
        gray_noised = np.clip(gray + noise, 0, 1).astype(np.float32)

        # keep a copy (B, H, W) for callers that want to visualize the network input
        gray_noised_out = gray_noised[:, 0, :, :].copy()

        images = (gray_noised - IMAGE_MEAN[0]) / IMAGE_STD[0]

        images = torch.from_numpy(images)
        target_h, target_w = self.input_image_shape
        if images.shape[2] != target_h or images.shape[3] != target_w:
            images = torch.nn.functional.interpolate(
                images, size=(target_h, target_w), mode='bilinear', align_corners=False
            )
        images = images.cuda()

        preds_list = []
        for i in range(0, images.shape[0], self.max_batch_size):
            sub_batch = images[i:i + self.max_batch_size]
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda'):
                    preds_list.append(self.model(sub_batch))

        preds = torch.cat(preds_list, dim=0)
        if self.config['from_logits']:
            preds = torch.sigmoid(preds)
        preds_np = preds.cpu().numpy()
        # calculate the peak pixel position of the predicted probability heatmaps
        # preds_np shape: (B, H, W)
        B, H, W = preds_np.shape
        flat = preds_np.reshape(B, -1)
        peak_indices = flat.argmax(axis=1)
        peak_y = (peak_indices // W).astype(np.float32) / H
        peak_x = (peak_indices % W).astype(np.float32) / W
        peak_positions = np.stack([peak_x, peak_y], axis=1)  # (B, 2) normalized (x, y)

        predictions = {
            'heatmaps': preds_np,
            'tip_positions': peak_positions,
            'gray_noised': gray_noised_out,
        }

        return predictions

