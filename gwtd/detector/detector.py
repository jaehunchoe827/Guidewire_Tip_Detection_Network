import os
import sys
import cv2
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
        max_batch_size: int = 8,
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

    def predict(
        self,
        images,
        gray_conversion_method: str = 'opencv',
    ) -> Dict[str, np.ndarray]:
        """
        Predict the guidewire tip from a list of images.

        The preprocessing pipeline resizes each frame to the network input
        resolution on the CPU with cv2.resize (cheap for uint8) and then runs
        float-scaling, BGR->gray, noise injection, and standardization on the
        GPU.

        Args:
            images: Either
                - List[np.ndarray] of BGR images (uint8, or float in [0, 1]),
                - a single (H, W, 3) BGR image,
                - an (B, H, W, 3) BGR batch.
                Images may have different (H, W); each is resized to
                self.input_image_shape independently.
            gray_conversion_method: str -> how to collapse BGR to a single channel
                before feeding the model.
                - 'opencv': BT.601 luma, Y = 0.114*B + 0.587*G + 0.299*R
                  (matches cv2.cvtColor(..., cv2.COLOR_BGR2GRAY)).
                - 'mean':   simple arithmetic mean across B, G, R channels.
        Returns:
            Dict[str, np.ndarray] -> dictionary of predictions
            'heatmaps': predicted probability heatmaps
            'tip_positions': peak pixel positions of the predicted probability
                heatmaps, normalized (x, y) in [0, 1].
            'gray_noised': per-image gray + gaussian-noise arrays at the
                NETWORK INPUT resolution, float32 in [0, 1], shape (B, H, W).
                Useful for visualizing what was fed to the network.
        """
        use_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')

        target_h, target_w = self.input_image_shape

        # Normalize the input container to a python list of (H, W, 3) arrays.
        if isinstance(images, np.ndarray):
            if images.ndim == 3:
                image_list = [images]
            elif images.ndim == 4:
                image_list = list(images)
            else:
                raise ValueError(f"Invalid image dimensions: {images.ndim}")
        else:
            image_list = list(images)
        if len(image_list) == 0:
            raise ValueError("predict() received no images")

        # Resize each image to the network input resolution on CPU (cv2 is fast for uint8).
        resized: List[np.ndarray] = []
        for img in image_list:
            if img.ndim != 3 or img.shape[-1] != 3:
                raise ValueError(
                    f"Expected (H, W, 3) BGR image, got shape {img.shape}"
                )
            if img.shape[0] != target_h or img.shape[1] != target_w:
                img = cv2.resize(
                    img, (target_w, target_h), interpolation=cv2.INTER_LINEAR
                )
            resized.append(img)
        batch_np = np.stack(resized, axis=0)  # (B, H, W, 3)

        batch = torch.from_numpy(batch_np).to(device, non_blocking=True)

        # Float [0, 1] in (B, 3, H, W) on device.
        if batch.dtype == torch.uint8:
            batch = batch.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
        else:
            batch = batch.permute(0, 3, 1, 2).contiguous().float()
            if float(batch.max().item()) > 1.0:
                batch = batch.div_(255.0)

        # BGR -> gray, (B, 1, H, W).
        if gray_conversion_method == 'opencv':
            # BT.601 luma for BGR input: 0.114 B + 0.587 G + 0.299 R
            bgr_weights = torch.tensor(
                [0.114, 0.587, 0.299], dtype=batch.dtype, device=device,
            ).view(1, 3, 1, 1)
            gray = (batch * bgr_weights).sum(dim=1, keepdim=True)
        elif gray_conversion_method == 'mean':
            gray = batch.mean(dim=1, keepdim=True)
        else:
            raise ValueError(
                f"Unknown gray_conversion_method: {gray_conversion_method!r} "
                f"(expected 'opencv' or 'mean')"
            )

        # torch.randn is the GPU analogue of np.random.normal. Distribution
        # matches (Gaussian with sigma = self.sigma_xray_noise), but
        # individual samples differ because the RNG source is different.
        noise = torch.randn_like(gray) * self.sigma_xray_noise
        gray_noised = (gray + noise).clamp_(0.0, 1.0)

        gray_noised_vis = gray_noised.squeeze(1)  # (B, H, W) on device

        images_std = (gray_noised - float(IMAGE_MEAN[0])) / float(IMAGE_STD[0])

        preds_list = []
        for i in range(0, images_std.shape[0], self.max_batch_size):
            sub_batch = images_std[i:i + self.max_batch_size]
            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type):
                    preds_list.append(self.model(sub_batch))
        preds = torch.cat(preds_list, dim=0)

        if self.config['from_logits']:
            preds = torch.sigmoid(preds)

        B, H, W = preds.shape
        flat = preds.view(B, -1)
        peak_indices_gpu = flat.argmax(dim=1)  # (B,)

        preds_np = preds.detach().cpu().numpy()
        gray_noised_out = gray_noised_vis.detach().cpu().numpy()
        peak_indices = peak_indices_gpu.detach().cpu().numpy()

        peak_y = (peak_indices // W).astype(np.float32) / H
        peak_x = (peak_indices % W).astype(np.float32) / W
        peak_positions = np.stack([peak_x, peak_y], axis=1)  # (B, 2) normalized (x, y)

        return {
            'heatmaps': preds_np,
            'tip_positions': peak_positions,
            'gray_noised': gray_noised_out,
        }

