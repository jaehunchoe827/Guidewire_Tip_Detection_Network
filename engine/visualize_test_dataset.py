import os
import sys
import random
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle

script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gwtd.data_loader.guidewire_data_loader import GuidewireDataPreprocessor, GuidewireDataSet
from gwtd.utils.standardization import destandardize_image
from engine.main import build_test_model


COL_GAP = 3
ROW_GAP = 3


# black -> green colormap for the heatmap column
HEATMAP_CMAP = LinearSegmentedColormap.from_list('black_green', [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)])


def load_raw_image_resized(image_path: str, output_size: tuple) -> np.ndarray:
    """Read the jpg and resize to the network output size (H, W). Returns RGB in [0, 1]."""
    image_bgr = cv2.imread(image_path).astype(np.float32)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) / 255.0
    height, width = output_size
    # cv2.resize expects (width, height)
    image_rgb = cv2.resize(image_rgb, (width, height))
    return np.clip(image_rgb, 0.0, 1.0)


def peak_xy(heatmap: np.ndarray) -> tuple:
    """Return (x, y) pixel coordinates of the argmax of a 2D heatmap."""
    y, x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    return x, y


def predict_heatmap(model, image_std: np.ndarray, from_logits: bool) -> np.ndarray:
    """Run the model on a standardized (H, W, 1) input and return a normalized
    heatmap in [0, 1] (peak pixel = 1)."""
    x = torch.from_numpy(image_std).permute(2, 0, 1).unsqueeze(0).float().cuda()
    with torch.no_grad():
        out = model(x)
    out = out.squeeze(0).float().cpu()
    if from_logits:
        out = torch.sigmoid(out)
    heatmap = out.numpy()
    peak = heatmap.max()
    if peak > 0:
        heatmap = heatmap / peak
    return heatmap


def draw_circle(ax, center_xy: tuple, color: str, radius: float = 12.0) -> None:
    ax.add_patch(Circle(center_xy, radius=radius, fill=False,
                        edgecolor=color, linewidth=2.5))


def select_samples(test_samples: list, num_optical: int, num_xray: int, seed: int) -> list:
    optical = [s for s in test_samples if not GuidewireDataSet.is_xray_image(s[0])]
    xray = [s for s in test_samples if GuidewireDataSet.is_xray_image(s[0])]
    if len(optical) < num_optical or len(xray) < num_xray:
        raise ValueError(
            f"Not enough samples: optical={len(optical)} (need {num_optical}), "
            f"xray={len(xray)} (need {num_xray})")
    rng = random.Random(seed)
    selected = [(s, 'optical') for s in rng.sample(optical, num_optical)]
    selected += [(s, 'x-ray') for s in rng.sample(xray, num_xray)]
    return selected


def visualize(config: dict, config_name: str, num_optical: int, num_xray: int,
              seed: int) -> str:
    output_size = tuple(config['network']['input_image_shape'])  # (H, W)
    from_logits = config.get('from_logits', True)

    dataset_path = os.path.join(project_root, 'datasets', 'guidewire')
    preprocessor = GuidewireDataPreprocessor(dir_dataset=dataset_path,
                                             split_ratio=config['dataset']['split_ratio'])
    test_samples = preprocessor.get_data_sample_names('test')
    selected = select_samples(test_samples, num_optical, num_xray, seed)

    # Dataset that reproduces exactly what the network is fed at test time
    # (skip_xray_default_noise=True: x-ray images are not given synthetic noise).
    selected_paths = [s for s, _ in selected]
    input_dataset = GuidewireDataSet(selected_paths, apply_augmentation=False,
                                     config=config, skip_xray_default_noise=True)

    model = build_test_model(config)
    model.eval()

    n_rows = len(selected)
    n_cols = 4
    cell_in = 3.0  # size of each image cell, in inches
    col_gap_in = COL_GAP / 72.0
    row_gap_in = ROW_GAP / 72.0
    fig_w = n_cols * cell_in + (n_cols - 1) * col_gap_in
    fig_h = n_rows * cell_in + (n_rows - 1) * row_gap_in
    fig = plt.figure(figsize=(fig_w, fig_h))

    def make_ax(row_index, col_index):
        left = col_index * (cell_in + col_gap_in)
        bottom = fig_h - (row_index + 1) * cell_in - row_index * row_gap_in
        ax = fig.add_axes([left / fig_w, bottom / fig_h,
                           cell_in / fig_w, cell_in / fig_h])
        ax.axis('off')
        return ax

    for row_index, ((image_path, label_path), modality) in enumerate(selected):
        raw_resized = load_raw_image_resized(image_path, output_size)

        image_std, gt_heatmap = input_dataset[row_index]  # (H, W, 1), (H, W)
        gt_xy = peak_xy(gt_heatmap)

        pred_heatmap = predict_heatmap(model, image_std, from_logits)
        pred_xy = peak_xy(pred_heatmap)

        # Column 2: what is shown depends on modality per the requested layout.
        if modality == 'x-ray':
            # x-ray: just the jpeg-converted image (no grayscale / noise), resized.
            col2_image = raw_resized
            col2_kwargs = {}
        else:
            # optical: the actual grayscale + noised input fed to the network.
            gray_noised = destandardize_image(image_std.copy()).squeeze(-1)
            col2_image = np.clip(gray_noised, 0.0, 1.0)
            col2_kwargs = {'cmap': 'gray', 'vmin': 0.0, 'vmax': 1.0}

        # Column 1: raw + GT tip (blue)
        ax = make_ax(row_index, 0)
        ax.imshow(raw_resized)
        draw_circle(ax, gt_xy, color='blue')

        # Column 2: network input
        ax = make_ax(row_index, 1)
        ax.imshow(col2_image, **col2_kwargs)

        # Column 3: predicted heatmap (black -> green)
        ax = make_ax(row_index, 2)
        ax.imshow(pred_heatmap, cmap=HEATMAP_CMAP, vmin=0.0, vmax=1.0)

        # Column 4: raw + predicted tip (green)
        ax = make_ax(row_index, 3)
        ax.imshow(raw_resized)
        draw_circle(ax, pred_xy, color='lime')

    output_dir = os.path.join(project_root, 'results', config_name, 'test')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'test_dataset_visualization.png')
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved visualization to {output_path}")
    return output_path


def main():
    parser = ArgumentParser(description='Visualize model predictions on test samples.')
    parser.add_argument('--config', default='default.yaml', type=str)
    parser.add_argument('--num_optical', default=3, type=int)
    parser.add_argument('--num_xray', default=2, type=int)
    parser.add_argument('--seed', default=6, type=int,
                        help='seed for selecting which test samples to visualize')
    args = parser.parse_args()

    config_path = os.path.join(project_root, 'gwtd', 'config', args.config)
    if not config_path.endswith('.yaml'):
        config_path = config_path + '.yaml'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config_name = args.config.split('.')[0]
    config['config_name'] = config_name
    print(f"Config: {args.config}")
    visualize(config, config_name, args.num_optical, args.num_xray, args.seed)


if __name__ == '__main__':
    main()
