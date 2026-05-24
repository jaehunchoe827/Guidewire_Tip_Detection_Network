"""Apply each augmentation listed in `gwtd/config/default.yaml` to a single
training image and save the result to disk.

Run from the repo root:
    python test/test_augment.py [--config default.yaml] [--index 0]

For every augmentation, the *maximum* value of its configured range is used
(e.g. brightness_factor_range: [-0.2, 0.2] -> 0.2).  No random sampling is
performed for the parameter itself; a fixed numpy seed is set so that the
remaining randomness inside an augmentation (e.g. cutout positions) is
reproducible.

Outputs are written to:
    results/aug_test_dump/<image_stem>/
with one PNG per augmentation, plus the original (resized) image for
comparison.  Each PNG is annotated with a red crosshair at the tip and a
magenta circle matching the configured heatmap sigma.
"""
import argparse
import copy
import os
import sys

import cv2
import numpy as np
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from gwtd.augmentation import pixel_coords as aug_functions
from gwtd.data_loader.guidewire_data_loader import GuidewireDataPreprocessor


# ---------------------------------------------------------------------------
# Per-augmentation runners. Each entry returns (image, coords) given the raw
# RGB image, normalized coords and the augmentation's `args` dict from yaml.
# Inside each runner we use the *maximum* of any *_range field.
# ---------------------------------------------------------------------------

def _run_brightness(image, coords, args):
    factor = args['brightness_factor_range'][1]
    return aug_functions.augment_brightness(image, coords, factor)


def _run_horizontal_flip(image, coords, args):
    return aug_functions.augment_horizontal_flip(image, coords)


def _run_vertical_flip(image, coords, args):
    return aug_functions.augment_vertical_flip(image, coords)


def _run_scale_intensity(image, coords, args):
    factor = args['scale_factor_range'][1]
    return aug_functions.augment_scale_intensity(image, coords, factor)


def _run_gamma(image, coords, args):
    gamma = args['gamma_range'][1]
    return aug_functions.augment_gamma(image, coords, gamma)


def _run_saturation(image, coords, args):
    factor = args['saturation_factor_range'][1]
    return aug_functions.augment_saturation(image, coords, factor)


def _run_hue_shift(image, coords, args):
    hue_shift = args['hue_shift_range'][1]
    return aug_functions.augment_hue_shift(image, coords, hue_shift)


def _run_gaussian_sharpness(image, coords, args):
    sigma = args['sigma_range'][1]
    kernel_size = args['kernel_size']
    return aug_functions.augment_gaussian_sharpness(image, coords, kernel_size, sigma)


def _run_motion_blur(image, coords, args):
    kernel_size = int(args['kernel_size_range'][1])
    if kernel_size % 2 == 0:
        kernel_size += 1
    angle_deg = float(args['angle_range'][1])
    return aug_functions.augment_motion_blur(image, coords, kernel_size, angle_deg)


def _run_elastic_deformation(image, coords, args):
    alpha = args['alpha_range'][1]
    sigma = args.get('sigma', 1.0)
    return aug_functions.augment_elastic_deformation(image, coords, alpha, sigma)


def _run_perspective(image, coords, args):
    factor = args['perspective_factor_range'][1]
    return aug_functions.augment_perspective(image, coords, factor)


def _run_resize(image, coords, args):
    scale_w = args['width_range'][1]
    scale_h = args['height_range'][1]
    return aug_functions.augment_resize(image, coords, (scale_w, scale_h))


def _run_shear(image, coords, args):
    shear_x = args['shear_x_range'][1]
    shear_y = args['shear_y_range'][1]
    return aug_functions.augment_shear(image, coords, shear_x, shear_y)


def _run_rotation(image, coords, args):
    angle = args['angle_range'][1]
    return aug_functions.augment_rotation(image, coords, angle)


def _run_crop(image, coords, args, crop_size):
    safe = args.get('safe_reigon', 0.1)
    return aug_functions.augment_random_crop(image, coords, crop_size=crop_size, safe_reigon=safe)


def _run_mosaic(image, coords, args):
    mosaic_size = args['mosaic_size_range'][1]
    num_mosaics = args.get('max_num_mosaics', 2)
    return aug_functions.augment_mosaic(image, coords, mosaic_size, num_mosaics)


def _run_cutout(image, coords, args):
    cutout_size = args['cutout_size_range'][1]
    num_cutouts = args.get('max_num_cutouts', 2)
    return aug_functions.augment_cutout(image, coords, cutout_size, num_cutouts)


def _run_gaussian_noise(image, coords, args):
    sigma = args['sigma_range'][1]
    return aug_functions.augment_gaussian_noise(image, coords, sigma)


AUG_RUNNERS = {
    'random_brightness': _run_brightness,
    'random_horizontal_flip': _run_horizontal_flip,
    'random_vertical_flip': _run_vertical_flip,
    'random_scale_intensity': _run_scale_intensity,
    'random_gamma': _run_gamma,
    'random_saturation': _run_saturation,
    'random_hue_shift': _run_hue_shift,
    'random_gaussian_sharpness': _run_gaussian_sharpness,
    'random_motion_blur': _run_motion_blur,
    'random_elastic_deformation': _run_elastic_deformation,
    'random_perspective': _run_perspective,
    'random_resize': _run_resize,
    'random_shear': _run_shear,
    'random_rotation': _run_rotation,
    'random_crop': _run_crop,
    'random_mosaic': _run_mosaic,
    'random_cutout': _run_cutout,
    'random_gaussian_noise': _run_gaussian_noise,
}


# ---------------------------------------------------------------------------
# Loading and saving helpers
# ---------------------------------------------------------------------------

def load_sample(image_path, label_path, initial_resize_ratio):
    """Mirror the pre-augmentation portion of GuidewireDataSet.__getitem__:
    BGR -> RGB float32 in [0, 1], normalized (x, y) tip, initial resize.
    """
    image_cv = cv2.imread(image_path).astype(np.float32)
    image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB) / 255.0
    height, width = image.shape[:2]

    with open(label_path, 'r') as f:
        info = f.readlines()[0].split(' ')
    coords = np.array([
        int(info[1]) / (width - 1),
        int(info[2]) / (height - 1),
    ], dtype=np.float32)

    new_size = (round(width * initial_resize_ratio),
                round(height * initial_resize_ratio))
    image = cv2.resize(image, new_size)
    return image, coords


def annotate_and_save(image_rgb_float, coords, sigma, out_path):
    """Save an RGB float [0,1] image as PNG with a red crosshair at coords
    and a magenta circle of radius `sigma`.
    """
    img = np.clip(image_rgb_float, 0.0, 1.0)
    img_u8 = (img * 255.0).astype(np.uint8)
    if img_u8.ndim == 2:
        bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    elif img_u8.shape[2] == 1:
        bgr = cv2.cvtColor(img_u8[..., 0], cv2.COLOR_GRAY2BGR)
    else:
        bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)

    H, W = bgr.shape[:2]
    x = float(coords[0]) * (W - 1)
    y = float(coords[1]) * (H - 1)
    if 0 <= x <= W - 1 and 0 <= y <= H - 1:
        cv2.drawMarker(bgr, (int(round(x)), int(round(y))),
                       color=(0, 0, 255), markerType=cv2.MARKER_CROSS,
                       markerSize=24, thickness=2)
        cv2.circle(bgr, (int(round(x)), int(round(y))),
                   radius=int(round(sigma)),
                   color=(255, 0, 255), thickness=1)
        in_bounds = 'in'
    else:
        in_bounds = 'OUT'

    cv2.imwrite(out_path, bgr)
    return bgr.shape, x, y, in_bounds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='default.yaml',
                        help='Config file under gwtd/config/')
    parser.add_argument('--index', type=int, default=0,
                        help='Index into the *train* split to augment')
    parser.add_argument('--seed', type=int, default=0,
                        help='Numpy seed for the augmentations that retain '
                             'inner randomness (cutout/mosaic positions, '
                             'elastic-deformation displacement, '
                             'scale-intensity reference, etc.)')
    args = parser.parse_args()

    config_path = os.path.join(REPO_ROOT, 'gwtd', 'config', args.config)
    if not config_path.endswith('.yaml'):
        config_path += '.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    aug_cfg = config['dataset']['augmentation']
    initial_resize_ratio = config['dataset']['image_initial_resize_ratio']
    sigma = float(config['dataset']['heatmap_sigma'])
    # crop_size is (width, height) per augment_random_crop's contract.
    input_h, input_w = config['network']['input_image_shape']
    crop_size = (input_w, input_h)

    dataset_dir = os.path.join(REPO_ROOT, 'datasets', 'guidewire')
    pre = GuidewireDataPreprocessor(dataset_dir, config['dataset']['split_ratio'])
    train_samples = pre.get_data_sample_names('train')
    if not train_samples:
        raise RuntimeError(f'No training samples found under {dataset_dir}')

    index = max(0, min(args.index, len(train_samples) - 1))
    image_path, label_path = train_samples[index]
    image_stem = os.path.splitext(os.path.basename(image_path))[0]

    out_dir = os.path.join(REPO_ROOT, 'results', 'aug_test_dump', image_stem)
    os.makedirs(out_dir, exist_ok=True)

    image, coords = load_sample(image_path, label_path, initial_resize_ratio)
    print(f'Loaded train index {index}: {os.path.relpath(image_path, REPO_ROOT)}')
    print(f'  shape={image.shape}, tip(norm)=({coords[0]:.4f},{coords[1]:.4f})')
    print(f'  output dir: {os.path.relpath(out_dir, REPO_ROOT)}')

    original_path = os.path.join(out_dir, '00_original.png')
    annotate_and_save(image, coords, sigma, original_path)

    summary_lines = []
    for i, (aug_name, aug_params) in enumerate(aug_cfg.items(), start=1):
        runner = AUG_RUNNERS.get(aug_name)
        if runner is None:
            print(f'  [skip] no runner registered for {aug_name!r}')
            continue

        np.random.seed(args.seed)

        params = copy.deepcopy(aug_params.get('args') or {})
        if aug_name == 'random_crop':
            aug_image, aug_coords = runner(image.copy(), coords.copy(),
                                           params, crop_size)
        else:
            aug_image, aug_coords = runner(image.copy(), coords.copy(), params)

        out_path = os.path.join(out_dir, f'{i:02d}_{aug_name}.png')
        shape, x, y, in_bounds = annotate_and_save(aug_image, aug_coords,
                                                   sigma, out_path)
        line = (f'  [{i:02d}] {aug_name:<28s} shape={shape}  '
                f'tip=({x:7.1f},{y:7.1f})  {in_bounds}')
        print(line)
        summary_lines.append(line)

    summary_path = os.path.join(out_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f'source: {image_path}\n')
        f.write(f'label : {label_path}\n')
        f.write(f'initial_resize_ratio: {initial_resize_ratio}\n')
        f.write(f'normalized_tip: ({float(coords[0]):.4f}, {float(coords[1]):.4f})\n')
        f.write(f'heatmap_sigma: {sigma}\n')
        f.write('---\n')
        f.write('\n'.join(summary_lines) + '\n')


if __name__ == '__main__':
    main()
