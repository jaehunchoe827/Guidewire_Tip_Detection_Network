"""Dump augmented samples for visual inspection.

Run from the repo root:
    /home/mag22/miniconda3/envs/gwtd/bin/python scripts/sanity_dump_augs.py

Produces three sets under results/aug_sanity_dump/:
  - mixed/             : 30 samples with the full YAML augmentation chain
  - motion_blur_only/  : 10 samples with ONLY random_motion_blur (p=1.0)
  - gamma_only/        : 10 samples with ONLY random_gamma (p=1.0)

Each output PNG is annotated with a red crosshair at the (post-augmentation)
tip location and a magenta circle matching the heatmap sigma.
"""
import copy
import os
import sys
import cv2
import numpy as np
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, REPO_ROOT)

from gwtd.data_loader.guidewire_data_loader import (
    GuidewireDataPreprocessor,
    GuidewireDataSet,
)


def annotate_and_save(image, label, sigma, out_path):
    """Convert (H, W, 1) float [0,1] image to BGR PNG with tip annotation."""
    if image.ndim == 3 and image.shape[2] == 1:
        img_u8 = np.clip(image[..., 0] * 255.0, 0, 255).astype(np.uint8)
    else:
        img_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)

    H, W = img_u8.shape[:2]
    bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    x = float(label[0]) * (W - 1)
    y = float(label[1]) * (H - 1)
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
    return img_u8.shape, x, y, in_bounds


def dump_set(label, dataset, indices, sigma, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f'\n=== {label} -> {os.path.relpath(out_dir, REPO_ROOT)} ===')
    for k, idx in enumerate(indices):
        image, lbl = dataset.__getitem__(int(idx), get_heatmap=False)
        out_path = os.path.join(out_dir, f'{label}_{k:02d}_idx{int(idx):05d}.png')
        shape, x, y, in_bounds = annotate_and_save(image, lbl, sigma, out_path)
        print(f'  [{k:02d}] idx={int(idx):>5d}  shape={shape}  '
              f'tip=({x:6.1f},{y:6.1f})  {in_bounds}')


def isolate_augmentation(base_config, aug_name):
    """Return a deep-copied config whose only enabled aug is `aug_name` at p=1.0."""
    cfg = copy.deepcopy(base_config)
    aug_cfg = cfg['dataset']['augmentation']
    if aug_name not in aug_cfg:
        raise KeyError(f'{aug_name!r} not present in base config augmentation block')
    isolated = {aug_name: copy.deepcopy(aug_cfg[aug_name])}
    isolated[aug_name]['probability'] = 1.0
    cfg['dataset']['augmentation'] = isolated
    return cfg


def main():
    config_path = os.path.join(REPO_ROOT, 'gwtd', 'config', 'ver3_default.yaml')
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    dataset_dir = os.path.join(REPO_ROOT, 'datasets', 'guidewire')
    base_out = os.path.join(REPO_ROOT, 'results', 'aug_sanity_dump')
    os.makedirs(base_out, exist_ok=True)

    pre = GuidewireDataPreprocessor(dataset_dir, base_config['dataset']['split_ratio'])
    train_names = pre.get_data_sample_names('train')

    sigma = float(base_config['dataset']['heatmap_sigma'])
    rng = np.random.default_rng(0)

    # 1) Full pipeline (30 samples)
    mixed_indices = rng.choice(len(train_names), size=min(30, len(train_names)), replace=False)
    mixed_dataset = GuidewireDataSet(
        train_names,
        apply_augmentation=True,
        config=base_config,
        apply_standardization=False,
    )
    dump_set('mixed', mixed_dataset, mixed_indices, sigma,
             os.path.join(base_out, 'mixed'))

    # Use a fresh, smaller index set for the isolated-aug dumps.
    iso_indices = rng.choice(len(train_names), size=min(10, len(train_names)), replace=False)

    # 2) Motion blur only (10 samples)
    mb_config = isolate_augmentation(base_config, 'random_motion_blur')
    mb_dataset = GuidewireDataSet(
        train_names,
        apply_augmentation=True,
        config=mb_config,
        apply_standardization=False,
    )
    dump_set('motion_blur', mb_dataset, iso_indices, sigma,
             os.path.join(base_out, 'motion_blur_only'))

    # 3) Gamma only (10 samples)
    gamma_config = isolate_augmentation(base_config, 'random_gamma')
    gamma_dataset = GuidewireDataSet(
        train_names,
        apply_augmentation=True,
        config=gamma_config,
        apply_standardization=False,
    )
    dump_set('gamma', gamma_dataset, iso_indices, sigma,
             os.path.join(base_out, 'gamma_only'))


if __name__ == '__main__':
    main()
