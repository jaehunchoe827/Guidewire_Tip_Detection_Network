import os
import sys
import csv
import copy
import time
import tqdm
import yaml
import torch
import random
import numpy as np
import warnings
from argparse import ArgumentParser
from torch.utils import data
from gwtd.data_loader.guidewire_data_loader import GuidewireDataPreprocessor, GuidewireDataSet
from gwtd.utils import training_utils
from gwtd.loss.loss import GuidewireHeatMapLoss

# Add project root to Python path for model loading
script_dir = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
print(f"PROJECT_ROOT: {PROJECT_ROOT}")

from gwtd.nets import nn
from gwtd.utils import util
from gwtd.utils import training_utils



def main():
    config_file_name = 'ver3_default_less_aug'

    config_file_path = os.path.join(PROJECT_ROOT, 'results', config_file_name, 'config.yaml')
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file not found: {config_file_path}")
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # create the model
    name_backbone = config['backbone']
    backbone_weights_path = os.path.join(PROJECT_ROOT, 'weights', f'{name_backbone}.pt')
    if not os.path.exists(backbone_weights_path):
        raise FileNotFoundError(f"Backbone weights not found: {backbone_weights_path}")
    input_image_shape = config['network']['input_image_shape']
    model = nn.YOLOwithCustomHead(
        name_backbone,
        backbone_weights_path,
        config['network']['input_image_shape'],
        config['network']['head'],
        from_logits=config['from_logits']
    )
    model.cuda()
    model.eval() # to evaluation mode
    # load weights
    weights_path = os.path.join(
        PROJECT_ROOT, 'results', config['config_name'], 'best.pt'
    )
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"gwtd weights not found: {weights_path}")
    util.load_weight(model, weights_path)

    H, W = input_image_shape
    num_runs = 50

    # warmup (first few CUDA calls have overhead)
    dummy = torch.randn(1, 1, H, W, device='cuda')
    for _ in range(10):
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda'):
                model(dummy)
    torch.cuda.synchronize()

    # --- batch size 1 ---
    batch1 = torch.randn(1, 1, H, W, device='cuda')
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_runs):
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda'):
                model(batch1)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    avg_bs1 = (t1 - t0) / num_runs * 1000  # ms

    # --- batch size 2 ---
    batch2 = torch.randn(2, 1, H, W, device='cuda')
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_runs):
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda'):
                model(batch2)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    avg_bs2 = (t1 - t0) / num_runs * 1000  # ms

    # --- batch size 3 ---
    batch3 = torch.randn(3, 1, H, W, device='cuda')
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_runs):
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda'):
                model(batch3)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    avg_bs3 = (t1 - t0) / num_runs * 1000  # ms

    # --- batch size 4 ---
    batch4 = torch.randn(4, 1, H, W, device='cuda')
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_runs):
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda'):
                model(batch4)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    avg_bs4 = (t1 - t0) / num_runs * 1000  # ms

    print(f"\n{'='*40}")
    print(f"  Input shape: (B, 1, {H}, {W})")
    print(f"  Runs per batch size: {num_runs}")
    print(f"{'='*40}")
    print(f"  Batch size 1: {avg_bs1:.3f} ms")
    print(f"  Batch size 2: {avg_bs2:.3f} ms  ({avg_bs2 / avg_bs1:.3f}x)")
    print(f"  Batch size 3: {avg_bs3:.3f} ms  ({avg_bs3 / avg_bs1:.3f}x)")
    print(f"  Batch size 4: {avg_bs4:.3f} ms  ({avg_bs4 / avg_bs1:.3f}x)")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    main()