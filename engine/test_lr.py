
import os
import sys
import csv
import copy
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
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    

from gwtd.nets import nn
from gwtd.utils import util
from gwtd.utils import training_utils

# warnings.filterwarnings("ignore")
data_dir = '/home/jaehun/YOLOv11-pt-master/datasets/coco'


def train(config):
    # create the model
    name_backbone = config['backbone']
    pretrained_weights_path = os.path.join(project_root, 'weights', f'{name_backbone}.pt')

    # setup
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    accumulate = config['training']['accumulate']
    unfreeze_backbone_epochs = config['training']['unfreeze_backbone_epochs']
    is_backbone_unfrozen = False

    # prepare dataset and loader
    dataset_path = os.path.join(project_root, 'datasets', 'guidewire')
    data_preprocessor = GuidewireDataPreprocessor(dir_dataset=dataset_path,
                                                  split_ratio=config['dataset']['split_ratio'])
    train_dataset = GuidewireDataSet(data_preprocessor.get_data_sample_names('train'),
                                     apply_augmentation=True, config=config)
    val_dataset = GuidewireDataSet(data_preprocessor.get_data_sample_names('val'),
                                     apply_augmentation=False, config=config)
    print (f"number of total samples: {data_preprocessor.n_total_samples}")
    print (f"number of train samples: {len(train_dataset)}")
    print (f"number of val samples: {len(val_dataset)}")

    # set num workers to # of cores - 2
    loader = data.DataLoader(train_dataset, config['training']['batch_size'],
                             shuffle=True, num_workers=os.cpu_count() - 2, pin_memory=True,
                             collate_fn=GuidewireDataSet.collate_fn)
    val_loader = data.DataLoader(val_dataset, config['training']['batch_size'],
                                 shuffle=False, num_workers=os.cpu_count() - 2, pin_memory=True,
                                 collate_fn=GuidewireDataSet.collate_fn)
    num_steps_per_epoch = len(loader)
    print (f"number of steps per epoch: {num_steps_per_epoch}")

    # Scheduler
    config['training']['lr_scheduler']['args']['unfreeze_backbone_epochs'] = unfreeze_backbone_epochs
    scheduler = training_utils.generate_lr_scheduler(epochs, num_steps_per_epoch, config['training']['lr_scheduler'])

    results_dir = os.path.join(project_root, 'results', config['config_name'])
    os.makedirs(results_dir, exist_ok=True)

    # plot the lr scheduler
    training_utils.plot_lr_scheduler(scheduler, os.path.join(results_dir, 'lr_scheduler.png'))

    return


def profile_model(model, input_image_shape):
    model.eval()
    # Add batch and color channel dimension: [H, W] -> [1, 1, H, W]
    batch_input_shape = (1, 1) + tuple(input_image_shape)
    
    # Print total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    x = torch.randn(batch_input_shape).cuda()  # Move input to GPU
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_flops=True
    ) as prof:
        model(x)
    print(prof.key_averages().table(sort_by="flops", row_limit=10))
    return


def main():
    print('Start testing...')
    parser = ArgumentParser()
    # here, the config path is relative to the project root / config folder
    parser.add_argument('--config', default='default.yaml', type=str)


    args = parser.parse_args()

    config_path = os.path.join(project_root, 'config', args.config)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config_name = args.config.split('.')[0]
    config['config_name'] = config_name

    print('config loaded. number of keys: %d' % len(config.keys()))

    util.setup_multi_processes()
    # set seed and deterministic behavior
    seed_value = config.get('seed', 0)
    util.setup_seed(seed_value)

    train(config)

if __name__ == "__main__":
    main()
