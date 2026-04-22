
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


def validation(config, weights_epoch):
    # create the model
    name_backbone = config['backbone']
    backbone_weights_path = os.path.join(project_root, 'weights', f'{name_backbone}.pt')
    model = nn.YOLOwithCustomHead(name_backbone,
                                  backbone_weights_path,
                                  config['network']['input_image_shape'],
                                  config['network']['head'],
                                  from_logits=config['from_logits'])
    pretrained_weights_path = os.path.join(project_root, 'results', config['config_name'], f'epoch_{weights_epoch:04d}.pt')
    util.load_weight(model, pretrained_weights_path)
    model.cuda()
    profile_model(model, config['network']['input_image_shape'])

    # setup
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']

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

    # deterministic seeding for DataLoader workers and shuffling
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    base_seed = config.get('seed', 0)
    g = torch.Generator()
    g.manual_seed(base_seed)

    # set num workers to # of cores - 2
    loader = data.DataLoader(train_dataset, config['training']['batch_size'],
                             shuffle=True, num_workers=os.cpu_count() - 2, pin_memory=True,
                             collate_fn=GuidewireDataSet.collate_fn,
                             worker_init_fn=seed_worker, generator=g)
    val_loader = data.DataLoader(val_dataset, config['training']['batch_size'],
                                 shuffle=False, num_workers=os.cpu_count() - 2, pin_memory=True,
                                 collate_fn=GuidewireDataSet.collate_fn,
                                 worker_init_fn=seed_worker, generator=g)
    num_steps_per_epoch = len(loader)
    print (f"number of steps per epoch: {num_steps_per_epoch}")

    results_dir = os.path.join(project_root, 'results', config['config_name'])
    os.makedirs(results_dir, exist_ok=True)

    criterion = GuidewireHeatMapLoss(from_logits=config['from_logits'])

    # Validation at end of epoch
    model.eval()
    val_losses_sum = {}
    num_val_batches = max(1, len(val_loader))
    with torch.no_grad():
        for i, (x_val, y_val) in enumerate(val_loader):
            x_val = x_val.cuda(non_blocking=True)
            y_val = y_val.cuda(non_blocking=True)
            # for validation, we disable AMP to avoid precision issues
            with torch.amp.autocast(device_type='cuda', enabled=False):
                pred_val = model(x_val)
            val_losses = criterion(pred_val, y_val)
            # INSERT_YOUR_CODE
            # Save or print raw values of y_val and pred_val for inspection
            # For demonstration, this will print the numpy arrays of the first batch in validation
            if i == 0:
                print("Raw y_val (ground truth):", y_val.cpu().numpy())
                print("Raw pred_val (prediction):", pred_val.cpu().numpy())
            # Accumulate validation losses
            for loss_name, loss_value in val_losses.items():
                if loss_name not in val_losses_sum:
                    val_losses_sum[loss_name] = 0.0
                val_losses_sum[loss_name] += float(loss_value.item())
    
    # Calculate average validation losses
    val_losses_avg = {loss_name: loss_sum / num_val_batches 
                        for loss_name, loss_sum in val_losses_sum.items()}
    
    # Calculate weighted total validation loss
    val_loss_total = 0.0
    for loss_name in config['training']['loss_main']:
        if loss_name in val_losses_avg and loss_name in config['training']['loss_weights']:
            val_loss_total += config['training']['loss_weights'][loss_name] * val_losses_avg[loss_name]
    
    print(f"val_loss_total: {val_loss_total:.6f}, "
            f"val_acc5: {val_losses_avg.get('5%_win_acc', 0):.5f}, "
            f"val_acc1: {val_losses_avg.get('1%_win_acc', 0):.5f}")

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
    print('Start validation...')
    parser = ArgumentParser()
    # here, the config path is relative to the project root / config folder
    parser.add_argument('--config', default='default.yaml', type=str)
    parser.add_argument('--weights_epoch', default=1, type=int)

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

    validation(config, args.weights_epoch)

if __name__ == "__main__":
    main()
