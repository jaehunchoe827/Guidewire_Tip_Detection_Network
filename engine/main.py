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
script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gwtd.nets import nn
from gwtd.utils import util
from gwtd.utils import training_utils


def train(config):
    # create the model
    name_backbone = config['backbone']
    pretrained_weights_path = os.path.join(project_root, 'weights', f'{name_backbone}.pt')
    model = nn.YOLOwithCustomHead(name_backbone,
                                  pretrained_weights_path,
                                  config['network']['input_image_shape'],
                                  config['network']['head'],
                                  from_logits=config['from_logits'])
    model.cuda()
    model.freeze_backbone()
    profile_model(model, config['network']['input_image_shape'])

    # setup
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    accumulate = config['training']['accumulate']
    unfreeze_backbone_epochs = config['training']['unfreeze_backbone_epochs']
    max_grad_norm = config['training']['max_grad_norm']
    is_backbone_unfrozen = False

    # Optimizer
    optimizer = training_utils.generate_optimizer(model, config['training']['optimizer'])
    # param groups for gradient clipping
    params_to_clip = [p for g in optimizer.param_groups for p in g['params']]

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

    # Scheduler
    config['training']['lr_scheduler']['args']['unfreeze_backbone_epochs'] = unfreeze_backbone_epochs
    scheduler = training_utils.generate_lr_scheduler(epochs, num_steps_per_epoch, config['training']['lr_scheduler'])

    best_score = -float('inf') # set best score to minus infinity
    amp_scale = torch.amp.GradScaler()
    results_dir = os.path.join(project_root, 'results', config['config_name'])
    os.makedirs(results_dir, exist_ok=True)

    # plot the lr scheduler
    training_utils.plot_lr_scheduler(scheduler, os.path.join(results_dir, 'lr_scheduler.png'))

    # save the config
    with open(os.path.join(results_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    criterion = GuidewireHeatMapLoss(from_logits=config['from_logits'])
    
    # Train
    with open(os.path.join(results_dir, 'step.csv'), 'w') as log, \
         open(os.path.join(results_dir, 'val_loss.csv'), 'w') as val_log:
        # Get loss names dynamically from criterion
        dummy_output = torch.zeros(1, 1, 1, 1)  # Dummy tensor to get loss names
        dummy_target = torch.zeros(1, 1, 1, 1)
        dummy_losses = criterion(dummy_output, dummy_target)
        loss_names = list(dummy_losses.keys())
        
        # Define CSV headers dynamically
        csv_headers = ['epoch', 'step', 'lr', 'loss_total'] + loss_names
        writer = csv.writer(log)
        writer.writerow(csv_headers)
        
        # Define CSV headers for validation losses (will be set after first validation)
        val_writer = csv.writer(val_log)
        val_headers_written = False
        global_step = 0
        model.train()
        # initial learning rate step
        scheduler.step(global_step, optimizer)
        for epoch in range(1, epochs+1):
            optimizer.zero_grad()

            if epoch >= unfreeze_backbone_epochs and not is_backbone_unfrozen:
                model.unfreeze_backbone()
                is_backbone_unfrozen = True
                # the trainable parameters has been changed.
                # so we need to re-initialize the optimizer, gradient scaler, and param group
                optimizer = training_utils.generate_optimizer(model, config['training']['optimizer'])
                amp_scale = torch.amp.GradScaler()
                params_to_clip = [p for g in optimizer.param_groups for p in g['params']]
                optimizer.zero_grad()
                print(f"Backbone unfreezed at epoch {epoch}")
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f'Trainable parameters: {trainable_params:,}')

            p_bar = tqdm.tqdm(loader, total=num_steps_per_epoch,
                              desc=f"Epoch {epoch}/{epochs}", leave=True,
                              ncols = 110)
            for batch_index, (x, y) in enumerate(p_bar):
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)
                ## print shape of x and y
                with torch.amp.autocast(device_type='cuda'): # use float16 by default
                    prediction = model(x)
                    losses = criterion(prediction, y)
                
                # Calculate weighted total loss
                loss_total = 0.0
                for loss_name in config['training']['loss_main']:
                    if loss_name in losses and loss_name in config['training']['loss_weights']:
                        loss_total += config['training']['loss_weights'][loss_name] * losses[loss_name]
                
                # Scale loss for gradient accumulation (divide by accumulate)
                loss_total = loss_total / accumulate
                amp_scale.scale(loss_total).backward()

                # step on accumulation boundary
                if (batch_index + 1) % accumulate == 0 or (batch_index + 1) == num_steps_per_epoch:
                    # Unscale then clip gradients before stepping
                    amp_scale.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params_to_clip, max_grad_norm)
                    # lr update per every accumulation boundary
                    scheduler.step(global_step, optimizer)
                    amp_scale.step(optimizer)
                    amp_scale.update()
                    optimizer.zero_grad()

                # log
                current_lr = optimizer.param_groups[0]['lr']
                
                # Prepare CSV row with all losses
                csv_row = [epoch, global_step, current_lr, loss_total.item() * accumulate]
                for loss_name in loss_names:
                    if loss_name in losses:
                        csv_row.append(losses[loss_name].item())
                    else:
                        csv_row.append(0.0)  # Default value if loss not computed
                
                writer.writerow(csv_row)
                
                # Update progress bar with key metrics
                p_bar.set_postfix(
                    loss=f"{(loss_total.item() * accumulate):.5f}", 
                    lr=f"{current_lr:.5f}",
                    acc2=f"{losses['2%_win_acc'].item():.3f}",
                    dist=f"{losses['dist'].item():.3f}"
                )

                # update global step
                global_step += 1

            # Validation at end of epoch
            model.eval()
            val_losses_sum = {}
            num_val_batches = max(1, len(val_loader))
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val = x_val.cuda(non_blocking=True)
                    y_val = y_val.cuda(non_blocking=True)
                    # for validation, we disable AMP to avoid precision issues
                    with torch.amp.autocast(device_type='cuda'):
                        pred_val = model(x_val)
                    val_losses = criterion(pred_val, y_val)
                    
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
            
            print(f"Epoch {epoch}/{epochs} - val_loss_total: {val_loss_total:.6f}, "
                  f"val_acc2: {val_losses_avg.get('2%_win_acc', 0):.5f}, "
                  f"val_dist: {val_losses_avg.get('dist', 0):.5f}")

            # Log validation losses to CSV
            if not val_headers_written:
                # Write headers dynamically based on available losses
                val_csv_headers = ['epoch', 'val_loss_total'] + list(val_losses_avg.keys())
                val_writer.writerow(val_csv_headers)
                val_headers_written = True
            
            val_csv_row = [epoch, val_loss_total]
            for loss_name, loss_value in val_losses_avg.items():
                val_csv_row.append(loss_value)
            val_writer.writerow(val_csv_row)

            # Save best checkpoint based on total validation loss
            one_percent_win_acc = val_losses_avg.get('1%_win_acc', 0)
            if one_percent_win_acc > best_score:
                best_score = one_percent_win_acc
                ckpt = {
                    'model': model.state_dict(),
                    'epoch': epoch,
                    'score': best_score,
                    'config': config,
                }
                torch.save(ckpt, os.path.join(results_dir, 'best.pt'))

            model.train()
    return

def test(config):
    # create the model
    name_backbone = config['backbone']
    backbone_weights_path = os.path.join(project_root, 'weights', f'{name_backbone}.pt')
    model = nn.YOLOwithCustomHead(name_backbone,
                                  backbone_weights_path,
                                  config['network']['input_image_shape'],
                                  config['network']['head'],
                                  from_logits=config['from_logits'])
    model.cuda()
    # load weights
    weights_path = os.path.join(project_root, 'results', config['config_name'], 'best.pt')
    util.load_weight(model, weights_path)
    # prepare dataset and loader
    dataset_path = os.path.join(project_root, 'datasets', 'guidewire')
    data_preprocessor = GuidewireDataPreprocessor(dir_dataset=dataset_path,
                                                  split_ratio=config['dataset']['split_ratio'])
    test_dataset = GuidewireDataSet(data_preprocessor.get_data_sample_names('test'),
                                     apply_augmentation=False, config=config)
    print (f"number of test samples: {len(test_dataset)}")

    # deterministic seeding for DataLoader workers and shuffling
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    base_seed = config.get('seed', 0)
    g = torch.Generator()
    g.manual_seed(base_seed)

    test_loader = data.DataLoader(test_dataset, config['training']['batch_size'],
                                 shuffle=False, num_workers=os.cpu_count() - 2, pin_memory=True,
                                 collate_fn=GuidewireDataSet.collate_fn,
                                 worker_init_fn=seed_worker, generator=g)


    results_dir = os.path.join(project_root, 'results', config['config_name'])
    os.makedirs(results_dir, exist_ok=True)

    # save the config
    with open(os.path.join(results_dir, 'config_test.yaml'), 'w') as f:
        yaml.dump(config, f)

    criterion = GuidewireHeatMapLoss(from_logits=config['from_logits'])
    
    # Train
    with open(os.path.join(results_dir, 'test_loss.csv'), 'w') as test_log:        
        # Define CSV headers for test losses (will be set after first test)
        test_writer = csv.writer(test_log)
        test_headers_written = False

        # Test at end of epoch
        model.eval()
        test_losses_sum = {}
        num_test_batches = max(1, len(test_loader))
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test = x_test.cuda(non_blocking=True)
                y_test = y_test.cuda(non_blocking=True)
                # for test, we disable AMP to avoid precision issues
                with torch.amp.autocast(device_type='cuda'):
                    pred_test = model(x_test)
                test_losses = criterion(pred_test, y_test)
                
                # Accumulate test losses
                for loss_name, loss_value in test_losses.items():
                    if loss_name not in test_losses_sum:
                        test_losses_sum[loss_name] = 0.0
                    test_losses_sum[loss_name] += float(loss_value.item())
        
        # Calculate average test losses
        test_losses_avg = {loss_name: loss_sum / num_test_batches 
                            for loss_name, loss_sum in test_losses_sum.items()}
        
        # Calculate weighted total test loss
        test_loss_total = 0.0
        for loss_name in config['training']['loss_main']:
            if loss_name in test_losses_avg and loss_name in config['training']['loss_weights']:
                test_loss_total += config['training']['loss_weights'][loss_name] * test_losses_avg[loss_name]
    
        # Log test losses to CSV
        if not test_headers_written:
            # Write headers dynamically based on available losses
            test_csv_headers = ['epoch', 'test_loss_total'] + list(test_losses_avg.keys())
            test_writer.writerow(test_csv_headers)
            test_headers_written = True

        test_csv_row = [test_loss_total]
        for loss_name, loss_value in test_losses_avg.items():
            test_csv_row.append(loss_value)
        test_writer.writerow(test_csv_row)

        print(f"Test loss: {test_loss_total:.6f}")
        for loss_name, loss_value in test_losses_avg.items():
            print(f"{loss_name}: {loss_value:.6f}")

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
    print('Start training...')
    parser = ArgumentParser()
    # here, the config path is relative to the project root / config folder
    parser.add_argument('--config', default='default.yaml', type=str)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

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

    print('config loaded. number of keys: %d' % len(config.keys()))

    util.setup_multi_processes()
    # set seed and deterministic behavior
    seed_value = config.get('seed', 0)
    util.setup_seed(seed_value)

    if args.train:
        train(config)

    if args.test:
        test(config)

if __name__ == "__main__":
    main()
