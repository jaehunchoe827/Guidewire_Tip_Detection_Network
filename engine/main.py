import os
import sys
import csv
import json
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

    # Optional EMA shadow model. Disabled by default for backward compatibility:
    # any config without `use_ema: true` runs identically to before.
    ema = None
    if config.get('use_ema', False):
        ema_cfg = config.get('ema', {}) or {}
        ema = training_utils.ModelEMA(
            model,
            decay=ema_cfg.get('decay', 0.9999),
            tau=ema_cfg.get('tau', 2000),
        )
        print(f"EMA enabled (decay={ema.decay_base}, tau={ema.tau})")

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
                    if ema is not None:
                        ema.update(model)

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
                    acc1=f"{losses['1%_win_acc'].item():.3f}",
                    dist=f"{losses['dist'].item():.3f}"
                )

                # update global step
                global_step += 1

            # Validation at end of epoch
            def _run_validation(eval_model):
                eval_model.eval()
                losses_sum = {}
                n_batches = max(1, len(val_loader))
                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        x_val = x_val.cuda(non_blocking=True)
                        y_val = y_val.cuda(non_blocking=True)
                        with torch.amp.autocast(device_type='cuda'):
                            pred_val = eval_model(x_val)
                        val_losses = criterion(pred_val, y_val)
                        for loss_name, loss_value in val_losses.items():
                            if loss_name not in losses_sum:
                                losses_sum[loss_name] = 0.0
                            losses_sum[loss_name] += float(loss_value.item())
                losses_avg = {ln: ls / n_batches for ln, ls in losses_sum.items()}
                total = 0.0
                for loss_name in config['training']['loss_main']:
                    if loss_name in losses_avg and loss_name in config['training']['loss_weights']:
                        total += config['training']['loss_weights'][loss_name] * losses_avg[loss_name]
                return losses_avg, total

            val_losses_avg, val_loss_total = _run_validation(model)
            ema_val_losses_avg, ema_val_loss_total = (None, None)
            if ema is not None:
                ema_val_losses_avg, ema_val_loss_total = _run_validation(ema.ema)

            print(f"Epoch {epoch}/{epochs} - val_loss_total: {val_loss_total:.6f}, "
                  f"val_acc1: {val_losses_avg.get('1%_win_acc', 0):.5f}, "
                  f"val_dist: {val_losses_avg.get('dist', 0):.5f}")
            if ema is not None:
                print(f"Epoch {epoch}/{epochs} - ema_val_loss_total: {ema_val_loss_total:.6f}, "
                      f"ema_val_acc1: {ema_val_losses_avg.get('1%_win_acc', 0):.5f}, "
                      f"ema_val_dist: {ema_val_losses_avg.get('dist', 0):.5f}")

            # Log validation losses to CSV
            if not val_headers_written:
                val_csv_headers = ['epoch', 'val_loss_total'] + list(val_losses_avg.keys())
                if ema is not None:
                    val_csv_headers += ['ema_val_loss_total'] + [
                        f'ema_{ln}' for ln in ema_val_losses_avg.keys()
                    ]
                val_writer.writerow(val_csv_headers)
                val_headers_written = True

            val_csv_row = [epoch, val_loss_total]
            for loss_name, loss_value in val_losses_avg.items():
                val_csv_row.append(loss_value)
            if ema is not None:
                val_csv_row.append(ema_val_loss_total)
                for loss_name, loss_value in ema_val_losses_avg.items():
                    val_csv_row.append(loss_value)
            val_writer.writerow(val_csv_row)

            # Save best checkpoint by 1%_win_acc. When EMA is enabled, the EMA
            # model is the published artifact and drives both selection and
            # the saved state_dict.
            score_source = ema_val_losses_avg if ema is not None else val_losses_avg
            one_percent_win_acc = score_source.get('1%_win_acc', 0)
            if one_percent_win_acc > best_score:
                best_score = one_percent_win_acc
                state_to_save = (ema.ema.state_dict()
                                 if ema is not None else model.state_dict())
                ckpt = {
                    'model': state_to_save,
                    'epoch': epoch,
                    'score': best_score,
                    'config': config,
                }
                torch.save(ckpt, os.path.join(results_dir, 'best.pt'))

            model.train()
    return

def build_test_model(config):
    """Create the model and load the trained weights for evaluation."""
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
    return model


def evaluate_samples(config, model, criterion, samples, desc='Testing'):
    """Run evaluation on the given (image_path, label_path) samples.

    Returns a dict with the number of samples, total test loss and every
    averaged loss term.
    """
    test_dataset = GuidewireDataSet(samples, apply_augmentation=False, config=config)
    print(f"number of test samples: {len(test_dataset)}")

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

    model.eval()
    test_losses_sum = {}
    num_test_batches = max(1, len(test_loader))
    with torch.no_grad():
        p_bar = tqdm.tqdm(test_loader, total=num_test_batches,
                          desc=desc, leave=True, ncols=50)
        for x_test, y_test in p_bar:
            x_test = x_test.cuda(non_blocking=True)
            y_test = y_test.cuda(non_blocking=True)
            with torch.amp.autocast(device_type='cuda'):
                pred_test = model(x_test)
            test_losses = criterion(pred_test, y_test)

            for loss_name, loss_value in test_losses.items():
                if loss_name not in test_losses_sum:
                    test_losses_sum[loss_name] = 0.0
                test_losses_sum[loss_name] += float(loss_value.item())

    test_losses_avg = {loss_name: loss_sum / num_test_batches
                       for loss_name, loss_sum in test_losses_sum.items()}

    test_loss_total = 0.0
    for loss_name in config['training']['loss_main']:
        if loss_name in test_losses_avg and loss_name in config['training']['loss_weights']:
            test_loss_total += config['training']['loss_weights'][loss_name] * test_losses_avg[loss_name]

    test_results = {
        'number_of_test_samples': len(test_dataset),
        'test_loss': test_loss_total,
    }
    test_results.update(test_losses_avg)
    return test_results


def test(config):
    model = build_test_model(config)
    # prepare dataset and loader
    dataset_path = os.path.join(project_root, 'datasets', 'guidewire')
    data_preprocessor = GuidewireDataPreprocessor(dir_dataset=dataset_path,
                                                  split_ratio=config['dataset']['split_ratio'])

    results_dir = os.path.join(project_root, 'results', config['config_name'])
    os.makedirs(results_dir, exist_ok=True)

    # save the config
    with open(os.path.join(results_dir, 'config_test.yaml'), 'w') as f:
        yaml.dump(config, f)

    criterion = GuidewireHeatMapLoss(from_logits=config['from_logits'])

    test_results = evaluate_samples(config, model, criterion,
                                    data_preprocessor.get_data_sample_names('test'))

    test_dir = os.path.join(results_dir, 'test')
    os.makedirs(test_dir, exist_ok=True)
    test_results_path = os.path.join(test_dir, 'test_results.json')
    with open(test_results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"Saved test results to {test_results_path}")

    test_loss_total = test_results['test_loss']
    print(f"Test loss: {test_loss_total:.6f}")
    for loss_name, loss_value in test_results.items():
        if loss_name in ('number_of_test_samples', 'test_loss'):
            continue
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
