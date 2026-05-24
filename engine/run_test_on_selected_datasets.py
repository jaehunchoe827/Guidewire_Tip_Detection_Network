import os
import sys
import csv
import yaml
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils import data

from gwtd.data_loader.guidewire_data_loader import GuidewireDataSet
from gwtd.loss.loss import GuidewireHeatMapLoss

# Add project root to Python path for model loading
script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gwtd.nets import nn
from gwtd.utils import util


# Edit these two constants per run.
CONFIG_NAME = "default"
SELECTED_DATASETS = [
    "103", "104", "106", "107", "108", "109", "110", "111", "112", "113",
    "114", "115", "117", "119", "120", "121", "122", "123", "124", "125",
    "126", "127", "128"
]


def collect_data_sample_names(dir_dataset: str, selected_datasets: list):
    """Walk only the specified video folders and pair Images/Labels by filename."""
    data_sample_names = []
    for video_dir in selected_datasets:
        video_path = os.path.join(dir_dataset, video_dir)
        labels_dir = os.path.join(video_path, 'Labels')
        images_dir = os.path.join(video_path, 'Images')
        if not os.path.isdir(labels_dir) or not os.path.isdir(images_dir):
            raise FileNotFoundError(
                f"Missing Images or Labels folder under '{video_path}'"
            )
        for label_name in sorted(os.listdir(labels_dir)):
            if not label_name.endswith('.txt'):
                continue
            label_path = os.path.join(labels_dir, label_name)
            image_path = os.path.join(images_dir, label_name.replace('.txt', '.jpg'))
            if not os.path.isfile(image_path):
                raise FileNotFoundError(
                    f"Image not found for label '{label_path}': '{image_path}'"
                )
            data_sample_names.append((image_path, label_path))
    return data_sample_names


def run_test_on_selected_datasets(config, selected_datasets):
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

    # gather samples from selected dataset folders only
    dataset_path = os.path.join(project_root, 'datasets', 'guidewire')
    sample_names = collect_data_sample_names(dataset_path, selected_datasets)
    test_dataset = GuidewireDataSet(sample_names,
                                    apply_augmentation=False, config=config)
    print(f"selected datasets: {selected_datasets}")
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

    results_dir = os.path.join(project_root, 'results', config['config_name'])
    os.makedirs(results_dir, exist_ok=True)

    criterion = GuidewireHeatMapLoss(from_logits=config['from_logits'])

    csv_filename = f"test_loss_{'_'.join(selected_datasets)}.csv"
    csv_path = os.path.join(results_dir, csv_filename)
    with open(csv_path, 'w') as test_log:
        test_writer = csv.writer(test_log)
        test_headers_written = False

        model.eval()
        test_losses_sum = {}
        num_test_batches = max(1, len(test_loader))
        with torch.no_grad():
            for x_test, y_test in tqdm(test_loader, total=num_test_batches, desc='Testing'):
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

        if not test_headers_written:
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
        print(f"Wrote results CSV to: {csv_path}")

    return


def main():
    print('Start running test on selected datasets...')
    config_path = os.path.join(project_root, 'gwtd', 'config', f'{CONFIG_NAME}.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['config_name'] = CONFIG_NAME

    print(f"config loaded: {CONFIG_NAME}. number of keys: {len(config.keys())}")

    util.setup_multi_processes()
    seed_value = config.get('seed', 0)
    util.setup_seed(seed_value)

    run_test_on_selected_datasets(config, SELECTED_DATASETS)


if __name__ == "__main__":
    main()
