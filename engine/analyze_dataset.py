import os
import sys
import json
import yaml
from argparse import ArgumentParser

from gwtd.data_loader.guidewire_data_loader import GuidewireDataPreprocessor

script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def is_xray_subfolder(subfolder_name: str) -> bool:
    """Optical if primary folder number < 100, X-ray if >= 100."""
    primary_number = int(subfolder_name.split('_')[0])
    return primary_number >= 100


def count_labels_per_subfolder(dir_dataset: str) -> dict[str, int]:
    counts = {}
    for subfolder_name in sorted(os.listdir(dir_dataset)):
        labels_dir = os.path.join(dir_dataset, subfolder_name, 'Labels')
        if not os.path.isdir(labels_dir):
            continue
        counts[subfolder_name] = sum(
            1 for label_name in os.listdir(labels_dir)
            if label_name.endswith('.txt')
        )
    return counts


def get_subfolder_from_sample(image_path: str, dir_dataset: str) -> str:
    return os.path.relpath(image_path, dir_dataset).split(os.sep)[0]


def count_modality(samples: list, dir_dataset: str) -> tuple[int, int]:
    optical = 0
    xray = 0
    for image_path, _ in samples:
        if is_xray_subfolder(get_subfolder_from_sample(image_path, dir_dataset)):
            xray += 1
        else:
            optical += 1
    return optical, xray


def build_modality_summary(optical: int, xray: int) -> dict:
    return {
        'optical': optical,
        'x-ray': xray,
        'total': optical + xray,
    }


def print_modality_counts(title: str, optical: int, xray: int) -> None:
    summary = build_modality_summary(optical, xray)
    print(f"{title}")
    print(f"  optical: {summary['optical']}")
    print(f"  x-ray:   {summary['x-ray']}")
    print(f"  total:   {summary['total']}")


def analyze_dataset(config: dict, config_name: str) -> dict:
    dataset_path = os.path.join(project_root, 'datasets', 'guidewire')
    split_ratio = config['dataset']['split_ratio']

    print('=' * 60)
    print('Labels per subfolder')
    print('=' * 60)
    labels_per_subfolder = count_labels_per_subfolder(dataset_path)
    labels_per_subfolder_detail = {}
    for subfolder_name, count in labels_per_subfolder.items():
        modality = 'x-ray' if is_xray_subfolder(subfolder_name) else 'optical'
        labels_per_subfolder_detail[subfolder_name] = {
            'label_count': count,
            'modality': modality,
        }
        print(f"  {subfolder_name:>8}: {count:>5} labels ({modality})")
    total_labels = sum(labels_per_subfolder.values())
    print(f"\nTotal labels across subfolders: {total_labels}")

    preprocessor = GuidewireDataPreprocessor(
        dir_dataset=dataset_path,
        split_ratio=split_ratio,
    )
    all_samples = preprocessor.data_sample_names
    train_samples = preprocessor.get_data_sample_names('train')
    val_samples = preprocessor.get_data_sample_names('val')
    test_samples = preprocessor.get_data_sample_names('test')

    image_label_pairs = {
        'total': len(all_samples),
        'train': len(train_samples),
        'val': len(val_samples),
        'test': len(test_samples),
    }

    print('\n' + '=' * 60)
    print('Image/label pairs loaded by GuidewireDataPreprocessor')
    print('=' * 60)
    print(f"Total pairs: {image_label_pairs['total']}")
    print(f"Split ratio (train/val/test): {split_ratio}")
    print(f"  train: {image_label_pairs['train']}")
    print(f"  val:   {image_label_pairs['val']}")
    print(f"  test:  {image_label_pairs['test']}")

    modality_breakdown = {
        'total': build_modality_summary(*count_modality(all_samples, dataset_path)),
        'train': build_modality_summary(*count_modality(train_samples, dataset_path)),
        'val': build_modality_summary(*count_modality(val_samples, dataset_path)),
        'test': build_modality_summary(*count_modality(test_samples, dataset_path)),
    }

    print('\n' + '=' * 60)
    print('Optical vs X-ray image/label pairs')
    print('=' * 60)
    for split_name in ['total', 'train', 'val', 'test']:
        summary = modality_breakdown[split_name]
        print_modality_counts(split_name.capitalize(), summary['optical'], summary['x-ray'])
        if split_name != 'test':
            print()

    breakdown = {
        'config_name': config_name,
        'split_ratio': split_ratio,
        'labels_per_subfolder': labels_per_subfolder_detail,
        'total_labels': total_labels,
        'image_label_pairs': image_label_pairs,
        'modality_breakdown': modality_breakdown,
    }

    results_dir = os.path.join(project_root, 'results', config_name, 'dataset')
    os.makedirs(results_dir, exist_ok=True)
    breakdown_path = os.path.join(results_dir, 'dataset_breakdown.json')
    with open(breakdown_path, 'w') as f:
        json.dump(breakdown, f, indent=2)
    print(f"\nSaved dataset breakdown to {breakdown_path}")

    return breakdown


def main():
    parser = ArgumentParser(description='Analyze guidewire dataset composition.')
    parser.add_argument('--config', default='default.yaml', type=str)
    args = parser.parse_args()

    config_path = os.path.join(project_root, 'gwtd', 'config', args.config)
    if not config_path.endswith('.yaml'):
        config_path = config_path + '.yaml'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config_name = args.config.split('.')[0]
    print(f"Config: {args.config}")
    analyze_dataset(config, config_name)


if __name__ == '__main__':
    main()
