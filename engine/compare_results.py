#!/usr/bin/env python3
"""
Compare training results across different experiment folders.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def moving_average(data: np.ndarray, window: int = 500) -> np.ndarray:
    """Apply moving average to smooth the data."""
    if len(data) < window:
        return data
    
    smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
    # Pad the beginning with original values
    padded = np.full(len(data), np.nan)
    padded[window-1:] = smoothed
    padded[:window-1] = data[:window-1]
    return padded


def load_training_data(folder_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load training step data and validation data from a folder."""
    step_file = folder_path / "step.csv"
    val_file = folder_path / "val_loss.csv"
    
    if not step_file.exists():
        raise FileNotFoundError(f"step.csv not found in {folder_path}")
    if not val_file.exists():
        raise FileNotFoundError(f"val_loss.csv not found in {folder_path}")
    
    step_df = pd.read_csv(step_file)
    val_df = pd.read_csv(val_file)
    
    return step_df, val_df


def calculate_steps_per_epoch(step_df: pd.DataFrame) -> int:
    """Calculate number of steps per epoch from training data."""
    # Find the first step of epoch 2
    epoch_2_start = step_df[step_df['epoch'] == 2]['step'].iloc[0] if len(step_df[step_df['epoch'] == 2]) > 0 else None
    epoch_1_start = step_df[step_df['epoch'] == 1]['step'].iloc[0] if len(step_df[step_df['epoch'] == 1]) > 0 else None
    
    if epoch_2_start is not None and epoch_1_start is not None:
        return epoch_2_start - epoch_1_start
    else:
        # Fallback: estimate from total steps and epochs
        max_epoch = step_df['epoch'].max()
        max_step = step_df['step'].max()
        return max_step // max_epoch


def plot_loss_comparison(results_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]], 
                        output_dir: Path):
    """Plot training and validation loss comparison."""
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_data)))
    
    for i, (folder_name, (step_df, val_df)) in enumerate(results_data.items()):
        color = colors[i]
        
        # Calculate steps per epoch
        steps_per_epoch = calculate_steps_per_epoch(step_df)
        
        # Plot training loss (smoothed)
        train_loss_smoothed = moving_average(step_df['loss_total'].values)
        plt.plot(step_df['step'], train_loss_smoothed, 
                label=f'{folder_name} (train)', color=color, linewidth=2)
        
        # Plot validation loss
        val_steps = val_df['epoch'] * steps_per_epoch
        plt.plot(val_steps, val_df['val_loss_total'], 
                label=f'{folder_name} (val)', color=color, linestyle='--', linewidth=2)
    
    plt.xlabel('Step')
    plt.ylabel('Loss (log scale)')
    plt.title('Training vs Validation Loss Comparison')
    plt.yscale('log')  # Use log scale for y-axis
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = output_dir / 'loss_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss comparison plot saved to: {output_file}")


def plot_metric_comparison(results_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]], 
                          output_dir: Path):
    """Plot comparison for each metric."""
    # Collect all metric names across every experiment (union)
    all_step_metrics = set()
    all_val_metrics = set()
    for step_df, val_df in results_data.values():
        all_step_metrics.update(col for col in step_df.columns
                                if col not in ['epoch', 'step', 'lr', 'loss_total'])
        all_val_metrics.update(col for col in val_df.columns
                               if col not in ['epoch', 'val_loss_total'])
    
    common_metrics = all_step_metrics & all_val_metrics
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(results_data)))
    
    for metric in common_metrics:
        plt.figure(figsize=(12, 8))
        
        for i, (folder_name, (step_df, val_df)) in enumerate(results_data.items()):
            color = colors[i]
            
            if metric not in step_df.columns or metric not in val_df.columns:
                continue
            
            # Calculate steps per epoch
            steps_per_epoch = calculate_steps_per_epoch(step_df)
            
            # Plot training metric (smoothed)
            train_metric_smoothed = moving_average(step_df[metric].values)
            plt.plot(step_df['step'], train_metric_smoothed, 
                    label=f'{folder_name} (train)', color=color, linewidth=2)
            
            # Plot validation metric
            val_steps = val_df['epoch'] * steps_per_epoch
            plt.plot(val_steps, val_df[metric], 
                    label=f'{folder_name} (val)', color=color, linestyle='--', linewidth=2)
        
        plt.xlabel('Step')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'{metric.replace("_", " ").title()} Comparison')
        
        # Use log scale for loss-related metrics
        if metric in ['bce', 'mse', 'mae', 'dist']:
            plt.yscale('log')
            plt.ylabel(f'{metric.replace("_", " ").title()} (log scale)')
        
        # Strict 0.0–1.0 y-axis with gridlines every 0.1 for window-accuracy metrics
        if 'win_acc' in metric:
            ax = plt.gca()
            ax.set_ylim(0.0, 1.0)
            ax.set_yticks(np.arange(0.0, 1.0 + 1e-9, 0.1))
            ax.yaxis.grid(True, which='major', alpha=0.5)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_file = output_dir / f'{metric}_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{metric} comparison plot saved to: {output_file}")


def plot_lr_comparison(results_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
                       output_dir: Path):
    """Plot training learning rate comparison across experiments."""
    plt.figure(figsize=(12, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results_data)))

    for i, (folder_name, (step_df, _)) in enumerate(results_data.items()):
        color = colors[i]

        if 'lr' not in step_df.columns:
            continue

        plt.plot(step_df['step'], step_df['lr'],
                 label=f'{folder_name}', color=color, linewidth=2)

    plt.xlabel('Step')
    plt.ylabel('Learning Rate (log scale)')
    plt.title('Learning Rate Comparison')
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_file = output_dir / 'lr_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Learning rate comparison plot saved to: {output_file}")


def main():
    """Main function to compare results across experiment folders."""
    parser = argparse.ArgumentParser(description='Compare training results across experiments')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Path to results directory')
    parser.add_argument('--output_dir', type=str, default='results/summary',
                       help='Path to output directory for plots')
    parser.add_argument('--exclude_folders', nargs='*', default=['summary', 'dataset_test'],
                       help='Folders to exclude from comparison')
    parser.add_argument('--target', nargs='+', default=None,
                       help='Specific experiment folder names (within results_dir) to compare')
    
    args = parser.parse_args()
    
    # Convert to Path objects
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all experiment folders (must contain step.csv and val_loss.csv)
    def _has_required_csvs(path: Path) -> bool:
        return (path / "step.csv").is_file() and (path / "val_loss.csv").is_file()

    experiment_folders = []
    if args.target:
        for name in args.target:
            path = results_dir / name
            if not path.is_dir():
                print(f"Warning: target folder not found: {path}")
                continue
            if not _has_required_csvs(path):
                print(f"Warning: target folder missing step.csv or val_loss.csv: {path}")
                continue
            experiment_folders.append(path)
    else:
        for item in results_dir.iterdir():
            if (item.is_dir()
                    and item.name not in args.exclude_folders
                    and _has_required_csvs(item)):
                experiment_folders.append(item)
    
    if not experiment_folders:
        print("No experiment folders found!")
        return
    
    print(f"Found {len(experiment_folders)} experiment folders:")
    for folder in experiment_folders:
        print(f"  - {folder.name}")
    
    # Load data from all folders
    results_data = {}
    for folder in experiment_folders:
        try:
            step_df, val_df = load_training_data(folder)
            results_data[folder.name] = (step_df, val_df)
            print(f"Loaded data from {folder.name}")
        except Exception as e:
            print(f"Error loading data from {folder.name}: {e}")
            continue
    
    if not results_data:
        print("No valid data found!")
        return
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Plot loss comparison
    plot_loss_comparison(results_data, output_dir)
    
    # Plot learning rate comparison
    plot_lr_comparison(results_data, output_dir)
    
    # Plot metric comparisons
    plot_metric_comparison(results_data, output_dir)
    
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
