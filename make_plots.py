import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_tensorboard_data(log_dir):
    """
    Load data from TensorBoard event files.
    
    Returns:
        dict: Dictionary with scalar data organized by tag
    """
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # Get all scalar tags
    tags = event_acc.Tags()['scalars']
    
    data = {}
    for tag in tags:
        events = event_acc.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = {'steps': steps, 'values': values}
    
    return data


def plot_training_losses(data, save_dir):
    """Plot training losses over epochs"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Total loss
    if 'Train/Loss_Total_Epoch' in data:
        ax = axes[0, 0]
        ax.plot(data['Train/Loss_Total_Epoch']['steps'], 
                data['Train/Loss_Total_Epoch']['values'], 
                'b-', linewidth=2, label='Train')
        if 'Val_ShapeNet/Loss_Total' in data:
            ax.plot(data['Val_ShapeNet/Loss_Total']['steps'], 
                    data['Val_ShapeNet/Loss_Total']['values'], 
                    'r-', linewidth=2, label='Val ShapeNet')
        if 'Val_MIT/Loss_Total' in data:
            ax.plot(data['Val_MIT/Loss_Total']['steps'], 
                    data['Val_MIT/Loss_Total']['values'], 
                    'g-', linewidth=2, label='Val MIT')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Total Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Reflectance loss
    if 'Train/Loss_Reflectance_Epoch' in data:
        ax = axes[0, 1]
        ax.plot(data['Train/Loss_Reflectance_Epoch']['steps'], 
                data['Train/Loss_Reflectance_Epoch']['values'], 
                'b-', linewidth=2, label='Train')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Reflectance Loss (L_R)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Shading loss
    if 'Train/Loss_Shading_Epoch' in data:
        ax = axes[1, 0]
        ax.plot(data['Train/Loss_Shading_Epoch']['steps'], 
                data['Train/Loss_Shading_Epoch']['values'], 
                'b-', linewidth=2, label='Train')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Shading Loss (L_S)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Image formation loss
    if 'Train/Loss_ImageFormation_Epoch' in data:
        ax = axes[1, 1]
        ax.plot(data['Train/Loss_ImageFormation_Epoch']['steps'], 
                data['Train/Loss_ImageFormation_Epoch']['values'], 
                'b-', linewidth=2, label='Train')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Image Formation Loss (L_IMF)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_losses.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(save_dir, 'training_losses.png')}")


def plot_validation_metrics(data, save_dir, dataset='ShapeNet'):
    """Plot validation metrics (MSE, LMSE, DSSIM)"""
    prefix = f'Val_{dataset}'
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # MSE
    ax = axes[0, 0]
    if f'{prefix}/MSE_Reflectance' in data:
        ax.plot(data[f'{prefix}/MSE_Reflectance']['steps'], 
                data[f'{prefix}/MSE_Reflectance']['values'], 
                'b-', linewidth=2, label='Reflectance')
    if f'{prefix}/MSE_Shading' in data:
        ax.plot(data[f'{prefix}/MSE_Shading']['steps'], 
                data[f'{prefix}/MSE_Shading']['values'], 
                'r-', linewidth=2, label='Shading')
    if f'{prefix}/MSE_Average' in data:
        ax.plot(data[f'{prefix}/MSE_Average']['steps'], 
                data[f'{prefix}/MSE_Average']['values'], 
                'g--', linewidth=2, label='Average')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.set_title('Mean Squared Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # LMSE
    ax = axes[0, 1]
    if f'{prefix}/LMSE_Reflectance' in data:
        ax.plot(data[f'{prefix}/LMSE_Reflectance']['steps'], 
                data[f'{prefix}/LMSE_Reflectance']['values'], 
                'b-', linewidth=2, label='Reflectance')
    if f'{prefix}/LMSE_Shading' in data:
        ax.plot(data[f'{prefix}/LMSE_Shading']['steps'], 
                data[f'{prefix}/LMSE_Shading']['values'], 
                'r-', linewidth=2, label='Shading')
    if f'{prefix}/LMSE_Average' in data:
        ax.plot(data[f'{prefix}/LMSE_Average']['steps'], 
                data[f'{prefix}/LMSE_Average']['values'], 
                'g--', linewidth=2, label='Average')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('LMSE')
    ax.set_title('Local Mean Squared Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # DSSIM
    ax = axes[0, 2]
    if f'{prefix}/DSSIM_Reflectance' in data:
        ax.plot(data[f'{prefix}/DSSIM_Reflectance']['steps'], 
                data[f'{prefix}/DSSIM_Reflectance']['values'], 
                'b-', linewidth=2, label='Reflectance')
    if f'{prefix}/DSSIM_Shading' in data:
        ax.plot(data[f'{prefix}/DSSIM_Shading']['steps'], 
                data[f'{prefix}/DSSIM_Shading']['values'], 
                'r-', linewidth=2, label='Shading')
    if f'{prefix}/DSSIM_Average' in data:
        ax.plot(data[f'{prefix}/DSSIM_Average']['steps'], 
                data[f'{prefix}/DSSIM_Average']['values'], 
                'g--', linewidth=2, label='Average')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('DSSIM')
    ax.set_title('Structural Dissimilarity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # MSE (log scale)
    ax = axes[1, 0]
    if f'{prefix}/MSE_Reflectance' in data:
        ax.semilogy(data[f'{prefix}/MSE_Reflectance']['steps'], 
                    data[f'{prefix}/MSE_Reflectance']['values'], 
                    'b-', linewidth=2, label='Reflectance')
    if f'{prefix}/MSE_Shading' in data:
        ax.semilogy(data[f'{prefix}/MSE_Shading']['steps'], 
                    data[f'{prefix}/MSE_Shading']['values'], 
                    'r-', linewidth=2, label='Shading')
    if f'{prefix}/MSE_Average' in data:
        ax.semilogy(data[f'{prefix}/MSE_Average']['steps'], 
                    data[f'{prefix}/MSE_Average']['values'], 
                    'g--', linewidth=2, label='Average')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE (log scale)')
    ax.set_title('MSE (Log Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # LMSE (log scale)
    ax = axes[1, 1]
    if f'{prefix}/LMSE_Reflectance' in data:
        ax.semilogy(data[f'{prefix}/LMSE_Reflectance']['steps'], 
                    data[f'{prefix}/LMSE_Reflectance']['values'], 
                    'b-', linewidth=2, label='Reflectance')
    if f'{prefix}/LMSE_Shading' in data:
        ax.semilogy(data[f'{prefix}/LMSE_Shading']['steps'], 
                    data[f'{prefix}/LMSE_Shading']['values'], 
                    'r-', linewidth=2, label='Shading')
    if f'{prefix}/LMSE_Average' in data:
        ax.semilogy(data[f'{prefix}/LMSE_Average']['steps'], 
                    data[f'{prefix}/LMSE_Average']['values'], 
                    'g--', linewidth=2, label='Average')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('LMSE (log scale)')
    ax.set_title('LMSE (Log Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # DSSIM (log scale)
    ax = axes[1, 2]
    if f'{prefix}/DSSIM_Reflectance' in data:
        ax.semilogy(data[f'{prefix}/DSSIM_Reflectance']['steps'], 
                    data[f'{prefix}/DSSIM_Reflectance']['values'], 
                    'b-', linewidth=2, label='Reflectance')
    if f'{prefix}/DSSIM_Shading' in data:
        ax.semilogy(data[f'{prefix}/DSSIM_Shading']['steps'], 
                    data[f'{prefix}/DSSIM_Shading']['values'], 
                    'r-', linewidth=2, label='Shading')
    if f'{prefix}/DSSIM_Average' in data:
        ax.semilogy(data[f'{prefix}/DSSIM_Average']['steps'], 
                    data[f'{prefix}/DSSIM_Average']['values'], 
                    'g--', linewidth=2, label='Average')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('DSSIM (log scale)')
    ax.set_title('DSSIM (Log Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'validation_metrics_{dataset.lower()}.png'
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(save_dir, filename)}")


def plot_learning_rate(data, save_dir):
    """Plot learning rate schedule"""
    if 'Train/LearningRate' not in data:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    steps = data['Train/LearningRate']['steps']
    values = data['Train/LearningRate']['values']
    
    ax.plot(steps, values, 'b-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(save_dir, 'learning_rate.png')}")


def plot_loss_components_comparison(data, save_dir):
    """Plot all loss components together for comparison"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    if 'Train/Loss_Reflectance_Epoch' in data:
        ax.plot(data['Train/Loss_Reflectance_Epoch']['steps'], 
                data['Train/Loss_Reflectance_Epoch']['values'], 
                'b-', linewidth=2, label='L_R (Reflectance)')
    
    if 'Train/Loss_Shading_Epoch' in data:
        ax.plot(data['Train/Loss_Shading_Epoch']['steps'], 
                data['Train/Loss_Shading_Epoch']['values'], 
                'r-', linewidth=2, label='L_S (Shading)')
    
    if 'Train/Loss_ImageFormation_Epoch' in data:
        ax.plot(data['Train/Loss_ImageFormation_Epoch']['steps'], 
                data['Train/Loss_ImageFormation_Epoch']['values'], 
                'g-', linewidth=2, label='L_IMF (Image Formation)')
    
    if 'Train/Loss_Total_Epoch' in data:
        ax.plot(data['Train/Loss_Total_Epoch']['steps'], 
                data['Train/Loss_Total_Epoch']['values'], 
                'k--', linewidth=2, label='Total Loss')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Components Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_components.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(save_dir, 'loss_components.png')}")


def plot_training_smoothed(data, save_dir, window=10):
    """Plot smoothed training curves"""
    def smooth(values, window):
        if len(values) < window:
            return values
        smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
        return smoothed
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    if 'Train/Loss_Total_Iter' in data:
        steps = data['Train/Loss_Total_Iter']['steps']
        values = data['Train/Loss_Total_Iter']['values']
        
        # Plot raw
        ax.plot(steps, values, 'b-', alpha=0.2, linewidth=0.5, label='Raw')
        
        # Plot smoothed
        smoothed_values = smooth(values, window)
        smoothed_steps = steps[window-1:]
        ax.plot(smoothed_steps, smoothed_values, 'b-', linewidth=2, label=f'Smoothed (window={window})')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss (Smoothed)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_loss_smoothed.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(save_dir, 'training_loss_smoothed.png')}")


def print_summary_statistics(data):
    """Print summary statistics from training"""
    print("\n" + "="*60)
    print("TRAINING SUMMARY STATISTICS")
    print("="*60)
    
    # Final epoch metrics
    for key in ['Train/Loss_Total_Epoch', 'Val_ShapeNet/Loss_Total', 'Val_MIT/Loss_Total']:
        if key in data and len(data[key]['values']) > 0:
            final_value = data[key]['values'][-1]
            min_value = min(data[key]['values'])
            max_value = max(data[key]['values'])
            print(f"\n{key}:")
            print(f"  Final: {final_value:.6f}")
            print(f"  Min: {min_value:.6f}")
            print(f"  Max: {max_value:.6f}")
    
    # Best validation metrics
    for dataset in ['ShapeNet', 'MIT']:
        prefix = f'Val_{dataset}'
        if f'{prefix}/MSE_Average' in data:
            print(f"\nBest {dataset} Metrics:")
            
            mse_avg = data[f'{prefix}/MSE_Average']['values']
            lmse_avg = data[f'{prefix}/LMSE_Average']['values']
            dssim_avg = data[f'{prefix}/DSSIM_Average']['values']
            
            best_mse_idx = np.argmin(mse_avg)
            best_lmse_idx = np.argmin(lmse_avg)
            best_dssim_idx = np.argmin(dssim_avg)
            
            print(f"  Best MSE: {mse_avg[best_mse_idx]:.6f} at epoch {best_mse_idx}")
            print(f"  Best LMSE: {lmse_avg[best_lmse_idx]:.6f} at epoch {best_lmse_idx}")
            print(f"  Best DSSIM: {dssim_avg[best_dssim_idx]:.6f} at epoch {best_dssim_idx}")
    
    print("="*60 + "\n")


def main(args):
    log_dir = args.log_dir
    save_dir = args.save_dir
    
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(log_dir), 'plots')
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Loading TensorBoard logs from: {log_dir}")
    data = load_tensorboard_data(log_dir)
    
    print(f"Found {len(data)} scalar tags")
    
    # Create plots
    print("\nGenerating plots...")
    plot_training_losses(data, save_dir)
    
    if any('Val_ShapeNet' in key for key in data.keys()):
        plot_validation_metrics(data, save_dir, dataset='ShapeNet')
    
    if any('Val_MIT' in key for key in data.keys()):
        plot_validation_metrics(data, save_dir, dataset='MIT')
    
    plot_learning_rate(data, save_dir)
    plot_loss_components_comparison(data, save_dir)
    plot_training_smoothed(data, save_dir, window=args.smooth_window)
    
    # Print summary
    print_summary_statistics(data)
    
    print(f"\nAll plots saved to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot TensorBoard logs')
    parser.add_argument('--log_dir', type=str, required=True,
                       help='Path to TensorBoard log directory (e.g., runs/retinet_shapenet)')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save plots (default: <log_dir>/../plots)')
    parser.add_argument('--smooth_window', type=int, default=10,
                       help='Window size for smoothing training curves')
    
    args = parser.parse_args()
    main(args)