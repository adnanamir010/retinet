# evaluate_retinet.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from torch.utils.data import DataLoader

from models.retinet import RetiNet
from losses.intrinsic_loss import IntrinsicLoss
from data.shapenet_dataset import ShapeNetIntrinsicsDataset
from data.mit_dataset import MITIntrinsicDataset
from utils.metrics import compute_all_metrics


class RetiNetEvaluator:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        print(f"\nLoading checkpoint: {checkpoint_path}")
        
        if self.model is None:
            self.model = RetiNet(use_dropout=False, compute_gradients=True).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        epoch = checkpoint.get('epoch', 'unknown')
        loss = checkpoint.get('loss', 'unknown')
        print(f"  Epoch: {epoch}, Loss: {loss}")
        
        return epoch, loss
    
    def evaluate_dataset(self, dataloader, dataset_name, save_vis=False, vis_dir=None, num_vis=None):
        """
        Evaluate model on a dataset
        
        Args:
            num_vis: Number of visualizations to save (None = save all)
        
        Returns:
            dict: Metrics averaged over the dataset
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_checkpoint first.")
        
        self.model.eval()
        
        all_metrics = {
            'mse_reflectance': [],
            'mse_shading': [],
            'lmse_reflectance': [],
            'lmse_shading': [],
            'dssim_reflectance': [],
            'dssim_shading': []
        }
        
        criterion = IntrinsicLoss(gamma_r=1.0, gamma_s=1.0, gamma_imf=1.0)
        total_loss = 0.0
        
        vis_saved = 0
        total_batches = 0
        total_images = 0
        
        # If num_vis is None, save all
        save_all_vis = (num_vis is None)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {dataset_name}")):
                total_batches += 1
                
                # Handle both tuple (ShapeNet) and dict (MIT) formats
                if isinstance(batch, dict):
                    original = batch['original'].to(self.device)
                    reflectance_gt = batch['reflectance'].to(self.device)
                    shading_gt = batch['shading'].to(self.device)
                    names = batch['name']
                else:
                    original, reflectance_gt, shading_gt = batch
                    original = original.to(self.device)
                    reflectance_gt = reflectance_gt.to(self.device)
                    shading_gt = shading_gt.to(self.device)
                    names = [f'sample_{batch_idx}_{i}' for i in range(original.size(0))]
                
                batch_size = original.size(0)
                total_images += batch_size
                
                # Forward pass
                reflectance_pred, shading_pred, _, _ = self.model(original)
                
                # Compute loss
                loss, _ = criterion(reflectance_pred, shading_pred,
                                   reflectance_gt, shading_gt, original)
                total_loss += loss.item()
                
                # Compute metrics for each image in batch
                for i in range(batch_size):
                    metrics = compute_all_metrics(
                        reflectance_pred[i:i+1], 
                        shading_pred[i:i+1],
                        reflectance_gt[i:i+1], 
                        shading_gt[i:i+1]
                    )
                    
                    for key in all_metrics.keys():
                        all_metrics[key].append(metrics[key])
                
                # Save visualizations
                if save_vis:
                    for i in range(batch_size):
                        # Check if we should save this visualization
                        if save_all_vis or vis_saved < num_vis:
                            name = names[i] if isinstance(names, list) else f'{names}_{i}'
                            self._save_visualization(
                                original[i:i+1], 
                                reflectance_pred[i:i+1], 
                                shading_pred[i:i+1],
                                reflectance_gt[i:i+1], 
                                shading_gt[i:i+1],
                                name, vis_dir, dataset_name
                            )
                            vis_saved += 1
        
        print(f"\nProcessed {total_batches} batches, {total_images} images")
        if save_vis:
            print(f"Saved {vis_saved} visualizations")
        
        # Average metrics
        results = {
            'loss': total_loss / total_batches,
            'num_samples': total_images
        }
        
        for key in all_metrics.keys():
            results[key] = np.mean(all_metrics[key])
            results[f'{key}_std'] = np.std(all_metrics[key])
        
        # Compute averages
        results['mse_avg'] = (results['mse_reflectance'] + results['mse_shading']) / 2
        results['lmse_avg'] = (results['lmse_reflectance'] + results['lmse_shading']) / 2
        results['dssim_avg'] = (results['dssim_reflectance'] + results['dssim_shading']) / 2
        
        return results
    
    def _save_visualization(self, original, r_pred, s_pred, r_gt, s_gt, name, vis_dir, dataset_name):
        """Save visualization of predictions vs ground truth"""
        os.makedirs(vis_dir, exist_ok=True)
        
        def to_numpy(tensor):
            return np.clip(tensor[0].cpu().numpy() / 255.0, 0, 1)
        
        orig_np = to_numpy(original).transpose(1, 2, 0)
        r_gt_np = to_numpy(r_gt).transpose(1, 2, 0)
        s_gt_np = to_numpy(s_gt)[0]
        r_pred_np = to_numpy(r_pred).transpose(1, 2, 0)
        s_pred_np = to_numpy(s_pred)[0]
        
        # Compute reconstruction
        recon_np = r_pred_np * s_pred_np[..., np.newaxis]
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Row 1: Ground truth
        axes[0, 0].imshow(orig_np)
        axes[0, 0].set_title('Input')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(r_gt_np)
        axes[0, 1].set_title('GT Reflectance')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(s_gt_np, cmap='gray')
        axes[0, 2].set_title('GT Shading')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(orig_np)
        axes[0, 3].set_title('Input (repeat)')
        axes[0, 3].axis('off')
        
        # Row 2: Predictions
        axes[1, 0].imshow(recon_np)
        axes[1, 0].set_title('Reconstructed (R x S)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(r_pred_np)
        axes[1, 1].set_title('Pred Reflectance')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(s_pred_np, cmap='gray')
        axes[1, 2].set_title('Pred Shading')
        axes[1, 2].axis('off')
        
        # Error map
        error = np.abs(orig_np - recon_np)
        axes[1, 3].imshow(error, cmap='hot')
        axes[1, 3].set_title('Reconstruction Error')
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        safe_name = str(name).replace('/', '_').replace('\\', '_')
        filename = f'{dataset_name}_{safe_name}.png'
        plt.savefig(os.path.join(vis_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()


def evaluate_single_checkpoint(checkpoint_path, datasets, evaluator, save_vis=False, 
                            vis_dir=None, checkpoint_name=None, num_vis_per_dataset=None):
    """Evaluate a single checkpoint on all datasets"""
    epoch, loss = evaluator.load_checkpoint(checkpoint_path)
    
    if checkpoint_name is None:
        checkpoint_name = Path(checkpoint_path).stem
    
    results = {
        'checkpoint': checkpoint_name,
        'epoch': epoch,
        'train_loss': loss
    }
    
    for dataset_name, dataloader in datasets.items():
        print(f"\n{'='*60}")
        print(f"Evaluating on {dataset_name}")
        print(f"Dataset size: {len(dataloader.dataset)} images")
        print(f"Batch size: {dataloader.batch_size}")
        print(f"Number of batches: {len(dataloader)}")
        print(f"{'='*60}")
        
        if vis_dir:
            dataset_vis_dir = os.path.join(vis_dir, checkpoint_name, dataset_name)
        else:
            dataset_vis_dir = None
        
        # Determine number of visualizations to save
        if num_vis_per_dataset is not None:
            num_vis = num_vis_per_dataset.get(dataset_name, None)
        else:
            num_vis = None
        
        metrics = evaluator.evaluate_dataset(
            dataloader, 
            dataset_name, 
            save_vis=save_vis,
            vis_dir=dataset_vis_dir,
            num_vis=num_vis
        )
        
        # Add dataset prefix to metric names
        for key, value in metrics.items():
            results[f'{dataset_name}_{key}'] = value
        
        # Print summary
        print(f"\nResults for {dataset_name}:")
        print(f"  Images evaluated: {metrics['num_samples']}")
        print(f"  Loss: {metrics['loss']:.6f}")
        print(f"  MSE - R: {metrics['mse_reflectance']:.6f}, S: {metrics['mse_shading']:.6f}, Avg: {metrics['mse_avg']:.6f}")
        print(f"  LMSE - R: {metrics['lmse_reflectance']:.6f}, S: {metrics['lmse_shading']:.6f}, Avg: {metrics['lmse_avg']:.6f}")
        print(f"  DSSIM - R: {metrics['dssim_reflectance']:.6f}, S: {metrics['dssim_shading']:.6f}, Avg: {metrics['dssim_avg']:.6f}")
    
    return results


def find_all_checkpoints(checkpoint_dir):
    """Find all checkpoint files in directory"""
    checkpoint_dir = Path(checkpoint_dir)
    
    # Find all .pth files
    checkpoints = list(checkpoint_dir.glob('*.pth'))
    
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    # Sort by modification time or name
    checkpoints.sort(key=lambda x: x.stat().st_mtime)
    
    return checkpoints


def create_comparison_table(all_results, save_path):
    """Create comparison table of all checkpoints"""
    df = pd.DataFrame(all_results)
    
    # Reorder columns for better readability
    fixed_cols = ['checkpoint', 'epoch', 'train_loss']
    other_cols = [col for col in df.columns if col not in fixed_cols]
    df = df[fixed_cols + sorted(other_cols)]
    
    # Save to CSV
    csv_path = save_path.replace('.txt', '.csv')
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"\nSaved CSV: {csv_path}")
    
    # Save formatted text table
    with open(save_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("CHECKPOINT EVALUATION RESULTS\n")
        f.write("="*100 + "\n\n")
        f.write(df.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
        f.write("\n\n")
        
        # Find best models per metric per dataset
        f.write("="*100 + "\n")
        f.write("BEST MODELS\n")
        f.write("="*100 + "\n\n")
        
        metric_cols = [col for col in df.columns if any(m in col for m in ['mse_avg', 'lmse_avg', 'dssim_avg', 'loss'])]
        
        for col in metric_cols:
            if col in df.columns:
                best_idx = df[col].idxmin()
                best_checkpoint = df.loc[best_idx, 'checkpoint']
                best_value = df.loc[best_idx, col]
                f.write(f"{col}: {best_checkpoint} ({best_value:.6f})\n")
    
    print(f"Saved comparison table: {save_path}")
    
    return df


def plot_checkpoint_comparison(df, save_dir):
    """Plot metrics across different checkpoints"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract dataset names
    datasets = set()
    for col in df.columns:
        if '_mse_avg' in col:
            dataset = col.replace('_mse_avg', '')
            datasets.add(dataset)
    
    for dataset in datasets:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # MSE
        if f'{dataset}_mse_avg' in df.columns:
            axes[0].plot(df['epoch'], df[f'{dataset}_mse_avg'], 'bo-', linewidth=2, markersize=6)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('MSE')
            axes[0].set_title(f'{dataset} - MSE Average')
            axes[0].grid(True, alpha=0.3)
        
        # LMSE
        if f'{dataset}_lmse_avg' in df.columns:
            axes[1].plot(df['epoch'], df[f'{dataset}_lmse_avg'], 'go-', linewidth=2, markersize=6)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('LMSE')
            axes[1].set_title(f'{dataset} - LMSE Average')
            axes[1].grid(True, alpha=0.3)
        
        # DSSIM
        if f'{dataset}_dssim_avg' in df.columns:
            axes[2].plot(df['epoch'], df[f'{dataset}_dssim_avg'], 'ro-', linewidth=2, markersize=6)
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('DSSIM')
            axes[2].set_title(f'{dataset} - DSSIM Average')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'checkpoint_comparison_{dataset}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved comparison plot: checkpoint_comparison_{dataset}.png")

def convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj


def main(args):
    # Setup
    evaluator = RetiNetEvaluator(device=args.device)
    
    # Prepare datasets
    datasets = {}
    
    if args.eval_shapenet:
        print("Loading ShapeNet val dataset...")
        shapenet_dataset = ShapeNetIntrinsicsDataset(
            root_dir=args.shapenet_dir,
            split='val',
            image_size=(args.image_height, args.image_width)
        )
        datasets['ShapeNet'] = DataLoader(
            shapenet_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        print(f"  ShapeNet val: {len(shapenet_dataset)} samples")
    
    if args.eval_mit:
        print("Loading MIT dataset...")
        mit_dataset = MITIntrinsicDataset(
            root_dir=args.mit_dir,
            augment=False,
            image_size=(args.image_height, args.image_width)
        )
        datasets['MIT'] = DataLoader(
            mit_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        print(f"  MIT: {len(mit_dataset)} samples")
    
    if not datasets:
        raise ValueError("No datasets specified for evaluation")
    
    # Determine number of visualizations per dataset
    num_vis_per_dataset = {}
    if args.save_vis:
        if args.num_vis_shapenet is not None:
            num_vis_per_dataset['ShapeNet'] = args.num_vis_shapenet
        if args.num_vis_mit is not None:
            num_vis_per_dataset['MIT'] = args.num_vis_mit
        
        # If not specified, use defaults
        if 'ShapeNet' not in num_vis_per_dataset and args.eval_shapenet:
            num_vis_per_dataset['ShapeNet'] = 20  # Save 20 samples
        if 'MIT' not in num_vis_per_dataset and args.eval_mit:
            num_vis_per_dataset['MIT'] = None  # Save all (20 images)
    else:
        num_vis_per_dataset = None
    
    # Find checkpoints
    if args.eval_all:
        checkpoints = find_all_checkpoints(args.checkpoint_dir)
        print(f"\nFound {len(checkpoints)} checkpoints")
    else:
        checkpoints = [Path(args.checkpoint)]
        print(f"\nEvaluating single checkpoint: {args.checkpoint}")
    
    # Evaluate
    all_results = []
    
    for checkpoint_path in checkpoints:
        print(f"\n{'#'*80}")
        print(f"CHECKPOINT: {checkpoint_path.name}")
        print(f"{'#'*80}")
        
        results = evaluate_single_checkpoint(
            str(checkpoint_path),
            datasets,
            evaluator,
            save_vis=args.save_vis,
            vis_dir=args.vis_dir,
            checkpoint_name=checkpoint_path.stem,
            num_vis_per_dataset=num_vis_per_dataset
        )
        
        all_results.append(results)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save JSON - convert numpy types first
    json_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(json_path, 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    print(f"\nSaved JSON results: {json_path}")
    
    # Create comparison table if multiple checkpoints
    if len(all_results) > 1:
        table_path = os.path.join(args.output_dir, 'checkpoint_comparison.txt')
        df = create_comparison_table(all_results, table_path)
        
        # Plot comparison
        plot_checkpoint_comparison(df, args.output_dir)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate RetiNet on multiple datasets')
    
    # Checkpoint
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to single checkpoint to evaluate')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/retinet_shapenet',
                       help='Directory containing checkpoints (for --eval_all)')
    parser.add_argument('--eval_all', action='store_true',
                       help='Evaluate all checkpoints in checkpoint_dir')
    
    # Datasets
    parser.add_argument('--eval_shapenet', action='store_true', default=True,
                       help='Evaluate on ShapeNet val split')
    parser.add_argument('--shapenet_dir', type=str, default=r'E:\dev\Shapenet_intrinsics',
                       help='Path to ShapeNet dataset')
    parser.add_argument('--eval_mit', action='store_true', default=False,
                       help='Evaluate on MIT dataset')
    parser.add_argument('--mit_dir', type=str, default='data',
                       help='Path to MIT dataset')
    
    # Data params
    parser.add_argument('--image_height', type=int, default=120)
    parser.add_argument('--image_width', type=int, default=160)
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save results')
    parser.add_argument('--save_vis', action='store_true',
                       help='Save visualizations')
    parser.add_argument('--vis_dir', type=str, default='evaluation_results/visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--num_vis_shapenet', type=int, default=5,
                       help='Number of ShapeNet visualizations to save (default: 5)')
    parser.add_argument('--num_vis_mit', type=int, default=5,
                       help='Number of MIT visualizations to save (None = all 20)')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    # Validation
    if not args.eval_all and args.checkpoint is None:
        parser.error("Must specify either --checkpoint or --eval_all")
    
    main(args)
