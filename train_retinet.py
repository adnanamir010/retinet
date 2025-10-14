import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt

from models.retinet import RetiNet
from losses.intrinsic_loss import IntrinsicLoss
from data.cgintrinsics_dataset import CGIntrinsicsDataset
from data.mit_dataset import MITIntrinsicDataset
from utils.metrics import compute_all_metrics


def save_visualizations(model, dataloader, device, save_dir, epoch, num_samples=5):
    """Save visual results"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_samples:
                break
            
            original = batch['original'].to(device)
            reflectance_gt = batch['reflectance'].to(device)
            shading_gt = batch['shading'].to(device)
            name = batch['name'][0] if isinstance(batch['name'], list) else batch['name']
            
            reflectance_pred, shading_pred, _, _ = model(original)
            
            def to_numpy(tensor):
                return np.clip(tensor[0].cpu().numpy() / 255.0, 0, 1)
            
            orig_np = to_numpy(original).transpose(1, 2, 0)
            r_gt_np = to_numpy(reflectance_gt).transpose(1, 2, 0)
            s_gt_np = to_numpy(shading_gt)[0]
            r_pred_np = to_numpy(reflectance_pred).transpose(1, 2, 0)
            s_pred_np = to_numpy(shading_pred)[0]
            
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            
            axes[0, 0].imshow(orig_np)
            axes[0, 0].set_title('Input')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(r_gt_np)
            axes[0, 1].set_title('GT Reflectance')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(s_gt_np, cmap='gray')
            axes[0, 2].set_title('GT Shading')
            axes[0, 2].axis('off')
            
            axes[1, 0].imshow(orig_np)
            axes[1, 0].set_title('Input')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(r_pred_np)
            axes[1, 1].set_title('Pred Reflectance')
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(s_pred_np, cmap='gray')
            axes[1, 2].set_title('Pred Shading')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            safe_name = name.replace('/', '_').replace('\\', '_')
            plt.savefig(os.path.join(save_dir, f'epoch_{epoch}_{safe_name}.png'), dpi=100)
            plt.close()


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer, global_step):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    total_loss_r = 0.0
    total_loss_s = 0.0
    total_loss_imf = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        original = batch['original'].to(device)
        reflectance_gt = batch['reflectance'].to(device)
        shading_gt = batch['shading'].to(device)
        
        reflectance_pred, shading_pred, _, _ = model(original)
        
        loss, loss_dict = criterion(reflectance_pred, shading_pred, 
                                    reflectance_gt, shading_gt, original)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss_dict['total']
        total_loss_r += loss_dict['loss_r']
        total_loss_s += loss_dict['loss_s']
        total_loss_imf += loss_dict['loss_imf']
        
        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.2f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
        })
        
        if batch_idx % 10 == 0:
            writer.add_scalar('Train/Loss_Total_Iter', loss_dict['total'], global_step)
            writer.add_scalar('Train/Loss_Reflectance_Iter', loss_dict['loss_r'], global_step)
            writer.add_scalar('Train/Loss_Shading_Iter', loss_dict['loss_s'], global_step)
            writer.add_scalar('Train/Loss_ImageFormation_Iter', loss_dict['loss_imf'], global_step)
            writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], global_step)
        
        global_step += 1
    
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_loss_r = total_loss_r / num_batches
    avg_loss_s = total_loss_s / num_batches
    avg_loss_imf = total_loss_imf / num_batches
    
    return avg_loss, avg_loss_r, avg_loss_s, avg_loss_imf, global_step


def validate(model, dataloader, criterion, device, epoch, writer):
    """Validate on MIT dataset"""
    model.eval()
    
    total_loss = 0.0
    all_metrics = {
        'mse_reflectance': 0.0,
        'mse_shading': 0.0,
        'lmse_reflectance': 0.0,
        'lmse_shading': 0.0,
        'dssim_reflectance': 0.0,
        'dssim_shading': 0.0
    }
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            original = batch['original'].to(device)
            reflectance_gt = batch['reflectance'].to(device)
            shading_gt = batch['shading'].to(device)
            
            reflectance_pred, shading_pred, _, _ = model(original)
            
            loss, _ = criterion(reflectance_pred, shading_pred, 
                               reflectance_gt, shading_gt, original)
            total_loss += loss.item()
            
            metrics = compute_all_metrics(reflectance_pred, shading_pred,
                                         reflectance_gt, shading_gt)
            
            for key in all_metrics.keys():
                all_metrics[key] += metrics[key]
    
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    
    for key in all_metrics.keys():
        all_metrics[key] /= num_batches
    
    # Compute averages
    all_metrics['mse_avg'] = (all_metrics['mse_reflectance'] + all_metrics['mse_shading']) / 2
    all_metrics['lmse_avg'] = (all_metrics['lmse_reflectance'] + all_metrics['lmse_shading']) / 2
    all_metrics['dssim_avg'] = (all_metrics['dssim_reflectance'] + all_metrics['dssim_shading']) / 2
    
    # Log to tensorboard
    writer.add_scalar('Val/Loss_Total', avg_loss, epoch)
    writer.add_scalar('Val/MSE_Reflectance', all_metrics['mse_reflectance'], epoch)
    writer.add_scalar('Val/MSE_Shading', all_metrics['mse_shading'], epoch)
    writer.add_scalar('Val/MSE_Average', all_metrics['mse_avg'], epoch)
    writer.add_scalar('Val/LMSE_Reflectance', all_metrics['lmse_reflectance'], epoch)
    writer.add_scalar('Val/LMSE_Shading', all_metrics['lmse_shading'], epoch)
    writer.add_scalar('Val/LMSE_Average', all_metrics['lmse_avg'], epoch)
    writer.add_scalar('Val/DSSIM_Reflectance', all_metrics['dssim_reflectance'], epoch)
    writer.add_scalar('Val/DSSIM_Shading', all_metrics['dssim_shading'], epoch)
    writer.add_scalar('Val/DSSIM_Average', all_metrics['dssim_avg'], epoch)
    
    return avg_loss, all_metrics


def save_checkpoint(model, optimizer, epoch, loss, metrics, checkpoint_dir, filename):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }
    
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, filepath)


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    metrics = checkpoint.get('metrics', {})
    
    print(f"Loaded checkpoint from epoch {epoch}, loss: {loss:.4f}")
    
    return epoch, loss, metrics


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Training dataset: CGIntrinsics
    train_dataset = CGIntrinsicsDataset(
        root_dir=args.data_dir,
        list_dir=args.list_dir,
        augment=args.augment,
        image_size=(args.image_height, args.image_width),
        max_samples=args.max_samples
    )
    
    # Validation dataset: MIT
    val_dataset = MITIntrinsicDataset(
        root_dir=args.val_dir,
        augment=False,
        image_size=(args.image_height, args.image_width)
    )
    
    print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = RetiNet(
        use_dropout=args.use_dropout,
        compute_gradients=True
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = IntrinsicLoss(
        gamma_r=args.gamma_r,
        gamma_s=args.gamma_s,
        gamma_imf=args.gamma_imf
    )
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler (polynomial decay as per paper)
    lr_lambda = lambda epoch: (1 - epoch / args.epochs) ** 0.9
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Save config
    config = vars(args)
    os.makedirs(args.log_dir, exist_ok=True)
    with open(os.path.join(args.log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Visualization directory
    vis_dir = os.path.join(args.log_dir, 'visualizations')
    
    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    
    if args.resume:
        start_epoch, _, _ = load_checkpoint(model, optimizer, args.resume)
        start_epoch += 1
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Iterations per epoch: {len(train_loader)}")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_loss_r, train_loss_s, train_loss_imf, global_step = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1, writer, global_step
        )
        
        # Log epoch training metrics
        writer.add_scalar('Train/Loss_Total_Epoch', train_loss, epoch)
        writer.add_scalar('Train/Loss_Reflectance_Epoch', train_loss_r, epoch)
        writer.add_scalar('Train/Loss_Shading_Epoch', train_loss_s, epoch)
        writer.add_scalar('Train/Loss_ImageFormation_Epoch', train_loss_imf, epoch)
        
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"  Reflectance: {train_loss_r:.4f}")
        print(f"  Shading: {train_loss_s:.4f}")
        print(f"  Image Formation: {train_loss_imf:.4f}")
        
        # Validate on MIT dataset
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, epoch, writer
        )
        
        print(f"\nVal Loss (MIT): {val_loss:.4f}")
        print(f"Metrics:")
        print(f"  MSE - R: {val_metrics['mse_reflectance']:.6f}, S: {val_metrics['mse_shading']:.6f}, Avg: {val_metrics['mse_avg']:.6f}")
        print(f"  LMSE - R: {val_metrics['lmse_reflectance']:.6f}, S: {val_metrics['lmse_shading']:.6f}, Avg: {val_metrics['lmse_avg']:.6f}")
        print(f"  DSSIM - R: {val_metrics['dssim_reflectance']:.6f}, S: {val_metrics['dssim_shading']:.6f}, Avg: {val_metrics['dssim_avg']:.6f}")
        
        # Save visualizations
        if (epoch + 1) % args.vis_freq == 0:
            save_visualizations(model, val_loader, device, vis_dir, epoch+1)
        
        # Step scheduler
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_metrics,
                args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'
            )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_metrics,
                args.checkpoint_dir, 'best_model.pth'
            )
            print(f"New best model! Val Loss: {val_loss:.4f}")
    
    # Save final model
    save_checkpoint(
        model, optimizer, args.epochs-1, val_loss, val_metrics,
        args.checkpoint_dir, 'final_model.pth'
    )
    
    writer.close()
    print("\nTraining completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RetiNet')
    
    # Data
    parser.add_argument('--data_dir', type=str, 
                       default='E:/dev/intrinsics_final/intrinsics_final',
                       help='Path to CGIntrinsics intrinsics_final directory')
    parser.add_argument('--list_dir', type=str,
                       default='E:/dev/intrinsics_final/intrinsics_final/train_list',
                       help='Path to train_list directory with img_batch.p')
    parser.add_argument('--val_dir', type=str, default='data',
                       help='Path to MIT dataset for validation')
    parser.add_argument('--max_samples', type=int, default=None,
                   help='Limit training samples for testing (None = use all)')
    parser.add_argument('--image_height', type=int, default=120)
    parser.add_argument('--image_width', type=int, default=160)
    parser.add_argument('--augment', action='store_true', default=True)
    
    # Model
    parser.add_argument('--use_dropout', action='store_true', default=True)
    
    # Loss weights
    parser.add_argument('--gamma_r', type=float, default=1.0)
    parser.add_argument('--gamma_s', type=float, default=1.0)
    parser.add_argument('--gamma_imf', type=float, default=1.0)
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/retinet')
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None)
    
    # Logging
    parser.add_argument('--log_dir', type=str, default='runs/retinet')
    parser.add_argument('--vis_freq', type=int, default=5)
    
    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    
    main(args)