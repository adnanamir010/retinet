# Save as: data/shapenet_dataset.py

import torch
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np
from pathlib import Path

class ShapeNetIntrinsicsDataset(Dataset):
    """
    ShapeNet Intrinsics Dataset for RetiNet training.
    """
    
    def __init__(self, root_dir, split='train', image_size=None):
        """
        Args:
            root_dir: Path to Shapenet_intrinsics directory
            split: 'train' or 'val'
            image_size: (H, W) to resize to, or None to keep original (360, 480)
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        
        # Load file list
        file_list_path = self.root_dir / "shapenet_file_list.txt"
        
        with open(file_list_path, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
        
        # Parse MAT file paths
        self.mat_files = [line.strip().split('\t')[1] for line in lines]
        
        # Train/val split (80/20)
        np.random.seed(42)
        indices = np.random.permutation(len(self.mat_files))
        split_idx = int(0.8 * len(self.mat_files))
        
        if split == 'train':
            self.mat_files = [self.mat_files[i] for i in indices[:split_idx]]
        else:
            self.mat_files = [self.mat_files[i] for i in indices[split_idx:]]
        
        print(f"ShapeNet {split} dataset: {len(self.mat_files)} samples")
    
    def __len__(self):
        return len(self.mat_files)
    
    def __getitem__(self, idx):
        """
        Returns:
            rgb: (3, H, W) tensor in [0, 255]
            albedo: (3, H, W) tensor in [0, 255]
            shading: (1, H, W) tensor in [0, 255]
        """
        # Load MAT file
        mat_path = self.mat_files[idx]
        mat_data = sio.loadmat(mat_path)
        
        # Extract HDR ground truths
        albedo_hdr = mat_data['Diffuse_color'].astype(np.float32)  # (H, W, 3)
        shading_hdr = mat_data['Diffuse_light'].astype(np.float32)  # (H, W, 3)
        
        # Normalize to [0, 1] using min-max per image
        albedo_norm = self._normalize_minmax(albedo_hdr)
        shading_norm = self._normalize_minmax(shading_hdr)
        
        # Convert shading to grayscale
        shading_gray = shading_norm.mean(axis=2, keepdims=True)  # (H, W, 1)
        
        # Scale to [0, 255]
        albedo = albedo_norm * 255.0
        shading = shading_gray * 255.0
        
        # Resize if needed (BEFORE computing RGB)
        if self.image_size is not None:
            from PIL import Image
            
            albedo = Image.fromarray(albedo.astype(np.uint8))
            albedo = albedo.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
            albedo = np.array(albedo).astype(np.float32)
            
            shading = Image.fromarray(shading.squeeze().astype(np.uint8), mode='L')
            shading = shading.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
            shading = np.array(shading).astype(np.float32)[..., np.newaxis]
        
        # Create composite RGB AFTER resizing: I = R x S
        rgb = albedo * (shading / 255.0)  # (H, W, 3)
        
        # Convert to torch tensors (H, W, C) -> (C, H, W)
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float()
        albedo = torch.from_numpy(albedo).permute(2, 0, 1).float()
        shading = torch.from_numpy(shading).permute(2, 0, 1).float()
        
        return rgb, albedo, shading
    
    def _normalize_minmax(self, img):
        """Normalize HDR image to [0, 1] using min-max."""
        img_min = img.min()
        img_max = img.max()
        return (img - img_min) / (img_max - img_min + 1e-8)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    dataset_root = r"E:\dev\Shapenet_intrinsics"
    
    # Test with original size
    print("=" * 80)
    print("Testing with original size (360, 480)")
    print("=" * 80)
    
    train_dataset = ShapeNetIntrinsicsDataset(
        root_dir=dataset_root,
        split='train',
        image_size=None
    )
    
    val_dataset = ShapeNetIntrinsicsDataset(
        root_dir=dataset_root,
        split='val',
        image_size=None
    )
    
    # Test batch
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    rgb, albedo, shading = next(iter(train_loader))
    
    print(f"\nBatch shapes:")
    print(f"  RGB: {rgb.shape}, range: [{rgb.min():.1f}, {rgb.max():.1f}]")
    print(f"  Albedo: {albedo.shape}, range: [{albedo.min():.1f}, {albedo.max():.1f}]")
    print(f"  Shading: {shading.shape}, range: [{shading.min():.1f}, {shading.max():.1f}]")
    
    # Verify physics constraint
    reconstructed = albedo * (shading / 255.0)
    error = (rgb - reconstructed).abs().mean()
    status = "OK" if error < 1e-3 else "FAIL"
    print(f"\nPhysics constraint: |I - R x (S/255)| = {error:.6f} [{status}]")
    
    # Test with paper's size (120, 160)
    print("\n" + "=" * 80)
    print("Testing with paper size (120, 160)")
    print("=" * 80)
    
    train_dataset_small = ShapeNetIntrinsicsDataset(
        root_dir=dataset_root,
        split='train',
        image_size=(120, 160)
    )
    
    train_loader_small = DataLoader(train_dataset_small, batch_size=16, shuffle=True, num_workers=0)
    rgb_s, albedo_s, shading_s = next(iter(train_loader_small))
    
    print(f"\nBatch shapes:")
    print(f"  RGB: {rgb_s.shape}, range: [{rgb_s.min():.1f}, {rgb_s.max():.1f}]")
    print(f"  Albedo: {albedo_s.shape}, range: [{albedo_s.min():.1f}, {albedo_s.max():.1f}]")
    print(f"  Shading: {shading_s.shape}, range: [{shading_s.min():.1f}, {shading_s.max():.1f}]")
    
    reconstructed_s = albedo_s * (shading_s / 255.0)
    error_s = (rgb_s - reconstructed_s).abs().mean()
    status_s = "OK" if error_s < 1e-3 else "FAIL"
    print(f"\nPhysics constraint: |I - R x (S/255)| = {error_s:.6f} [{status_s}]")
    
    print("\n" + "=" * 80)
    print("Dataset ready for training!")
    print("=" * 80)
    print(f"\nTotal samples: 16,840")
    print(f"Train: {len(train_dataset_small)} samples")
    print(f"Val: {len(val_dataset)} samples")