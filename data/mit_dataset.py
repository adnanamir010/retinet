import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random

class MITIntrinsicDataset(Dataset):
    """
    MIT Intrinsic Images Dataset
    20 object-centered images with ground truth intrinsic decomposition
    """
    
    def __init__(self, root_dir, transform=None, augment=False, image_size=(120, 160)):
        """
        Args:
            root_dir: Path to data directory containing object folders
            transform: Optional transform to apply
            augment: Whether to apply data augmentation
            image_size: (height, width) to resize images to
        """
        self.root_dir = root_dir
        self.transform = transform
        self.augment = augment
        self.image_size = image_size
        
        # Get all object folders (filter out __pycache__ and other non-data dirs)
        self.objects = []
        for d in os.listdir(root_dir):
            obj_path = os.path.join(root_dir, d)
            # Check if it's a directory and contains original.png
            if os.path.isdir(obj_path) and os.path.exists(os.path.join(obj_path, 'original.png')):
                self.objects.append(d)
        
        self.objects.sort()
        
        if len(self.objects) == 0:
            raise ValueError(f"No valid object folders found in {root_dir}")
        
        print(f"Found {len(self.objects)} objects in {root_dir}")
    
    def __len__(self):
        return len(self.objects)
    
    def __getitem__(self, idx):
        obj_name = self.objects[idx]
        obj_dir = os.path.join(self.root_dir, obj_name)
        
        # Load images
        original = Image.open(os.path.join(obj_dir, 'original.png')).convert('RGB')
        reflectance = Image.open(os.path.join(obj_dir, 'reflectance.png')).convert('RGB')
        shading = Image.open(os.path.join(obj_dir, 'shading.png')).convert('L')
        
        # Resize to target size
        original = TF.resize(original, self.image_size)
        reflectance = TF.resize(reflectance, self.image_size)
        shading = TF.resize(shading, self.image_size)
        
        # Apply augmentation if enabled
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                original = TF.hflip(original)
                reflectance = TF.hflip(reflectance)
                shading = TF.hflip(shading)
            
            # Random vertical flip
            if random.random() > 0.5:
                original = TF.vflip(original)
                reflectance = TF.vflip(reflectance)
                shading = TF.vflip(shading)
            
            # Random shift (as per paper: [-20, 20] pixels)
            max_shift = 20
            shift_h = random.randint(-max_shift, max_shift)
            shift_w = random.randint(-max_shift, max_shift)
            
            if shift_h != 0 or shift_w != 0:
                original = TF.affine(original, angle=0, translate=(shift_w, shift_h), 
                                    scale=1.0, shear=0)
                reflectance = TF.affine(reflectance, angle=0, translate=(shift_w, shift_h),
                                       scale=1.0, shear=0)
                shading = TF.affine(shading, angle=0, translate=(shift_w, shift_h),
                                   scale=1.0, shear=0)
        
        # Convert to tensors and scale to [0, 255]
        original = TF.to_tensor(original) * 255.0
        reflectance = TF.to_tensor(reflectance) * 255.0
        shading = TF.to_tensor(shading) * 255.0
        
        return {
            'original': original,
            'reflectance': reflectance,
            'shading': shading,
            'name': obj_name
        }


if __name__ == "__main__":
    dataset = MITIntrinsicDataset(root_dir='../data', augment=True)
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"\nSample: {sample['name']}")
    print(f"Original: {sample['original'].shape}, range: [{sample['original'].min():.1f}, {sample['original'].max():.1f}]")
    print(f"Reflectance: {sample['reflectance'].shape}, range: [{sample['reflectance'].min():.1f}, {sample['reflectance'].max():.1f}]")
    print(f"Shading: {sample['shading'].shape}, range: [{sample['shading'].min():.1f}, {sample['shading'].max():.1f}]")
    
    print("\nLoading all samples to verify dataset...")
    for i in range(len(dataset)):
        sample = dataset[i]
        print(f"  {i+1}/{len(dataset)}: {sample['name']}")