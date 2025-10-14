import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import random
import pickle
from skimage.morphology import binary_erosion, square

class CGIntrinsicsDataset(Dataset):
    """
    CGIntrinsics Dataset - Fixed to maintain I = R × S relationship
    """
    
    def __init__(self, root_dir, list_dir, augment=True, image_size=(120, 160), max_samples=None):
        self.root_dir = root_dir
        self.augment = augment
        self.image_size = image_size
        
        list_file = os.path.join(list_dir, "img_batch.p")
        with open(list_file, "rb") as f:
            self.img_list = pickle.load(f)
        
        if max_samples is not None:
            self.img_list = self.img_list[:max_samples]
        
        print(f"Loaded {len(self.img_list)} images from CGIntrinsics")
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        
        # Full paths
        img_file = os.path.join(self.root_dir, "images", img_path)
        albedo_file = img_file[:-4] + "_albedo.png"
        mask_file = img_file[:-4] + "_mask.png"
        
        # Load images (keep in [0, 255] range from the start)
        srgb_img = np.array(Image.open(img_file).convert('RGB')).astype(np.float32)
        gt_albedo = np.array(Image.open(albedo_file).convert('RGB')).astype(np.float32)
        mask = np.array(Image.open(mask_file).convert('L')).astype(np.float32) / 255.0
        
        # Compute shading to maintain I = R × (S/255)
        # S = I / R, but we need to ensure proper range
        gt_albedo[gt_albedo < 1e-2] = 1e-2  # Avoid division by zero
        gt_shading = srgb_img / gt_albedo  # This gives S where I = R × S
        
        # Convert to grayscale shading
        gt_shading = np.mean(gt_shading, axis=2)
        
        # Clamp shading to reasonable range
        gt_shading = np.clip(gt_shading, 0.0, 255.0)
        
        # Process mask
        mask = binary_erosion(mask, square(11)).astype(np.float32)
        
        # Data augmentation
        if self.augment:
            h, w = srgb_img.shape[:2]
            
            if h > 20 and w > 20:
                y_start = random.randint(0, 9)
                x_start = random.randint(0, 9)
                
                srgb_img = srgb_img[y_start:h-10+y_start, x_start:w-10+x_start]
                gt_albedo = gt_albedo[y_start:h-10+y_start, x_start:w-10+x_start]
                gt_shading = gt_shading[y_start:h-10+y_start, x_start:w-10+x_start]
                mask = mask[y_start:h-10+y_start, x_start:w-10+x_start]
            
            if random.random() > 0.5:
                srgb_img = np.fliplr(srgb_img).copy()
                gt_albedo = np.fliplr(gt_albedo).copy()
                gt_shading = np.fliplr(gt_shading).copy()
                mask = np.fliplr(mask).copy()
        
        # Resize
        srgb_img = Image.fromarray(srgb_img.astype(np.uint8))
        gt_albedo = Image.fromarray(gt_albedo.astype(np.uint8))
        gt_shading = Image.fromarray(gt_shading.astype(np.uint8))
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        
        srgb_img = TF.resize(srgb_img, self.image_size)
        gt_albedo = TF.resize(gt_albedo, self.image_size)
        gt_shading = TF.resize(gt_shading, self.image_size)
        mask_img = TF.resize(mask_img, self.image_size)
        
        # Convert to tensors (already in [0, 255])
        srgb_img = torch.from_numpy(np.array(srgb_img)).permute(2, 0, 1).float()
        gt_albedo = torch.from_numpy(np.array(gt_albedo)).permute(2, 0, 1).float()
        gt_shading = torch.from_numpy(np.array(gt_shading)).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(np.array(mask_img)).unsqueeze(0).float() / 255.0
        
        return {
            'original': srgb_img,
            'reflectance': gt_albedo,
            'shading': gt_shading,
            'mask': mask_tensor,
            'name': img_path
        }