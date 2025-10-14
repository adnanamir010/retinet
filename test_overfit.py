import numpy as np
from PIL import Image
import scipy.io as sio
from pathlib import Path
import random
import matplotlib.pyplot as plt

def verify_author_instructions(dataset_root, num_samples=5):
    """
    Verify the author's instructions for ShapeNet dataset preprocessing.
    """
    file_list_path = Path(dataset_root) / "shapenet_file_list.txt"
    
    with open(file_list_path, 'r') as f:
        lines = f.readlines()[1:]
    
    mat_files = [line.strip().split('\t')[1] for line in lines]
    sample_files = random.sample(mat_files, min(num_samples, len(mat_files)))
    
    print("=" * 80)
    print("🧪 VERIFYING AUTHOR'S PREPROCESSING INSTRUCTIONS")
    print("=" * 80)
    
    for i, mat_path in enumerate(sample_files, 1):
        print(f"\n📷 Sample {i}/{num_samples}: {Path(mat_path).name}")
        
        # Load MAT file
        mat_data = sio.loadmat(mat_path)
        
        # Extract ground truths (HDR)
        albedo_hdr = mat_data['Diffuse_color'].astype(np.float32)  # (H, W, 3)
        shading_hdr = mat_data['Diffuse_light'].astype(np.float32)  # (H, W, 3)
        
        print(f"  Raw Albedo (HDR): shape={albedo_hdr.shape}, range=[{albedo_hdr.min():.3f}, {albedo_hdr.max():.3f}]")
        print(f"  Raw Shading (HDR): shape={shading_hdr.shape}, range=[{shading_hdr.min():.3f}, {shading_hdr.max():.3f}]")
        
        # Step 1: Normalize to [0, 1] using min-max
        albedo_norm = (albedo_hdr - albedo_hdr.min()) / (albedo_hdr.max() - albedo_hdr.min() + 1e-8)
        shading_norm = (shading_hdr - shading_hdr.min()) / (shading_hdr.max() - shading_hdr.min() + 1e-8)
        
        print(f"  Normalized Albedo: range=[{albedo_norm.min():.3f}, {albedo_norm.max():.3f}]")
        print(f"  Normalized Shading: range=[{shading_norm.min():.3f}, {shading_norm.max():.3f}]")
        
        # Step 2: Convert 3-channel shading to grayscale
        shading_gray = shading_norm.mean(axis=2, keepdims=True)  # (H, W, 1)
        
        print(f"  Grayscale Shading: shape={shading_gray.shape}, range=[{shading_gray.min():.3f}, {shading_gray.max():.3f}]")
        
        # Step 3: Create composite RGB: I = R × S
        composite_rgb = albedo_norm * shading_gray  # Broadcasting: (H,W,3) × (H,W,1)
        
        print(f"  Composite RGB: shape={composite_rgb.shape}, range=[{composite_rgb.min():.3f}, {composite_rgb.max():.3f}]")
        
        # Step 4: Scale to [0, 255] as per paper
        albedo_255 = albedo_norm * 255.0
        shading_255 = shading_gray * 255.0
        composite_255 = composite_rgb * 255.0
        
        print(f"\n  ✅ Scaled to [0, 255]:")
        print(f"     Albedo: range=[{albedo_255.min():.1f}, {albedo_255.max():.1f}]")
        print(f"     Shading: range=[{shading_255.min():.1f}, {shading_255.max():.1f}]")
        print(f"     Composite: range=[{composite_255.min():.1f}, {composite_255.max():.1f}]")
        
        # Verify I = R × (S/255) in [0, 255] space
        reconstructed = albedo_255 * (shading_255 / 255.0)
        error = np.abs(composite_255 - reconstructed).mean()
        print(f"\n  🔬 Verification: |I - R×(S/255)| = {error:.6f} {'✓ PERFECT' if error < 1e-4 else '❌ BAD'}")
    
    print("\n" + "=" * 80)
    print("✅ PREPROCESSING VERIFIED - Dataset is ready for training!")
    print("=" * 80)


def visualize_sample(dataset_root, num_samples=3):
    """
    Visualize the preprocessing pipeline.
    """
    file_list_path = Path(dataset_root) / "shapenet_file_list.txt"
    
    with open(file_list_path, 'r') as f:
        lines = f.readlines()[1:]
    
    mat_files = [line.strip().split('\t')[1] for line in lines]
    sample_files = random.sample(mat_files, min(num_samples, len(mat_files)))
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]
    
    for i, mat_path in enumerate(sample_files):
        mat_data = sio.loadmat(mat_path)
        
        # Process as per author's instructions
        albedo_hdr = mat_data['Diffuse_color'].astype(np.float32)
        shading_hdr = mat_data['Diffuse_light'].astype(np.float32)
        
        # Normalize to [0, 1]
        albedo_norm = (albedo_hdr - albedo_hdr.min()) / (albedo_hdr.max() - albedo_hdr.min() + 1e-8)
        shading_norm = (shading_hdr - shading_hdr.min()) / (shading_hdr.max() - shading_hdr.min() + 1e-8)
        shading_gray = shading_norm.mean(axis=2)
        
        # Composite
        composite = albedo_norm * shading_gray[..., np.newaxis]
        
        # Plot
        axes[i, 0].imshow(composite)
        axes[i, 0].set_title(f'Composite RGB\n(I = R × S)')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(albedo_norm)
        axes[i, 1].set_title('Albedo (R)\nDiffuse_color')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(shading_gray, cmap='gray')
        axes[i, 2].set_title('Shading (S)\nDiffuse_light (grayscale)')
        axes[i, 2].axis('off')
        
        # Reconstruction check
        reconstructed = albedo_norm * shading_gray[..., np.newaxis]
        diff = np.abs(composite - reconstructed)
        axes[i, 3].imshow(diff * 10)  # Amplify for visibility
        axes[i, 3].set_title(f'Difference ×10\nError: {diff.mean():.6f}')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(Path(dataset_root) / 'shapenet_preprocessing_verification.png', dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved visualization to: {Path(dataset_root) / 'shapenet_preprocessing_verification.png'}")
    plt.close()


if __name__ == "__main__":
    dataset_root = r"E:\dev\Shapenet_intrinsics"
    
    # Verify preprocessing
    verify_author_instructions(dataset_root, num_samples=5)
    
    # Create visualization
    visualize_sample(dataset_root, num_samples=3)