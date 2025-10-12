import torch
import torch.nn as nn
import torch.optim as optim
from models.retinet import RetiNet
from losses.intrinsic_loss import IntrinsicLoss
import matplotlib.pyplot as plt
import numpy as np

def create_synthetic_data(batch_size=4, height=120, width=160):
    """
    Create synthetic intrinsic decomposition data in [0, 255] range.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create piecewise constant reflectance
    reflectance_gt = torch.zeros(batch_size, 3, height, width)
    
    for b in range(batch_size):
        num_blocks = 4
        block_h, block_w = height // num_blocks, width // num_blocks
        
        for i in range(num_blocks):
            for j in range(num_blocks):
                color = torch.rand(3) * 200 + 50  # [50, 250]
                reflectance_gt[b, :, 
                              i*block_h:(i+1)*block_h, 
                              j*block_w:(j+1)*block_w] = color.view(3, 1, 1)
    
    # Create smooth shading
    shading_gt = torch.zeros(batch_size, 1, height, width)
    
    for b in range(batch_size):
        y = torch.linspace(0.3, 1.0, height).view(-1, 1)
        x = torch.linspace(0.3, 1.0, width).view(1, -1)
        shading = (y @ x) / width
        shading = shading + 0.05 * torch.randn(height, width)
        shading = torch.clamp(shading, 0.2, 1.0)
        shading_gt[b, 0] = shading * 255.0
    
    # Compose input: I = R × (S/255)
    input_images = reflectance_gt * (shading_gt / 255.0)
    input_images = torch.clamp(input_images, 0.0, 255.0)
    
    return input_images.to(device), reflectance_gt.to(device), shading_gt.to(device)


def test_architecture():
    """Test 1: Architecture verification"""
    print("\n" + "="*70)
    print("TEST 1: RetiNet Architecture")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = RetiNet(use_dropout=True, compute_gradients=True).to(device)
    
    # Test input
    rgb = torch.rand(2, 3, 120, 160).to(device) * 255.0
    
    model.eval()
    with torch.no_grad():
        albedo, shading, a_grad, s_grad = model(rgb)
    
    print(f"\n✓ Input shape: {list(rgb.shape)}")
    print(f"✓ Albedo shape: {list(albedo.shape)}")
    print(f"✓ Shading shape: {list(shading.shape)}")
    print(f"✓ Albedo grad shape: {list(a_grad.shape)}")
    print(f"✓ Shading grad shape: {list(s_grad.shape)}")
    
    # Parameter counts
    stage1_params = sum(p.numel() for p in model.stage1.parameters())
    stage2_params = sum(p.numel() for p in model.stage2.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\nParameters:")
    print(f"  Stage 1 (Gradient Separation): {stage1_params:,}")
    print(f"  Stage 2 (Reintegration): {stage2_params:,}")
    print(f"  Total: {total_params:,}")
    
    return model


def test_gradient_computation():
    """Test 2: Gradient computation"""
    print("\n" + "="*70)
    print("TEST 2: Gradient Computation")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = RetiNet(use_dropout=False, compute_gradients=True).to(device)
    model.eval()
    
    # Test image
    rgb = torch.rand(1, 3, 120, 160).to(device) * 255.0
    
    # Compute gradients
    with torch.no_grad():
        rgb_grad = model._compute_gradient(rgb)
    
    print(f"RGB image: {rgb.shape}, range: [{rgb.min():.1f}, {rgb.max():.1f}]")
    print(f"RGB gradient: {rgb_grad.shape}, range: [{rgb_grad.min():.1f}, {rgb_grad.max():.1f}]")
    print("✓ Gradient computation successful!")


def test_two_stage_pipeline():
    """Test 3: Two-stage pipeline"""
    print("\n" + "="*70)
    print("TEST 3: Two-Stage Pipeline")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = RetiNet(use_dropout=False, compute_gradients=True).to(device)
    model.eval()
    
    rgb = torch.rand(2, 3, 120, 160).to(device) * 255.0
    
    with torch.no_grad():
        # Test Stage 1 only
        a_grad, s_grad = model.forward_stage1_only(rgb)
        print(f"✓ Stage 1 output:")
        print(f"  Albedo grad: {a_grad.shape}, range: [{a_grad.min():.1f}, {a_grad.max():.1f}]")
        print(f"  Shading grad: {s_grad.shape}, range: [{s_grad.min():.1f}, {s_grad.max():.1f}]")
        
        # Test Stage 2 only
        albedo, shading = model.forward_stage2_only(rgb, a_grad, s_grad)
        print(f"\n✓ Stage 2 output:")
        print(f"  Albedo: {albedo.shape}, range: [{albedo.min():.1f}, {albedo.max():.1f}]")
        print(f"  Shading: {shading.shape}, range: [{shading.min():.1f}, {shading.max():.1f}]")
        
        # Test full pipeline
        albedo_full, shading_full, _, _ = model(rgb)
        print(f"\n✓ Full pipeline output:")
        print(f"  Albedo: {albedo_full.shape}, range: [{albedo_full.min():.1f}, {albedo_full.max():.1f}]")
        print(f"  Shading: {shading_full.shape}, range: [{shading_full.min():.1f}, {shading_full.max():.1f}]")


def test_training_loop(num_iterations=20):
    """Test 4: Training loop"""
    print("\n" + "="*70)
    print("TEST 4: Training Loop")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model and optimizer
    model = RetiNet(use_dropout=True, compute_gradients=True).to(device)
    criterion = IntrinsicLoss(gamma_r=1.0, gamma_s=1.0, gamma_imf=1.0)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0005)
    
    # Fixed batch for overfitting test
    input_img, r_gt, s_gt = create_synthetic_data(batch_size=4)
    
    model.train()
    losses = []
    
    print(f"\nTraining for {num_iterations} iterations...")
    print("-" * 70)
    
    for it in range(num_iterations):
        # Forward
        albedo, shading, _, _ = model(input_img)
        
        # Compute loss
        loss, loss_dict = criterion(albedo, shading, r_gt, s_gt, input_img)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss_dict['total'])
        
        if it % 5 == 0 or it == num_iterations - 1:
            print(f"Iter {it+1:3d} | Total: {loss_dict['total']:8.2f} | "
                  f"R: {loss_dict['loss_r']:8.2f} | "
                  f"S: {loss_dict['loss_s']:8.2f} | "
                  f"IMF: {loss_dict['loss_imf']:8.2f}")
    
    improvement = (losses[0] - losses[-1]) / losses[0] * 100
    print("-" * 70)
    print(f"Initial: {losses[0]:.2f} → Final: {losses[-1]:.2f}")
    print(f"Improvement: {improvement:.1f}%")
    
    if losses[-1] < losses[0]:
        print("✓ Loss decreased - network can learn!")
    
    return model


def visualize_results(model):
    """Test 5: Visualization"""
    print("\n" + "="*70)
    print("TEST 5: Visualization")
    print("="*70)
    
    device = next(model.parameters()).device
    
    # Create test data
    input_img, r_gt, s_gt = create_synthetic_data(batch_size=1)
    
    model.eval()
    with torch.no_grad():
        r_pred, s_pred, r_grad, s_grad = model(input_img)
    
    # Convert to numpy
    def to_numpy(tensor):
        return tensor[0].cpu().numpy()
    
    input_np = to_numpy(input_img).transpose(1, 2, 0) / 255.0
    r_gt_np = to_numpy(r_gt).transpose(1, 2, 0) / 255.0
    s_gt_np = to_numpy(s_gt)[0] / 255.0
    r_pred_np = to_numpy(r_pred).transpose(1, 2, 0) / 255.0
    s_pred_np = to_numpy(s_pred)[0] / 255.0
    r_grad_np = to_numpy(r_grad).transpose(1, 2, 0) / 360.63
    s_grad_np = to_numpy(s_grad)[0] / 360.63
    
    # Create visualization
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('RetiNet - Two-Stage Pipeline Test', fontsize=16, fontweight='bold')
    
    # Row 1: Ground Truth
    axes[0, 0].imshow(np.clip(input_np, 0, 1))
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(np.clip(r_gt_np, 0, 1))
    axes[0, 1].set_title('GT Reflectance')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(s_gt_np, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title('GT Shading')
    axes[0, 2].axis('off')
    
    axes[0, 3].text(0.5, 0.5, 'Ground Truth', ha='center', va='center',
                    fontsize=14, fontweight='bold', transform=axes[0, 3].transAxes)
    axes[0, 3].axis('off')
    
    # Row 2: Stage 1 - Gradients
    axes[1, 0].text(0.5, 0.5, 'Stage 1:\nGradient\nSeparation', ha='center', va='center',
                    fontsize=12, fontweight='bold', transform=axes[1, 0].transAxes)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(np.clip(r_grad_np, 0, 1))
    axes[1, 1].set_title('Predicted Albedo Gradient')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(np.clip(s_grad_np, 0, 1), cmap='gray')
    axes[1, 2].set_title('Predicted Shading Gradient')
    axes[1, 2].axis('off')
    
    axes[1, 3].axis('off')
    
    # Row 3: Stage 2 - Final Output
    axes[2, 0].text(0.5, 0.5, 'Stage 2:\nReintegration', ha='center', va='center',
                    fontsize=12, fontweight='bold', transform=axes[2, 0].transAxes)
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(np.clip(r_pred_np, 0, 1))
    axes[2, 1].set_title('Final Predicted Reflectance')
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(np.clip(s_pred_np, 0, 1), cmap='gray')
    axes[2, 2].set_title('Final Predicted Shading')
    axes[2, 2].axis('off')
    
    # Reconstruction
    recon = r_pred_np * np.expand_dims(s_pred_np, -1)
    axes[2, 3].imshow(np.clip(recon, 0, 1))
    axes[2, 3].set_title('Reconstructed Image')
    axes[2, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('retinet_test_results.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to 'retinet_test_results.png'")
    plt.close()


def main():
    """Run all tests"""
    print("\n" + "#"*70)
    print("# RetiNet - Complete Two-Stage Pipeline Test")
    print("# PyTorch Recreation of ConvNets1.py + ConvNets2.py")
    print("#"*70)
    
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Run tests
    model = test_architecture()
    test_gradient_computation()
    test_two_stage_pipeline()
    model = test_training_loop(num_iterations=20)
    visualize_results(model)
    
    print("\n" + "#"*70)
    print("# All RetiNet tests completed successfully! ✓")
    print("#"*70)
    print("\nNext steps:")
    print("  1. Train Stage 1 on gradient separation task")
    print("  2. Train Stage 2 on reintegration task")
    print("  3. Evaluate on MIT intrinsic benchmark")
    print("  4. Compare IntrinsicNet vs RetiNet performance")
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()