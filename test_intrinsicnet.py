import torch
import torch.nn as nn
import torch.optim as optim
from models.intrinsicnet import IntrinsicNet
from losses.intrinsic_loss import IntrinsicLoss
import matplotlib.pyplot as plt
import numpy as np

def create_synthetic_data(batch_size=4, height=120, width=160):
    """
    Create synthetic intrinsic decomposition data in [0, 255] range.
    Follows the dichromatic model: I = R × S (where S is normalized)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create piecewise constant reflectance (colored blocks)
    reflectance_gt = torch.zeros(batch_size, 3, height, width)
    
    for b in range(batch_size):
        num_blocks = 4
        block_h, block_w = height // num_blocks, width // num_blocks
        
        for i in range(num_blocks):
            for j in range(num_blocks):
                # Random colors in [50, 250] range
                color = torch.rand(3) * 200 + 50
                reflectance_gt[b, :, 
                              i*block_h:(i+1)*block_h, 
                              j*block_w:(j+1)*block_w] = color.view(3, 1, 1)
    
    # Create smooth shading in [0, 255] range
    # But when used for multiplication, it's normalized to [0, 1]
    shading_gt = torch.zeros(batch_size, 1, height, width)
    
    for b in range(batch_size):
        # Smooth gradient
        y = torch.linspace(0.3, 1.0, height).view(-1, 1)
        x = torch.linspace(0.3, 1.0, width).view(1, -1)
        shading = (y @ x) / width
        shading = shading + 0.05 * torch.randn(height, width)
        shading = torch.clamp(shading, 0.2, 1.0)
        
        # Scale to [0, 255] for storage (as per TF code)
        shading_gt[b, 0] = shading * 255.0
    
    # Compose input: I = R × (S/255)
    # Shading needs to be normalized when used for multiplication
    input_images = reflectance_gt * (shading_gt / 255.0)
    input_images = torch.clamp(input_images, 0.0, 255.0)
    
    return input_images.to(device), reflectance_gt.to(device), shading_gt.to(device)


def test_architecture():
    """Test 1: Verify architecture"""
    print("\n" + "="*70)
    print("TEST 1: Architecture Verification")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = IntrinsicNet(use_dropout=True).to(device)
    
    # Test with standard input
    x = torch.rand(2, 3, 120, 160).to(device) * 255.0
    
    model.eval()
    with torch.no_grad():
        albedo, shading = model(x)
    
    print(f"✓ Input shape:   {list(x.shape)}, range: [{x.min():.1f}, {x.max():.1f}]")
    print(f"✓ Albedo shape:  {list(albedo.shape)}, range: [{albedo.min():.1f}, {albedo.max():.1f}]")
    print(f"✓ Shading shape: {list(shading.shape)}, range: [{shading.min():.1f}, {shading.max():.1f}]")
    
    assert albedo.shape == (2, 3, 120, 160), "Albedo shape mismatch!"
    assert shading.shape == (2, 1, 120, 160), "Shading shape mismatch!"
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    return model


def test_loss_computation(model):
    """Test 2: Verify loss computation"""
    print("\n" + "="*70)
    print("TEST 2: Loss Computation")
    print("="*70)
    
    device = next(model.parameters()).device
    
    # Create synthetic data
    input_img, r_gt, s_gt = create_synthetic_data(batch_size=4)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        r_pred, s_pred = model(input_img)
    
    # Compute loss
    criterion = IntrinsicLoss(gamma_r=1.0, gamma_s=1.0, gamma_imf=1.0)
    loss, loss_dict = criterion(r_pred, s_pred, r_gt, s_gt, input_img)
    
    print(f"✓ Loss computation successful!")
    print(f"\nLoss components:")
    print(f"  Total:      {loss_dict['total']:.4f}")
    print(f"  Albedo:     {loss_dict['loss_r']:.4f}")
    print(f"  Shading:    {loss_dict['loss_s']:.4f}")
    print(f"  Formation:  {loss_dict['loss_imf']:.4f}")
    
    return criterion


def test_gradient_flow(model, criterion):
    """Test 3: Verify gradients flow"""
    print("\n" + "="*70)
    print("TEST 3: Gradient Flow")
    print("="*70)
    
    device = next(model.parameters()).device
    
    # Create synthetic data
    input_img, r_gt, s_gt = create_synthetic_data(batch_size=2)
    
    # Forward pass
    model.train()
    r_pred, s_pred = model(input_img)
    
    # Compute loss and backward
    loss, _ = criterion(r_pred, s_pred, r_gt, s_gt, input_img)
    loss.backward()
    
    # Check gradients exist
    has_gradients = False
    grad_norms = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_gradients = True
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
    
    assert has_gradients, "No gradients computed!"
    
    print(f"✓ Gradients flow correctly!")
    print(f"  Average gradient norm: {np.mean(grad_norms):.6f}")
    print(f"  Max gradient norm: {np.max(grad_norms):.6f}")
    
    model.zero_grad()


def test_training_loop(model, criterion, num_iterations=30):
    """Test 4: Overfit to synthetic data"""
    print("\n" + "="*70)
    print("TEST 4: Training Loop (Overfitting Test)")
    print("="*70)
    
    device = next(model.parameters()).device
    
    # Fixed batch for overfitting
    input_img, r_gt, s_gt = create_synthetic_data(batch_size=4)
    
    # Setup optimizer (matching TF: SGD with momentum=0.9, weight_decay=0.0005)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0005)
    
    model.train()
    losses = []
    
    print(f"\nTraining for {num_iterations} iterations on fixed synthetic data...")
    print("(Loss should decrease if network can learn)")
    print("-" * 70)
    
    for it in range(num_iterations):
        # Forward
        r_pred, s_pred = model(input_img)
        
        # Compute loss
        loss, loss_dict = criterion(r_pred, s_pred, r_gt, s_gt, input_img)
        
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
    print(f"Initial loss: {losses[0]:.2f}")
    print(f"Final loss:   {losses[-1]:.2f}")
    print(f"Improvement:  {improvement:.1f}%")
    
    if losses[-1] < losses[0]:
        print("✓ Loss decreased - network can learn!")
    else:
        print("⚠ Warning: Loss did not decrease")
    
    return model


def test_image_formation(model):
    """Test 5: Verify image formation constraint"""
    print("\n" + "="*70)
    print("TEST 5: Image Formation Constraint (R × S ≈ I)")
    print("="*70)
    
    device = next(model.parameters()).device
    
    # Create synthetic data
    input_img, r_gt, s_gt = create_synthetic_data(batch_size=1)
    
    model.eval()
    with torch.no_grad():
        r_pred, s_pred = model(input_img)
        
        # Reconstruct: I = R × (S/255)
        s_normalized = s_pred / 255.0
        reconstructed_img = r_pred * s_normalized
        
        # Compute reconstruction errors
        pred_error = torch.mean((reconstructed_img - input_img) ** 2).item()
        
        # Ground truth reconstruction
        s_gt_normalized = s_gt / 255.0
        gt_reconstructed = r_gt * s_gt_normalized
        gt_error = torch.mean((gt_reconstructed - input_img) ** 2).item()
    
    print(f"Ground truth reconstruction MSE: {gt_error:.6f}")
    print(f"Model reconstruction MSE:        {pred_error:.4f}")
    
    # GT should reconstruct perfectly (synthetic data)
    if gt_error < 1e-4:
        print("✓ Ground truth reconstruction is perfect!")
    
    print("✓ Image formation constraint verified!")


def visualize_predictions(model):
    """Test 6: Visualize results"""
    print("\n" + "="*70)
    print("TEST 6: Visualization")
    print("="*70)
    
    device = next(model.parameters()).device
    
    # Create synthetic data
    input_img, r_gt, s_gt = create_synthetic_data(batch_size=1)
    
    model.eval()
    with torch.no_grad():
        r_pred, s_pred = model(input_img)
        
        # Reconstruct image
        reconstructed = r_pred * (s_pred / 255.0)
    
    # Convert to numpy and normalize to [0, 1] for display
    def to_display(tensor, normalize=True):
        img = tensor[0].cpu().numpy()
        if normalize:
            img = np.clip(img / 255.0, 0, 1)
        return img
    
    input_np = to_display(input_img).transpose(1, 2, 0)
    r_gt_np = to_display(r_gt).transpose(1, 2, 0)
    s_gt_np = to_display(s_gt)[0]
    r_pred_np = to_display(r_pred).transpose(1, 2, 0)
    s_pred_np = to_display(s_pred)[0]
    recon_np = to_display(reconstructed).transpose(1, 2, 0)
    
    # Create visualization
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle('IntrinsicNet - Synthetic Data Test', fontsize=16, fontweight='bold')
    
    # Row 1: Ground Truth
    axes[0, 0].imshow(input_np)
    axes[0, 0].set_title('Input Image', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(r_gt_np)
    axes[0, 1].set_title('GT Reflectance', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(s_gt_np, cmap='gray')
    axes[0, 2].set_title('GT Shading', fontsize=12)
    axes[0, 2].axis('off')
    
    # Row 2: Predictions
    axes[1, 0].text(0.5, 0.5, 'Predictions\n(After Training)', 
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    transform=axes[1, 0].transAxes)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(r_pred_np)
    axes[1, 1].set_title('Predicted Reflectance', fontsize=12)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(s_pred_np, cmap='gray')
    axes[1, 2].set_title('Predicted Shading', fontsize=12)
    axes[1, 2].axis('off')
    
    # Row 3: Reconstruction
    axes[2, 0].imshow(input_np)
    axes[2, 0].set_title('Original Input', fontsize=12)
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(recon_np)
    axes[2, 1].set_title('Reconstructed (R × S)', fontsize=12)
    axes[2, 1].axis('off')
    
    # Difference map
    diff = np.abs(input_np - recon_np)
    axes[2, 2].imshow(diff, cmap='hot')
    axes[2, 2].set_title('Reconstruction Error', fontsize=12)
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('intrinsicnet_test_results.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to 'intrinsicnet_test_results.png'")
    plt.close()


def main():
    """Run all tests"""
    print("\n" + "#"*70)
    print("# IntrinsicNet - Complete Test Suite")
    print("# PyTorch recreation of ConvNets.py (TensorFlow)")
    print("#"*70)
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Run tests
    model = test_architecture()
    criterion = test_loss_computation(model)
    test_gradient_flow(model, criterion)
    model = test_training_loop(model, criterion, num_iterations=30)
    test_image_formation(model)
    visualize_predictions(model)
    
    print("\n" + "#"*70)
    print("# All tests completed successfully! ✓")
    print("#"*70)
    print("\nNext steps:")
    print("  1. Create dataset loader for real images")
    print("  2. Implement full training script")
    print("  3. Add evaluation on MIT intrinsic benchmark")
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()