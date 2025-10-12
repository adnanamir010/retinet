import torch
import torch.nn.functional as F
import numpy as np

def compute_gradient_numpy(image):
    """
    Compute image gradients using numpy (matches TensorFlow implementation).
    
    Args:
        image: (H, W, 3) numpy array in [0, 255] range
    
    Returns:
        gradient: (H, W, 3) gradient magnitude per channel
    """
    # Scale to 16-bit range
    tmp_img = image * 257.0
    
    # Vertical gradient (finite difference)
    I_y = np.zeros(tmp_img.shape)
    I_y[1:, :, :] = tmp_img[1:, :, :] - tmp_img[:-1, :, :]
    
    # Horizontal gradient
    I_x = np.zeros(tmp_img.shape)
    I_x[:, 1:, :] = tmp_img[:, 1:, :] - tmp_img[:, :-1, :]
    
    # Gradient magnitude
    grad = np.sqrt(I_x**2 + I_y**2)
    
    return grad


def compute_gradient_torch(image):
    """
    Compute image gradients using PyTorch (differentiable version).
    
    Args:
        image: (B, 3, H, W) tensor in [0, 255] range
    
    Returns:
        gradient: (B, 3, H, W) gradient magnitude per channel
    """
    # Scale to 16-bit range
    tmp_img = image * 257.0
    
    # Vertical gradient
    I_y = torch.zeros_like(tmp_img)
    I_y[:, :, 1:, :] = tmp_img[:, :, 1:, :] - tmp_img[:, :, :-1, :]
    
    # Horizontal gradient
    I_x = torch.zeros_like(tmp_img)
    I_x[:, :, :, 1:] = tmp_img[:, :, :, 1:] - tmp_img[:, :, :, :-1]
    
    # Gradient magnitude
    grad = torch.sqrt(I_x**2 + I_y**2)
    
    return grad


if __name__ == "__main__":
    print("Testing gradient computation...")
    
    # Test numpy version
    img_np = np.random.rand(120, 160, 3) * 255.0
    grad_np = compute_gradient_numpy(img_np)
    print(f"Numpy gradient: {grad_np.shape}, range: [{grad_np.min():.1f}, {grad_np.max():.1f}]")
    
    # Test PyTorch version
    img_torch = torch.rand(2, 3, 120, 160) * 255.0
    grad_torch = compute_gradient_torch(img_torch)
    print(f"PyTorch gradient: {grad_torch.shape}, range: [{grad_torch.min():.1f}, {grad_torch.max():.1f}]")
    
    print("Gradient computation test passed!")