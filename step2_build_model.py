"""
=============================================================================
STEP 2: BUILDING A CONVOLUTIONAL NEURAL NETWORK (CNN)
=============================================================================
In this step, you'll learn:
- Why CNNs are great for images
- How convolutional layers work
- How pooling reduces dimensions
- How to build a CNN in PyTorch
=============================================================================
"""

import torch
import torch.nn as nn

# =============================================================================
# WHY CNNs FOR IMAGES?
# =============================================================================
"""
Regular neural networks treat each pixel independently, which:
❌ Ignores spatial relationships (nearby pixels matter!)
❌ Requires too many parameters (32x32x3 = 3072 inputs per neuron!)

Convolutional Neural Networks (CNNs) solve this by:
✓ Using small filters that slide across the image
✓ Detecting features (edges, shapes, textures)
✓ Building up from simple to complex features

Layer by layer:
1. Conv layers → detect features (edges → shapes → objects)
2. Pooling layers → reduce size, keep important info
3. Fully connected layers → make final classification
"""

# =============================================================================
# OUR CNN ARCHITECTURE
# =============================================================================
"""
Input: 32x32x3 image (32 height, 32 width, 3 RGB channels)

Layer 1 (Conv): 32x32x3 → 32x32x32 (32 feature maps)
Layer 1 (Pool): 32x32x32 → 16x16x32 (halve the size)

Layer 2 (Conv): 16x16x32 → 16x16x64 (64 feature maps)
Layer 2 (Pool): 16x16x64 → 8x8x64 (halve again)

Layer 3 (Conv): 8x8x64 → 8x8x128 (128 feature maps)
Layer 3 (Pool): 8x8x128 → 4x4x128 (halve again)

Flatten: 4x4x128 = 2048 neurons
FC1: 2048 → 256 neurons
FC2: 256 → 10 classes (our output!)
"""


class ImageClassifier(nn.Module):
    """
    A CNN for classifying CIFAR-10 images.
    
    nn.Module is the base class for all neural networks in PyTorch.
    We must:
    1. Define layers in __init__
    2. Define forward pass in forward()
    """
    
    def __init__(self, num_classes=10):
        super(ImageClassifier, self).__init__()
        
        # =================================================================
        # CONVOLUTIONAL LAYERS
        # =================================================================
        # nn.Conv2d(in_channels, out_channels, kernel_size, padding)
        # - in_channels: number of input feature maps
        # - out_channels: number of output feature maps (filters)
        # - kernel_size: size of the sliding window (3x3 is common)
        # - padding: adds zeros around edges (keeps size same with padding=1)
        
        # First conv block: 3 → 32 channels
        self.conv1 = nn.Conv2d(
            in_channels=3,       # RGB input
            out_channels=32,     # 32 feature maps
            kernel_size=3,       # 3x3 filter
            padding=1            # Keep same size
        )
        self.bn1 = nn.BatchNorm2d(32)  # Normalizes outputs (faster training)
        
        # Second conv block: 32 → 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third conv block: 64 → 128 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # =================================================================
        # POOLING LAYER
        # =================================================================
        # MaxPool2d takes the maximum value in each 2x2 region
        # This reduces dimensions by half (32→16→8→4)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # =================================================================
        # FULLY CONNECTED LAYERS
        # =================================================================
        # After 3 pooling operations: 32 → 16 → 8 → 4
        # So we have 4 x 4 x 128 = 2048 features
        self.fc1 = nn.Linear(4 * 4 * 128, 256)  # 2048 → 256
        self.fc2 = nn.Linear(256, num_classes)  # 256 → 10 classes
        
        # =================================================================
        # ACTIVATION & REGULARIZATION
        # =================================================================
        self.relu = nn.ReLU()           # Activation function
        self.dropout = nn.Dropout(0.5)  # Randomly zeros 50% during training
    
    def forward(self, x):
        """
        Forward pass: defines how data flows through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)
        
        Returns:
            Output tensor of shape (batch_size, 10)
        """
        # Block 1: Conv → BatchNorm → ReLU → Pool
        # Shape: (batch, 3, 32, 32) → (batch, 32, 16, 16)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        
        # Block 2: Conv → BatchNorm → ReLU → Pool
        # Shape: (batch, 32, 16, 16) → (batch, 64, 8, 8)
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        
        # Block 3: Conv → BatchNorm → ReLU → Pool
        # Shape: (batch, 64, 8, 8) → (batch, 128, 4, 4)
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        # Flatten: (batch, 128, 4, 4) → (batch, 2048)
        x = x.view(x.size(0), -1)  # -1 means "figure out this dimension"
        
        # Fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))  # (batch, 2048) → (batch, 256)
        x = self.fc2(x)                            # (batch, 256) → (batch, 10)
        
        return x


# =============================================================================
# LET'S TEST OUR MODEL!
# =============================================================================
if __name__ == "__main__":
    # Create the model
    model = ImageClassifier(num_classes=10)
    
    # Print architecture
    print("="*60)
    print("📐 MODEL ARCHITECTURE")
    print("="*60)
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Total parameters: {total_params:,}")
    print(f"📊 Trainable parameters: {trainable_params:,}")
    
    # Test with dummy input
    print("\n" + "="*60)
    print("🧪 TESTING WITH DUMMY INPUT")
    print("="*60)
    
    # Create a fake batch of 4 images
    dummy_input = torch.randn(4, 3, 32, 32)  # (batch, channels, height, width)
    print(f"Input shape:  {dummy_input.shape}")
    
    # Forward pass
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Output (raw logits for first image): {output[0].detach()}")
    
    # Apply softmax to get probabilities
    probabilities = torch.softmax(output[0], dim=0)
    print(f"Probabilities: {probabilities.detach()}")
    print(f"Sum of probs: {probabilities.sum().item():.4f} (should be 1.0)")
    
    print("\n" + "="*60)
    print("🎉 STEP 2 COMPLETE!")
    print("="*60)
    print("""
KEY TAKEAWAYS:
1. CNNs use convolutional layers to detect image features
2. Pooling layers reduce spatial dimensions
3. Fully connected layers make the final classification
4. forward() defines how data flows through the network
5. Output is 10 raw scores (logits) - one per class

Next: Run 'step3_train_model.py' to train the model!
""")



