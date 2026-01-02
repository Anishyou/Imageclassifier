"""
=============================================================================
STEP 1: DATA LOADING IN PYTORCH
=============================================================================
In this step, you'll learn:
- What transforms are and why we use them
- How to load datasets using torchvision
- What DataLoaders do and why they're important
=============================================================================
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# PART 1: TRANSFORMS
# =============================================================================
# Transforms are operations we apply to our images before feeding them to
# the neural network. Common transforms include:
# - ToTensor(): Converts image to PyTorch tensor (required!)
# - Normalize(): Scales pixel values to a standard range (helps training)
# - RandomHorizontalFlip(): Data augmentation (makes model more robust)

# For CIFAR-10, images are 32x32 pixels with 3 color channels (RGB)
# We normalize using mean and std calculated from the dataset
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts PIL Image to tensor (0-255 -> 0.0-1.0)
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),  # Mean for each RGB channel
        std=(0.2470, 0.2435, 0.2616)    # Std for each RGB channel
    )
])

# =============================================================================
# PART 2: LOADING THE DATASET
# =============================================================================
# CIFAR-10 contains 60,000 images in 10 classes:
# airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
# - 50,000 training images
# - 10,000 test images

print("Downloading CIFAR-10 dataset (this may take a moment)...")

# Training dataset
train_dataset = torchvision.datasets.CIFAR10(
    root='./data',           # Where to store the data
    train=True,              # Get the training set
    download=True,           # Download if not already present
    transform=transform      # Apply our transforms
)

# Test dataset
test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,             # Get the test set
    download=True,
    transform=transform
)

# Class names for CIFAR-10
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

print(f"\n✓ Training samples: {len(train_dataset)}")
print(f"✓ Test samples: {len(test_dataset)}")
print(f"✓ Number of classes: {len(classes)}")

# =============================================================================
# PART 3: DATA LOADERS
# =============================================================================
# DataLoaders help us:
# - Load data in batches (more efficient than one image at a time)
# - Shuffle data (important for training - prevents learning order)
# - Use multiple CPU cores for loading (num_workers)

BATCH_SIZE = 64  # Process 64 images at a time

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,            # Shuffle training data each epoch
    num_workers=0            # Use 0 for Windows compatibility
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,           # No need to shuffle test data
    num_workers=0
)

print(f"\n✓ Training batches: {len(train_loader)}")
print(f"✓ Test batches: {len(test_loader)}")

# =============================================================================
# PART 4: VISUALIZING THE DATA
# =============================================================================
# Let's look at some sample images to understand our data

def show_images(images, labels):
    """Display a grid of images with their labels."""
    # Unnormalize images for display
    images = images * torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    images = images + torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    images = torch.clamp(images, 0, 1)
    
    # Create a grid
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            # Convert from (C, H, W) to (H, W, C) for matplotlib
            img = images[i].permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.set_title(classes[labels[i]], fontsize=12)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=150)
    plt.show()
    print("\n✓ Sample images saved to 'sample_images.png'")

# Get one batch of training images
images, labels = next(iter(train_loader))

print(f"\n📦 Batch shape: {images.shape}")
print(f"   - Batch size: {images.shape[0]}")
print(f"   - Channels (RGB): {images.shape[1]}")  
print(f"   - Height: {images.shape[2]} pixels")
print(f"   - Width: {images.shape[3]} pixels")

# Show 8 sample images
show_images(images[:8], labels[:8])

print("\n" + "="*60)
print("🎉 STEP 1 COMPLETE!")
print("="*60)
print("""
KEY TAKEAWAYS:
1. Transforms prepare images for the neural network
2. Datasets hold the actual image data
3. DataLoaders serve batches of data during training
4. Batch shape is: (batch_size, channels, height, width)

Next: Run 'step2_build_model.py' to build the CNN architecture!
""")



