"""
=============================================================================
STEP 5: DATA AUGMENTATION
=============================================================================
In this step, you'll learn:
- What data augmentation is
- Why it helps prevent overfitting
- Common augmentation techniques
- How to implement them in PyTorch
=============================================================================
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# WHAT IS DATA AUGMENTATION?
# =============================================================================
"""
Data augmentation = artificially creating more training data by applying
random transformations to existing images.

WHY DO WE NEED IT?
- More training data = better model (usually)
- Helps prevent overfitting (model memorizing training data)
- Makes model robust to variations (rotation, lighting, etc.)

EXAMPLE:
One cat image can become:
- Flipped cat 🐱↔️
- Rotated cat 🐱↻
- Cropped cat 🐱✂️
- Brightened cat 🐱☀️
- All of these are still a cat!

The model learns: "A cat is a cat, no matter how it's positioned"
"""

# =============================================================================
# COMMON AUGMENTATION TECHNIQUES
# =============================================================================
"""
| Augmentation        | What it does                    | When to use          |
|---------------------|--------------------------------|----------------------|
| RandomHorizontalFlip | Flip image left-right          | Most cases           |
| RandomRotation      | Rotate by random angle          | When rotation varies |
| RandomCrop          | Take random portion of image    | Most cases           |
| ColorJitter         | Change brightness/contrast/etc  | When lighting varies |
| RandomAffine        | Scale, rotate, translate        | Complex augmentation |
| RandomErasing       | Randomly erase part of image    | Occlusion robustness |
"""

# =============================================================================
# LET'S IMPLEMENT AUGMENTATION!
# =============================================================================

# BASIC transforms (what we used before - no augmentation)
basic_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# AUGMENTED transforms (for training only!)
augmented_transform = transforms.Compose([
    # ------------------------------------------------------------------
    # AUGMENTATION 1: Random Horizontal Flip
    # ------------------------------------------------------------------
    # 50% chance to flip the image horizontally
    # A flipped cat is still a cat!
    transforms.RandomHorizontalFlip(p=0.5),
    
    # ------------------------------------------------------------------
    # AUGMENTATION 2: Random Rotation
    # ------------------------------------------------------------------
    # Rotate randomly between -15 and +15 degrees
    # Objects in real life aren't always perfectly aligned
    transforms.RandomRotation(degrees=15),
    
    # ------------------------------------------------------------------
    # AUGMENTATION 3: Random Crop with Padding
    # ------------------------------------------------------------------
    # Add 4 pixels of padding, then crop back to 32x32
    # This simulates slight shifts in position
    transforms.RandomCrop(32, padding=4),
    
    # ------------------------------------------------------------------
    # AUGMENTATION 4: Color Jitter
    # ------------------------------------------------------------------
    # Randomly change brightness, contrast, saturation
    # Real-world images have different lighting conditions
    transforms.ColorJitter(
        brightness=0.2,  # ±20% brightness
        contrast=0.2,    # ±20% contrast
        saturation=0.2,  # ±20% saturation
        hue=0.1          # ±10% hue shift
    ),
    
    # ------------------------------------------------------------------
    # Now the standard transforms
    # ------------------------------------------------------------------
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# TEST transforms (NO augmentation - we want consistent evaluation)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

print("="*60)
print("📸 DATA AUGMENTATION DEMO")
print("="*60)

# =============================================================================
# VISUALIZE AUGMENTATION
# =============================================================================

def show_augmentations(image, transform, num_augmentations=8):
    """Show multiple augmented versions of the same image."""
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle("Same image with different augmentations", fontsize=14)
    
    for i, ax in enumerate(axes.flat):
        if i == 0:
            # Show original
            ax.imshow(image)
            ax.set_title("Original", fontsize=10)
        else:
            # Apply augmentation (random each time!)
            augmented = transform(image)
            
            # Unnormalize for display
            augmented = augmented * torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
            augmented = augmented + torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
            augmented = torch.clamp(augmented, 0, 1)
            
            # Convert to displayable format
            img = augmented.permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.set_title(f"Augmented {i}", fontsize=10)
        
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_demo.png', dpi=150)
    plt.show()
    print("\n✓ Augmentation demo saved to 'augmentation_demo.png'")

# Load one image from CIFAR-10 (as PIL Image, without transforms)
raw_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=None
)

image, label = raw_dataset[0]
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

print(f"\nShowing augmentations of a {classes[label]}...")
show_augmentations(image, augmented_transform)

# =============================================================================
# TRAINING WITH AUGMENTATION
# =============================================================================
print("\n" + "="*60)
print("🏋️ HOW TO USE IN TRAINING")
print("="*60)

print("""
IMPORTANT: Only augment TRAINING data, not TEST data!

Why?
- Training: We want variety to help the model generalize
- Testing: We want consistent, fair evaluation

Code example:
""")

print("""
```python
# Training dataset - WITH augmentation
train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=augmented_transform  # 👈 Use augmented transform!
)

# Test dataset - WITHOUT augmentation
test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=test_transform  # 👈 Use basic transform!
)
```
""")

# =============================================================================
# ADVANCED: AUTOAUGMENT
# =============================================================================
print("\n" + "="*60)
print("🤖 ADVANCED: AUTOAUGMENT")
print("="*60)

print("""
PyTorch also includes AutoAugment - a learned augmentation policy!

Google trained a neural network to find the best augmentation strategy
for different datasets. You can use it easily:

```python
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

auto_augment_transform = transforms.Compose([
    AutoAugment(policy=AutoAugmentPolicy.CIFAR10),  # Uses CIFAR-10 policy
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])
```

This can improve accuracy by 1-2% without manual tuning!
""")

# =============================================================================
# AUGMENTATION COMPARISON
# =============================================================================
print("\n" + "="*60)
print("📊 EXPECTED IMPROVEMENT")
print("="*60)

print("""
| Configuration         | Expected Accuracy |
|-----------------------|-------------------|
| No augmentation       | ~70-73%           |
| With augmentation     | ~75-78%           |
| With AutoAugment      | ~77-80%           |

Augmentation typically adds 3-5% accuracy improvement!

WHY DOES IT HELP?
1. More "virtual" training data
2. Model sees variations it might encounter in real life
3. Prevents overfitting (can't memorize augmented images)
4. Forces model to learn robust features
""")

print("\n" + "="*60)
print("🎉 STEP 5 COMPLETE!")
print("="*60)
print("""
KEY TAKEAWAYS:
1. Data augmentation creates variations of training images
2. Common techniques: flip, rotate, crop, color jitter
3. Only augment TRAINING data, not test data
4. Can improve accuracy by 3-5%
5. AutoAugment provides learned policies

Next: Run 'step6_transfer_learning.py' to use pre-trained models!
""")



