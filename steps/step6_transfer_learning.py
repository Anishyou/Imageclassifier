"""
=============================================================================
STEP 6: TRANSFER LEARNING
=============================================================================
In this step, you'll learn:
- What transfer learning is
- Why it's incredibly powerful
- How to use pre-trained models (ResNet, VGG, etc.)
- Fine-tuning vs feature extraction
=============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt

# =============================================================================
# WHAT IS TRANSFER LEARNING?
# =============================================================================
"""
Transfer Learning = using a model trained on one task to help with another task.

THE PROBLEM:
- Training from scratch requires lots of data and time
- CIFAR-10 has 50,000 images - that's actually small!
- Training our CNN from scratch gives ~70% accuracy

THE SOLUTION:
- Use a model already trained on ImageNet (14 million images!)
- These models have learned general features (edges, textures, shapes)
- We just adapt the final layer for our specific task

ANALOGY:
🎓 It's like hiring an experienced photographer:
   - They already know about lighting, composition, focus
   - You just teach them your specific style
   - Much faster than training a complete beginner!

RESULTS:
- Training from scratch: ~70% accuracy (10 epochs)
- Transfer learning: ~85-90% accuracy (5 epochs)
"""

# =============================================================================
# AVAILABLE PRE-TRAINED MODELS
# =============================================================================
"""
| Model           | Parameters | Top-1 Accuracy | Speed   |
|-----------------|------------|----------------|---------|
| ResNet18        | 11.7M      | 69.8%          | Fast    |
| ResNet50        | 25.6M      | 80.9%          | Medium  |
| VGG16           | 138M       | 71.6%          | Slow    |
| MobileNetV2     | 3.5M       | 72.0%          | Fastest |
| EfficientNet-B0 | 5.3M       | 77.7%          | Fast    |

For learning, we'll use ResNet18 - good balance of speed and accuracy!
"""

# =============================================================================
# DEVICE SETUP
# =============================================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🍎 Using Apple Silicon GPU")
else:
    device = torch.device("cpu")
    print("💻 Using CPU")

# =============================================================================
# DATA PREPARATION
# =============================================================================
print("\n📥 Preparing data...")

# IMPORTANT: Pre-trained models expect:
# 1. Images of size 224x224 (not 32x32!)
# 2. Normalized with ImageNet mean/std

# Training transforms (with augmentation)
train_transform = transforms.Compose([
    transforms.Resize(224),  # Resize to 224x224
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
])

# Test transforms (no augmentation)
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load datasets
train_dataset = torchvision.datasets.CIFAR10(
    root='../data', train=True, download=True, transform=train_transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root='../data', train=False, download=True, transform=test_transform
)

# Use smaller batch size (224x224 images use more memory!)
BATCH_SIZE = 32

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

print(f"✓ Training samples: {len(train_dataset)}")
print(f"✓ Test samples: {len(test_dataset)}")

# =============================================================================
# LOAD PRE-TRAINED MODEL
# =============================================================================
print("\n" + "="*60)
print("🏗️ LOADING PRE-TRAINED RESNET18")
print("="*60)

# Load ResNet18 with pre-trained ImageNet weights
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

print("\nOriginal ResNet18 structure (last few layers):")
print(f"  ... → AdaptiveAvgPool2d → Flatten → Linear(512, 1000)")
print(f"  The final layer outputs 1000 classes (ImageNet classes)")

# =============================================================================
# MODIFY FOR OUR TASK (10 CLASSES)
# =============================================================================
"""
ResNet18 was trained for 1000 ImageNet classes.
We need to modify the final layer for our 10 CIFAR-10 classes.

The key insight:
- Early layers learned general features (edges, textures)
- Final layer is task-specific (1000 ImageNet classes)
- We replace the final layer with our own!
"""

# Get the number of input features to the final layer
num_features = model.fc.in_features  # This is 512 for ResNet18
print(f"\n✓ Final layer input features: {num_features}")

# Replace the final layer
model.fc = nn.Linear(num_features, 10)  # 10 classes for CIFAR-10
print(f"✓ Replaced final layer: Linear({num_features}, 10)")

# Move model to device
model = model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Total parameters: {total_params:,}")

# =============================================================================
# TWO APPROACHES: FEATURE EXTRACTION vs FINE-TUNING
# =============================================================================
print("\n" + "="*60)
print("🎯 TWO APPROACHES TO TRANSFER LEARNING")
print("="*60)

print("""
1. FEATURE EXTRACTION (Freeze all except final layer)
   - Freeze pre-trained layers (don't update them)
   - Only train the new final layer
   - Faster training, less memory
   - Good when: limited data, limited compute
   
2. FINE-TUNING (Update all layers)
   - Update all layers during training
   - Pre-trained weights are just a starting point
   - Better accuracy, slower training
   - Good when: more data, more compute

We'll try BOTH approaches!
""")

# =============================================================================
# APPROACH 1: FEATURE EXTRACTION
# =============================================================================
print("\n" + "="*60)
print("📦 APPROACH 1: FEATURE EXTRACTION")
print("="*60)

def create_feature_extractor():
    """Create a model for feature extraction (freeze base layers)."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # FREEZE all layers except the final one
    for param in model.parameters():
        param.requires_grad = False  # Don't update these weights!
    
    # Replace and unfreeze final layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    # The new layer is trainable by default
    
    return model

# Create feature extractor
feature_model = create_feature_extractor().to(device)

# Count trainable parameters
trainable_params = sum(p.numel() for p in feature_model.parameters() if p.requires_grad)
print(f"✓ Trainable parameters: {trainable_params:,} (only the final layer!)")
print(f"✓ Frozen parameters: {total_params - trainable_params:,}")

# =============================================================================
# APPROACH 2: FINE-TUNING
# =============================================================================
print("\n" + "="*60)
print("🔧 APPROACH 2: FINE-TUNING")
print("="*60)

def create_finetune_model():
    """Create a model for fine-tuning (all layers trainable)."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # All layers are trainable by default
    # Just replace the final layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    
    return model

# Create fine-tuning model
finetune_model = create_finetune_model().to(device)

trainable_params_ft = sum(p.numel() for p in finetune_model.parameters() if p.requires_grad)
print(f"✓ Trainable parameters: {trainable_params_ft:,} (all layers!)")

# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_model(model, train_loader, test_loader, criterion, optimizer, 
                num_epochs=5, model_name="model"):
    """Train and evaluate the model."""
    
    train_losses = []
    test_accs = []
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{running_loss/len(train_loader):.3f}'})
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * correct / total
        train_losses.append(running_loss / len(train_loader))
        test_accs.append(test_acc)
        
        print(f"  → Test Accuracy: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f'../models/{model_name}_best.pth')
    
    return train_losses, test_accs, best_acc

# =============================================================================
# TRAIN FEATURE EXTRACTOR (QUICK DEMO)
# =============================================================================
print("\n" + "="*60)
print("🏋️ TRAINING FEATURE EXTRACTOR")
print("="*60)
print("(This is fast because we only train the final layer!)\n")

criterion = nn.CrossEntropyLoss()

# Only optimize the final layer (fc)
optimizer_fe = optim.Adam(feature_model.fc.parameters(), lr=0.001)

# Train for just 2 epochs as demo
fe_losses, fe_accs, fe_best = train_model(
    feature_model, train_loader, test_loader, 
    criterion, optimizer_fe, num_epochs=2, model_name="feature_extractor"
)

print(f"\n✓ Feature Extraction Best Accuracy: {fe_best:.2f}%")

# =============================================================================
# TRAIN FINE-TUNED MODEL (QUICK DEMO)
# =============================================================================
print("\n" + "="*60)
print("🏋️ TRAINING FINE-TUNED MODEL")
print("="*60)
print("(This takes longer but usually achieves better accuracy)\n")

# Use smaller learning rate for pre-trained layers
# Larger learning rate for the new final layer
optimizer_ft = optim.Adam([
    {'params': finetune_model.fc.parameters(), 'lr': 0.001},  # New layer
    {'params': [p for n, p in finetune_model.named_parameters() 
                if 'fc' not in n], 'lr': 0.0001}  # Pre-trained layers
])

# Train for just 2 epochs as demo
ft_losses, ft_accs, ft_best = train_model(
    finetune_model, train_loader, test_loader,
    criterion, optimizer_ft, num_epochs=2, model_name="finetuned"
)

print(f"\n✓ Fine-Tuning Best Accuracy: {ft_best:.2f}%")

# =============================================================================
# COMPARISON
# =============================================================================
print("\n" + "="*60)
print("📊 RESULTS COMPARISON")
print("="*60)

print(f"""
| Approach           | Trainable Params | Accuracy (2 epochs) |
|--------------------|------------------|---------------------|
| Our CNN (scratch)  | 596,234          | ~50-60%             |
| Feature Extraction | {trainable_params:,}           | {fe_best:.1f}%              |
| Fine-Tuning        | {trainable_params_ft:,}       | {ft_best:.1f}%              |

With more epochs (5-10), you can expect:
- Feature Extraction: ~80-85%
- Fine-Tuning: ~85-92%

That's a HUGE improvement over training from scratch!
""")

# =============================================================================
# OTHER PRE-TRAINED MODELS
# =============================================================================
print("\n" + "="*60)
print("🔄 OTHER PRE-TRAINED MODELS")
print("="*60)

print("""
You can easily swap to other models:

```python
# VGG16 (larger, slower)
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
model.classifier[-1] = nn.Linear(4096, 10)

# ResNet50 (deeper, more accurate)
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(2048, 10)

# MobileNetV2 (small, fast, mobile-friendly)
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[-1] = nn.Linear(1280, 10)

# EfficientNet (state-of-the-art efficiency)
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
model.classifier[-1] = nn.Linear(1280, 10)
```
""")

print("\n" + "="*60)
print("🎉 STEP 6 COMPLETE!")
print("="*60)
print("""
KEY TAKEAWAYS:
1. Transfer learning uses models pre-trained on large datasets
2. Pre-trained models have learned general image features
3. Feature Extraction: Freeze base, train only final layer (fast)
4. Fine-Tuning: Train all layers with pre-trained start (best accuracy)
5. Always use ImageNet normalization for pre-trained models
6. Resize images to 224x224 for most pre-trained models

Transfer learning is the most powerful technique for real-world 
image classification! It's used in almost all production systems.

Next: Run 'step7_learning_rate_scheduler.py' for smarter training!
""")



