"""
=============================================================================
STEP 8: USING YOUR OWN IMAGES
=============================================================================
In this step, you'll learn:
- How to create a custom dataset
- How to organize your own images
- How to train on your own data
- How to make predictions on any image
=============================================================================
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

# =============================================================================
# ORGANIZING YOUR OWN IMAGES
# =============================================================================
"""
To use your own images, organize them like this:

my_dataset/
├── train/
│   ├── cats/          <- All training cat images
│   │   ├── cat1.jpg
│   │   ├── cat2.jpg
│   │   └── ...
│   ├── dogs/          <- All training dog images
│   │   ├── dog1.jpg
│   │   └── ...
│   └── birds/         <- All training bird images
│       └── ...
└── test/
    ├── cats/          <- All test cat images
    ├── dogs/
    └── birds/

The folder name becomes the class label!
"""

# =============================================================================
# METHOD 1: USING IMAGEFOLDER (EASIEST!)
# =============================================================================
print("="*60)
print("📁 METHOD 1: ImageFolder (Recommended)")
print("="*60)

print("""
PyTorch's ImageFolder automatically loads images from folders!

```python
from torchvision.datasets import ImageFolder

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset - folder structure defines classes!
train_dataset = ImageFolder(
    root='my_dataset/train',
    transform=transform
)

# Check the classes
print(train_dataset.classes)  # ['birds', 'cats', 'dogs']
print(train_dataset.class_to_idx)  # {'birds': 0, 'cats': 1, 'dogs': 2}

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Now train like normal!
for images, labels in train_loader:
    # images: (32, 3, 224, 224)
    # labels: (32,) - values 0, 1, or 2
    pass
```
""")

# =============================================================================
# METHOD 2: CUSTOM DATASET CLASS
# =============================================================================
print("\n" + "="*60)
print("🔧 METHOD 2: Custom Dataset Class")
print("="*60)

print("""
For more control, create a custom Dataset:

```python
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
```
""")


class CustomImageDataset(Dataset):
    """
    Custom dataset for loading images from a list of paths.
    
    Args:
        image_paths: List of paths to images
        labels: List of integer labels (same length as image_paths)
        transform: Optional transforms to apply
    """
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


# =============================================================================
# PREDICT ON ANY IMAGE
# =============================================================================
print("\n" + "="*60)
print("🔮 PREDICTING ON ANY IMAGE")
print("="*60)

# Device setup
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def load_trained_model(model_path, num_classes, use_pretrained=True):
    """
    Load a trained model for inference.
    
    Args:
        model_path: Path to saved model weights
        num_classes: Number of output classes
        use_pretrained: Whether the model uses ResNet architecture
    
    Returns:
        Loaded model ready for inference
    """
    if use_pretrained:
        # For transfer learning models
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        # For our custom CNN
        from step2_build_model import ImageClassifier
        model = ImageClassifier(num_classes=num_classes)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model


def predict_image(model, image_path, class_names, show_image=True):
    """
    Predict the class of a single image.
    
    Args:
        model: Trained model
        image_path: Path to the image file
        class_names: List of class names
        show_image: Whether to display the image
    
    Returns:
        predicted_class: String name of predicted class
        confidence: Float confidence percentage
        all_probs: Dict of all class probabilities
    """
    # Transforms for prediction
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = probabilities.max(0)
    
    predicted_class = class_names[predicted_idx.item()]
    confidence = confidence.item() * 100
    
    all_probs = {
        class_names[i]: probabilities[i].item() * 100 
        for i in range(len(class_names))
    }
    
    # Display results
    if show_image:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Show image
        ax1.imshow(image)
        ax1.set_title(f"Predicted: {predicted_class}\nConfidence: {confidence:.1f}%")
        ax1.axis('off')
        
        # Show probabilities
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        classes = [x[0] for x in sorted_probs[:5]]
        probs = [x[1] for x in sorted_probs[:5]]
        
        colors = ['green' if c == predicted_class else 'steelblue' for c in classes]
        ax2.barh(classes, probs, color=colors)
        ax2.set_xlabel('Probability (%)')
        ax2.set_title('Top 5 Predictions')
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.show()
    
    return predicted_class, confidence, all_probs


print("""
Example usage:

```python
# Define your classes
class_names = ['cat', 'dog', 'bird']

# Load your trained model
model = load_trained_model('my_model.pth', num_classes=3)

# Predict on any image!
predicted, confidence, probs = predict_image(
    model, 
    'path/to/my_photo.jpg', 
    class_names
)

print(f"This is a {predicted} ({confidence:.1f}% confident)")
```
""")

# =============================================================================
# COMPLETE EXAMPLE: TRAINING ON YOUR OWN DATA
# =============================================================================
print("\n" + "="*60)
print("📚 COMPLETE TRAINING EXAMPLE")
print("="*60)

print("""
Here's a complete script to train on your own images:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# ==================== CONFIGURATION ====================
DATA_DIR = 'my_dataset'  # Your dataset folder
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 3  # Change based on your classes!

# ==================== DATA LOADING ====================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = ImageFolder(f'{DATA_DIR}/train', transform=train_transform)
test_dataset = ImageFolder(f'{DATA_DIR}/test', transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Classes: {train_dataset.classes}")
print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# ==================== MODEL ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use pre-trained ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==================== TRAINING ====================
best_acc = 0.0

for epoch in range(NUM_EPOCHS):
    # Train
    model.train()
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Evaluate
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
    
    acc = 100. * correct / total
    print(f'Epoch {epoch+1}: Accuracy = {acc:.2f}%')
    
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'  Saved new best model!')

print(f'\\nTraining complete! Best accuracy: {best_acc:.2f}%')
```
""")

# =============================================================================
# TIPS FOR YOUR OWN DATASET
# =============================================================================
print("\n" + "="*60)
print("💡 TIPS FOR YOUR OWN DATASET")
print("="*60)

print("""
1. DATA QUANTITY
   - Aim for at least 100 images per class
   - More data = better results (usually)
   - Balance classes (similar number for each)

2. DATA QUALITY
   - Clear, focused images
   - Variety of angles, lighting, backgrounds
   - Remove duplicates and low-quality images

3. SPLITTING DATA
   - Typically: 80% train, 20% test
   - Or: 70% train, 15% validation, 15% test
   - Keep splits balanced

4. AUGMENTATION
   - Critical for small datasets!
   - Can 5-10x your effective training data
   - Match augmentations to real-world variations

5. TRANSFER LEARNING
   - Almost always use pre-trained models
   - Even with small datasets (50-100 images), works well
   - Fine-tune all layers for best results

6. COMMON ISSUES
   - Overfitting: Use augmentation, dropout, early stopping
   - Poor accuracy: Get more data, try different model
   - Slow training: Reduce image size, use smaller model
""")

print("\n" + "="*60)
print("🎉 STEP 8 COMPLETE!")
print("="*60)
print("""
CONGRATULATIONS! 🎊

You've completed the full PyTorch Image Classification course!

WHAT YOU'VE LEARNED:
1. ✅ Data loading and transforms
2. ✅ Building CNNs from scratch
3. ✅ The training loop (forward → loss → backward → update)
4. ✅ Evaluation and making predictions
5. ✅ Data augmentation
6. ✅ Transfer learning with pre-trained models
7. ✅ Learning rate scheduling and early stopping
8. ✅ Training on your own images

NEXT STEPS:
- Try other datasets (ImageNet, MNIST, Fashion-MNIST)
- Explore other architectures (ResNet50, EfficientNet)
- Learn object detection (YOLO, Faster R-CNN)
- Learn image segmentation (U-Net, DeepLab)
- Deploy your model (Flask API, mobile app)

Happy deep learning! 🚀
""")



