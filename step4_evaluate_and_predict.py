"""
=============================================================================
STEP 4: EVALUATION AND PREDICTION
=============================================================================
In this step, you'll learn:
- How to load a saved model
- How to make predictions on new images
- How to analyze model performance per class
- Visualizing correct and incorrect predictions
=============================================================================
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from step2_build_model import ImageClassifier

# =============================================================================
# CONFIGURATION
# =============================================================================
BATCH_SIZE = 64

# Device setup
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Class names
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# =============================================================================
# LOAD THE TRAINED MODEL
# =============================================================================
print("\n📥 Loading trained model...")

model = ImageClassifier(num_classes=10)
model = model.to(device)

# Load the saved checkpoint
try:
    checkpoint = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Model loaded from epoch {checkpoint['epoch']+1}")
    print(f"✓ Best accuracy was: {checkpoint['best_acc']:.2f}%")
except FileNotFoundError:
    print("❌ No saved model found! Please run step3_train_model.py first.")
    exit()

# Set to evaluation mode
model.eval()

# =============================================================================
# LOAD TEST DATA
# =============================================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

# =============================================================================
# EVALUATE ON ENTIRE TEST SET
# =============================================================================
print("\n" + "="*60)
print("📊 EVALUATING ON TEST SET")
print("="*60)

correct = 0
total = 0
class_correct = [0] * 10
class_total = [0] * 10

# Collect all predictions for confusion matrix
all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = outputs.max(1)
        
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Per-class accuracy
        for i in range(len(labels)):
            label = labels[i].item()
            class_total[label] += 1
            if predicted[i] == labels[i]:
                class_correct[label] += 1
        
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

overall_acc = 100. * correct / total
print(f"\n🎯 Overall Test Accuracy: {overall_acc:.2f}%")

print("\n📋 Per-Class Accuracy:")
print("-" * 40)
for i in range(10):
    acc = 100. * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
    bar = "█" * int(acc // 5) + "░" * (20 - int(acc // 5))
    print(f"{classes[i]:12s}: {bar} {acc:5.1f}%")

# =============================================================================
# VISUALIZE PREDICTIONS
# =============================================================================
def unnormalize(img):
    """Convert normalized tensor to displayable image."""
    img = img * torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    img = img + torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    return torch.clamp(img, 0, 1)

def predict_and_show(model, test_loader, device, num_images=16):
    """Show predictions on random test images."""
    model.eval()
    
    # Get a batch
    images, labels = next(iter(test_loader))
    images = images.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        confidences, predicted = probs.max(1)
    
    # Plot
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            img = unnormalize(images[i].cpu())
            img = img.permute(1, 2, 0).numpy()
            
            ax.imshow(img)
            
            pred_class = classes[predicted[i]]
            true_class = classes[labels[i]]
            conf = confidences[i].item() * 100
            
            # Green if correct, red if wrong
            color = 'green' if predicted[i] == labels[i] else 'red'
            ax.set_title(
                f"Pred: {pred_class} ({conf:.0f}%)\nTrue: {true_class}",
                color=color,
                fontsize=10
            )
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150)
    plt.show()
    print("\n✓ Predictions saved to 'predictions.png'")

print("\n" + "="*60)
print("🔍 SAMPLE PREDICTIONS")
print("="*60)
predict_and_show(model, test_loader, device)

# =============================================================================
# SHOW CONFUSION MATRIX
# =============================================================================
def plot_confusion_matrix(all_labels, all_predictions, classes):
    """Plot a confusion matrix."""
    from collections import Counter
    
    # Create confusion matrix
    n_classes = len(classes)
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    
    for true, pred in zip(all_labels, all_predictions):
        confusion[true][pred] += 1
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(confusion, cmap='Blues')
    
    # Labels
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)
    
    # Add text annotations
    for i in range(n_classes):
        for j in range(n_classes):
            text = ax.text(j, i, confusion[i, j],
                          ha='center', va='center',
                          color='white' if confusion[i, j] > confusion.max()/2 else 'black',
                          fontsize=8)
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()
    print("\n✓ Confusion matrix saved to 'confusion_matrix.png'")

print("\n" + "="*60)
print("📊 CONFUSION MATRIX")
print("="*60)
plot_confusion_matrix(all_labels, all_predictions, classes)

# =============================================================================
# MAKING PREDICTIONS ON SINGLE IMAGES
# =============================================================================
print("\n" + "="*60)
print("🎯 HOW TO PREDICT ON A SINGLE IMAGE")
print("="*60)

def predict_single_image(model, image_tensor, device):
    """
    Predict class for a single image.
    
    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor (1, 3, 32, 32)
        device: torch.device
    
    Returns:
        predicted_class: str
        confidence: float (0-100)
        all_probabilities: dict
    """
    model.eval()
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        
        # Add batch dimension if needed
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Get prediction
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        
        confidence, predicted_idx = probabilities.max(0)
        
        all_probs = {
            classes[i]: probabilities[i].item() * 100 
            for i in range(10)
        }
        
        return classes[predicted_idx], confidence.item() * 100, all_probs

# Demo with a test image
demo_image, demo_label = test_dataset[0]
pred_class, conf, all_probs = predict_single_image(model, demo_image, device)

print(f"""
Example code to predict on a single image:

    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2470, 0.2435, 0.2616))
    ])
    
    from PIL import Image
    image = Image.open('your_image.jpg')
    image_tensor = transform(image)
    
    # Predict
    pred_class, confidence, all_probs = predict_single_image(
        model, image_tensor, device
    )
    
Demo result:
- True class: {classes[demo_label]}
- Predicted: {pred_class} (confidence: {conf:.1f}%)
""")

print("\n" + "="*60)
print("🎉 STEP 4 COMPLETE!")
print("="*60)
print("""
KEY TAKEAWAYS:
1. torch.load() restores your trained model
2. model.eval() disables dropout for consistent predictions
3. torch.no_grad() saves memory during inference
4. torch.softmax() converts logits to probabilities
5. Confusion matrix helps identify which classes confuse the model

Congratulations! You've built a complete image classifier! 🎊

NEXT STEPS TO EXPLORE:
- Try different architectures (ResNet, VGG)
- Add data augmentation to improve accuracy
- Try other datasets (MNIST, Fashion-MNIST, your own images!)
- Use learning rate schedulers
- Implement early stopping
""")



