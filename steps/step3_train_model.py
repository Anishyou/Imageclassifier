"""
=============================================================================
STEP 3: TRAINING THE MODEL
=============================================================================
In this step, you'll learn:
- The training loop (the heart of deep learning!)
- Loss functions (how we measure errors)
- Optimizers (how we update weights)
- Forward and backward passes
=============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import our model from step 2
from step2_build_model import ImageClassifier

# =============================================================================
# CONFIGURATION
# =============================================================================
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 10  # Number of complete passes through the training data

# =============================================================================
# DEVICE SETUP (CPU vs GPU)
# =============================================================================
"""
PyTorch can use:
- CPU: Works on any computer (slower)
- CUDA: NVIDIA GPUs (much faster!)
- MPS: Apple Silicon (M1/M2 chips)

We check what's available and use the best option.
"""
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🍎 Using Apple Silicon GPU")
else:
    device = torch.device("cpu")
    print("💻 Using CPU (training will be slower)")

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n📥 Loading CIFAR-10 dataset...")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

train_dataset = torchvision.datasets.CIFAR10(
    root='../data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root='../data', train=False, download=True, transform=transform
)

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
# CREATE MODEL, LOSS FUNCTION, AND OPTIMIZER
# =============================================================================
print("\n🏗️ Creating model...")

# Initialize model and move to device (GPU/CPU)
model = ImageClassifier(num_classes=10)
model = model.to(device)

# -----------------------------------------------------------------------------
# LOSS FUNCTION: CrossEntropyLoss
# -----------------------------------------------------------------------------
"""
CrossEntropyLoss is used for classification tasks.
It measures the difference between:
- What the model predicted (probabilities)
- What the actual class is (ground truth)

Lower loss = better predictions!
"""
criterion = nn.CrossEntropyLoss()

# -----------------------------------------------------------------------------
# OPTIMIZER: Adam
# -----------------------------------------------------------------------------
"""
The optimizer updates the model's weights to reduce the loss.
Adam is a popular choice because:
- It adapts the learning rate for each parameter
- It works well for most problems

Learning rate controls how big the weight updates are:
- Too high: model might overshoot and not converge
- Too low: training takes forever
- 0.001 is a good starting point
"""
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"✓ Loss function: CrossEntropyLoss")
print(f"✓ Optimizer: Adam (lr={LEARNING_RATE})")

# =============================================================================
# THE TRAINING LOOP
# =============================================================================
"""
The training loop is the heart of deep learning!

For each epoch:
    For each batch:
        1. Forward pass: Get predictions
        2. Calculate loss: How wrong are we?
        3. Backward pass: Calculate gradients (how to improve)
        4. Update weights: Apply the gradients
"""

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train the model for one epoch."""
    model.train()  # Set model to training mode (enables dropout, etc.)
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    
    for batch_idx, (images, labels) in enumerate(pbar):
        # Move data to device
        images = images.to(device)
        labels = labels.to(device)
        
        # =====================================================================
        # STEP 1: Zero the gradients
        # =====================================================================
        # Gradients accumulate by default, so we need to reset them
        optimizer.zero_grad()
        
        # =====================================================================
        # STEP 2: Forward pass
        # =====================================================================
        # Get model predictions
        outputs = model(images)  # Shape: (batch_size, 10)
        
        # =====================================================================
        # STEP 3: Calculate loss
        # =====================================================================
        loss = criterion(outputs, labels)
        
        # =====================================================================
        # STEP 4: Backward pass (backpropagation)
        # =====================================================================
        # Calculate gradients for all parameters
        loss.backward()
        
        # =====================================================================
        # STEP 5: Update weights
        # =====================================================================
        # Apply the gradients to update model parameters
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)  # Get class with highest score
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/(batch_idx+1):.3f}',
            'acc': f'{100.*correct/total:.1f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device):
    """Evaluate the model on test data."""
    model.eval()  # Set model to evaluation mode (disables dropout)
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # No gradient computation during evaluation (saves memory)
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc


# =============================================================================
# TRAINING!
# =============================================================================
print("\n" + "="*60)
print("🏋️ STARTING TRAINING")
print("="*60 + "\n")

# Track metrics for plotting
train_losses = []
train_accs = []
test_losses = []
test_accs = []

best_acc = 0.0

for epoch in range(NUM_EPOCHS):
    # Train
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device, epoch
    )
    
    # Evaluate
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    # Store metrics
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    print(f"\n📊 Epoch {epoch+1} Results:")
    print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"   Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
    
    # Save best model
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
        }, '../models/best_model.pth')
        print(f"   💾 New best model saved to models/ (Accuracy: {best_acc:.2f}%)")
    
    print()

# =============================================================================
# PLOT TRAINING CURVES
# =============================================================================
print("📈 Saving training curves...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Loss plot
ax1.plot(train_losses, label='Train Loss', marker='o')
ax1.plot(test_losses, label='Test Loss', marker='s')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training & Test Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accuracy plot
ax2.plot(train_accs, label='Train Accuracy', marker='o')
ax2.plot(test_accs, label='Test Accuracy', marker='s')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training & Test Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/training_curves.png', dpi=150)
plt.show()

print("\n" + "="*60)
print("🎉 TRAINING COMPLETE!")
print("="*60)
print(f"""
RESULTS:
- Best Test Accuracy: {best_acc:.2f}%
- Model saved to: 'models/best_model.pth'
- Training curves saved to: 'outputs/training_curves.png'

KEY TAKEAWAYS:
1. Training loop: forward → loss → backward → update
2. model.train() for training, model.eval() for evaluation
3. torch.no_grad() saves memory during evaluation
4. Save checkpoints to keep your best model!

Next: Run 'step4_evaluate_and_predict.py' to see the model in action!
""")



