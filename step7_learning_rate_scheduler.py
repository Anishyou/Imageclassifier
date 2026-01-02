"""
=============================================================================
STEP 7: LEARNING RATE SCHEDULING & EARLY STOPPING
=============================================================================
In this step, you'll learn:
- Why learning rate matters
- Different learning rate schedulers
- Early stopping to prevent overfitting
- Putting it all together
=============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import our model
from step2_build_model import ImageClassifier

# =============================================================================
# WHY LEARNING RATE MATTERS
# =============================================================================
"""
Learning rate (LR) controls how big steps the optimizer takes.

THE PROBLEM:
- High LR at start: Learn fast, but might overshoot optimal weights
- Low LR at start: Very slow training, might get stuck
- Same LR throughout: Not optimal!

THE SOLUTION: Learning Rate Scheduling
- Start with higher LR: Make quick progress
- Gradually decrease LR: Fine-tune and converge

ANALOGY:
🎯 Like approaching a target:
   - Start with big jumps to get close quickly
   - Then take small steps for precise landing
   
WITHOUT scheduling:    O→→→→→→→→→→→→→→→→→ (might overshoot)
WITH scheduling:       O→→→→→→→→→→→→→ (slow down near target)
"""

# =============================================================================
# DEVICE SETUP
# =============================================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# =============================================================================
# DATA LOADING
# =============================================================================
print("\n📥 Loading data...")

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=test_transform
)

BATCH_SIZE = 64
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

print(f"✓ Data loaded")

# =============================================================================
# LEARNING RATE SCHEDULERS
# =============================================================================
print("\n" + "="*60)
print("📊 LEARNING RATE SCHEDULERS")
print("="*60)

print("""
PyTorch offers many schedulers. Here are the most common:

1. StepLR
   - Reduce LR by factor every N epochs
   - Example: LR * 0.1 every 30 epochs
   
2. ExponentialLR
   - Reduce LR by factor every epoch
   - Smooth, continuous decay
   
3. CosineAnnealingLR
   - LR follows cosine curve
   - Popular in modern training
   
4. ReduceLROnPlateau
   - Reduce LR when metric stops improving
   - Smart, adaptive approach
   
5. OneCycleLR
   - Increase then decrease LR (one big cycle)
   - Fast convergence, often best results
""")

# =============================================================================
# VISUALIZE SCHEDULERS
# =============================================================================
print("\n📈 Visualizing different schedulers...")

def visualize_schedulers():
    """Show how different schedulers change learning rate over time."""
    epochs = 50
    initial_lr = 0.1
    
    # Create dummy model and optimizer for each scheduler
    schedulers = {}
    lr_histories = {}
    
    scheduler_configs = [
        ("StepLR", lambda opt: lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)),
        ("ExponentialLR", lambda opt: lr_scheduler.ExponentialLR(opt, gamma=0.95)),
        ("CosineAnnealingLR", lambda opt: lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)),
        ("OneCycleLR", lambda opt: lr_scheduler.OneCycleLR(
            opt, max_lr=initial_lr, epochs=epochs, steps_per_epoch=1
        )),
    ]
    
    for name, scheduler_fn in scheduler_configs:
        model = nn.Linear(10, 10)
        optimizer = optim.SGD(model.parameters(), lr=initial_lr)
        scheduler = scheduler_fn(optimizer)
        
        lrs = []
        for epoch in range(epochs):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        
        lr_histories[name] = lrs
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, lrs in lr_histories.items():
        ax.plot(lrs, label=name, linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedulers Comparison', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lr_schedulers.png', dpi=150)
    plt.show()
    print("✓ Scheduler comparison saved to 'lr_schedulers.png'")

visualize_schedulers()

# =============================================================================
# EARLY STOPPING
# =============================================================================
print("\n" + "="*60)
print("🛑 EARLY STOPPING")
print("="*60)

print("""
Early Stopping = Stop training when the model stops improving.

WHY?
- Prevents overfitting (training too long on training data)
- Saves time (no point training if not improving)
- Keeps the best model

HOW IT WORKS:
1. Track validation accuracy/loss after each epoch
2. If no improvement for N epochs (patience), stop
3. Restore the best model weights
""")


class EarlyStopping:
    """
    Early stopping to stop training when validation doesn't improve.
    
    Args:
        patience: How many epochs to wait before stopping
        min_delta: Minimum improvement to count as "better"
        restore_best: Whether to restore best weights when stopping
    """
    
    def __init__(self, patience=5, min_delta=0, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        
        self.best_score = None
        self.best_weights = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, score, model):
        """
        Check if we should stop.
        
        Args:
            score: Current validation metric (higher is better)
            model: The model being trained
        
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            # First epoch
            self.best_score = score
            self.best_weights = model.state_dict().copy()
            return False
        
        if score > self.best_score + self.min_delta:
            # Improved!
            self.best_score = score
            self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0
            return False
        else:
            # No improvement
            self.counter += 1
            print(f"    ⚠️ No improvement for {self.counter}/{self.patience} epochs")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best:
                    print(f"    ⏹️ Early stopping! Restoring best weights (score: {self.best_score:.2f})")
                    model.load_state_dict(self.best_weights)
                return True
        
        return False


# =============================================================================
# TRAINING WITH SCHEDULER + EARLY STOPPING
# =============================================================================
print("\n" + "="*60)
print("🏋️ COMPLETE TRAINING EXAMPLE")
print("="*60)

def train_with_scheduler(model, train_loader, test_loader, 
                         num_epochs=20, patience=5):
    """
    Train with learning rate scheduler and early stopping.
    """
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    
    # Scheduler: ReduceLROnPlateau (reduces when accuracy stops improving)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',        # We want to maximize accuracy
        factor=0.5,        # Reduce LR by half
        patience=2,        # Wait 2 epochs before reducing
        verbose=True       # Print when LR changes
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, min_delta=0.1)
    
    # History
    train_losses = []
    test_accs = []
    lr_history = []
    
    print(f"\n📊 Training for up to {num_epochs} epochs (patience={patience})")
    print("="*50)
    
    for epoch in range(num_epochs):
        # Track learning rate
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        
        # Training
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
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
        test_accs.append(test_acc)
        
        print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={test_acc:.2f}%, LR={current_lr:.6f}")
        
        # Update scheduler based on accuracy
        scheduler.step(test_acc)
        
        # Check early stopping
        if early_stopping(test_acc, model):
            print(f"\n🛑 Training stopped early at epoch {epoch+1}")
            break
    
    return train_losses, test_accs, lr_history, early_stopping.best_score

# Train the model
print("\nCreating model...")
model = ImageClassifier(num_classes=10)

print("Starting training with scheduler and early stopping...\n")
train_losses, test_accs, lr_history, best_acc = train_with_scheduler(
    model, train_loader, test_loader, num_epochs=15, patience=5
)

# =============================================================================
# PLOT RESULTS
# =============================================================================
print("\n📈 Plotting results...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Loss
axes[0].plot(train_losses, marker='o')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(test_accs, marker='o', color='green')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Test Accuracy')
axes[1].grid(True, alpha=0.3)

# Learning Rate
axes[2].plot(lr_history, marker='o', color='red')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Learning Rate')
axes[2].set_title('Learning Rate')
axes[2].set_yscale('log')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_with_scheduler.png', dpi=150)
plt.show()

print(f"\n✓ Results saved to 'training_with_scheduler.png'")
print(f"✓ Best accuracy achieved: {best_acc:.2f}%")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*60)
print("🎉 STEP 7 COMPLETE!")
print("="*60)
print("""
KEY TAKEAWAYS:

1. LEARNING RATE SCHEDULERS
   - StepLR: Simple, reduce every N epochs
   - CosineAnnealingLR: Smooth cosine decay (popular)
   - ReduceLROnPlateau: Adaptive, reduces when stuck
   - OneCycleLR: Increase then decrease (fast convergence)

2. EARLY STOPPING
   - Monitors validation metric
   - Stops when no improvement for N epochs
   - Prevents overfitting and saves time
   - Always restore best weights!

3. BEST PRACTICES
   - Start with higher LR, decrease over time
   - Use momentum for smoother updates
   - Combine scheduler + early stopping
   - Monitor both training loss and validation accuracy

RECOMMENDED SETUP:
```python
optimizer = optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
early_stopping = EarlyStopping(patience=10)
```

🎊 Congratulations! You've completed the advanced PyTorch tutorials!
""")



