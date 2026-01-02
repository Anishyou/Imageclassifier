# 🖼️ PyTorch Image Classifier

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive, step-by-step deep learning project for image classification using PyTorch. Learn to build, train, and deploy CNN models from scratch, and apply transfer learning for state-of-the-art results.

![Sample Predictions](sample_images.png)

---

## ✨ Features

- 📚 **8 Progressive Learning Steps** - From data loading to deploying on your own images
- 🧠 **Custom CNN Architecture** - Build a convolutional neural network from scratch
- 🔄 **Transfer Learning** - Use pre-trained ResNet18 for ~90% accuracy
- 📈 **Data Augmentation** - Boost performance with image transformations
- ⚡ **GPU Support** - Automatic CUDA/MPS detection for fast training
- 🖼️ **Classify Your Own Images** - Use the trained model on any image

---

## 📋 Learning Path

| Step | File | What You'll Learn | Difficulty |
|:----:|------|-------------------|:----------:|
| 1 | `step1_data_loading.py` | Datasets, transforms, DataLoaders | ⭐ |
| 2 | `step2_build_model.py` | CNN architecture (Conv, Pool, FC layers) | ⭐⭐ |
| 3 | `step3_train_model.py` | Training loop, loss functions, optimizers | ⭐⭐ |
| 4 | `step4_evaluate_and_predict.py` | Evaluation, predictions, confusion matrix | ⭐⭐ |
| 5 | `step5_data_augmentation.py` | Image augmentation techniques | ⭐⭐ |
| 6 | `step6_transfer_learning.py` | Pre-trained models, fine-tuning | ⭐⭐⭐ |
| 7 | `step7_learning_rate_scheduler.py` | Learning rate scheduling strategies | ⭐⭐⭐ |
| 8 | `step8_your_own_images.py` | Classify your own images! | ⭐ |

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Anishyou/Imageclassifier.git
cd Imageclassifier
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Steps

```bash
# Learn the fundamentals
python step1_data_loading.py      # Understand data loading
python step2_build_model.py       # Explore CNN architecture

# Train and evaluate
python step3_train_model.py       # Train the model (~10 min CPU, ~2 min GPU)
python step4_evaluate_and_predict.py  # See results

# Advanced techniques
python step5_data_augmentation.py     # Data augmentation
python step6_transfer_learning.py     # Transfer learning with ResNet
python step7_learning_rate_scheduler.py  # LR scheduling

# Use on your own images
python step8_your_own_images.py   # Classify any image!
```

---

## 📊 Dataset: CIFAR-10

| Property | Value |
|----------|-------|
| Total Images | 60,000 (50k train, 10k test) |
| Image Size | 32×32 RGB |
| Classes | 10 |

**Classes:** ✈️ airplane, 🚗 automobile, 🐦 bird, 🐱 cat, 🦌 deer, 🐕 dog, 🐸 frog, 🐴 horse, 🚢 ship, 🚚 truck

---

## 🏗️ Model Architecture

### Custom CNN (from scratch)

```
Input (3×32×32)
    ↓
Conv1 (32 filters) → BatchNorm → ReLU → MaxPool → (32×16×16)
    ↓
Conv2 (64 filters) → BatchNorm → ReLU → MaxPool → (64×8×8)
    ↓
Conv3 (128 filters) → BatchNorm → ReLU → MaxPool → (128×4×4)
    ↓
Flatten (2048)
    ↓
FC1 (256) → ReLU → Dropout(0.5)
    ↓
FC2 (10) → Output (class scores)
```

**Parameters:** ~596K trainable parameters

---

## 📈 Results

| Model | Accuracy | Training Time |
|-------|:--------:|:-------------:|
| Custom CNN (10 epochs) | ~70-75% | ~10 min (GPU) |
| With Data Augmentation | ~75-80% | ~15 min (GPU) |
| Transfer Learning (ResNet18) | ~85-92% | ~20 min (GPU) |

---

## 💾 Trained Models

Pre-trained model weights are included:

| File | Description | How to Use |
|------|-------------|------------|
| `best_model.pth` | Custom CNN trained on CIFAR-10 | Load with `step2_build_model.ImageClassifier` |
| `feature_extractor_best.pth` | ResNet18 transfer learning | Load with `torchvision.models.resnet18` |

### Loading a Trained Model

```python
import torch
from step2_build_model import ImageClassifier

# Load custom CNN
model = ImageClassifier(num_classes=10)
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Check accuracy achieved
print(f"Best accuracy: {checkpoint['best_acc']:.2f}%")
```

---

## 🖼️ Classify Your Own Images

```python
python step8_your_own_images.py --image path/to/your/image.jpg
```

Or use in code:

```python
from PIL import Image
import torch
import torchvision.transforms as transforms
from step2_build_model import ImageClassifier

# Classes
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Load model
model = ImageClassifier(num_classes=10)
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

image = Image.open('your_image.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    _, predicted = output.max(1)
    
print(f"Prediction: {classes[predicted.item()]}")
```

---

## 📁 Project Structure

```
Imageclassifier/
├── 📂 data/                    # CIFAR-10 dataset (auto-downloaded)
├── 📄 step1_data_loading.py    # Data loading tutorial
├── 📄 step2_build_model.py     # CNN architecture
├── 📄 step3_train_model.py     # Training loop
├── 📄 step4_evaluate_and_predict.py  # Evaluation
├── 📄 step5_data_augmentation.py     # Augmentation
├── 📄 step6_transfer_learning.py     # Transfer learning
├── 📄 step7_learning_rate_scheduler.py  # LR scheduling
├── 📄 step8_your_own_images.py       # Use your own images
├── 🔧 best_model.pth           # Trained CNN weights
├── 🔧 feature_extractor_best.pth  # Transfer learning weights
├── 📄 requirements.txt         # Dependencies
└── 📄 README.md               # This file
```

---

## 💡 Key Concepts

<details>
<summary><b>🔹 Data Transforms</b></summary>

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
```
</details>

<details>
<summary><b>🔹 Training Loop</b></summary>

```python
for epoch in range(epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()           # Reset gradients
        outputs = model(images)         # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()                 # Backward pass
        optimizer.step()                # Update weights
```
</details>

<details>
<summary><b>🔹 Evaluation Mode</b></summary>

```python
model.eval()
with torch.no_grad():
    outputs = model(images)
    _, predicted = outputs.max(1)
```
</details>

<details>
<summary><b>🔹 Transfer Learning</b></summary>

```python
from torchvision import models

# Load pre-trained ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Replace final layer for 10 classes
model.fc = nn.Linear(512, 10)
```
</details>

---

## 🛠️ Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- matplotlib
- numpy
- tqdm
- Pillow

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) by Alex Krizhevsky
- [PyTorch](https://pytorch.org/) for the amazing deep learning framework
- [torchvision](https://pytorch.org/vision/) for pre-trained models

---

<p align="center">
  Made with ❤️ for learning deep learning
</p>
