import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# =========================
# SETTINGS
# =========================
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = "fruit_tree_dataset"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# TRANSFORMS
# =========================
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

val_test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])


# DATASETS

train_data = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_transform)
val_data   = datasets.ImageFolder(f"{DATA_DIR}/validation", transform=val_test_transform)
test_data  = datasets.ImageFolder(f"{DATA_DIR}/test", transform=val_test_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


# MODEL (ResNet18)

model = models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False


model.fc = nn.Linear(model.fc.in_features, 2)

model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# TRAINING 

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    correct = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = correct / len(train_data)

    # VALIDATION
    model.eval()
    val_correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    val_acc = val_correct / len(val_data)

    print(f"Epoch {epoch+1}")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print("-" * 30)


# SAVE

torch.save(model.state_dict(), "tree_classifier.pth")

print("✅ Training complete and model saved!")