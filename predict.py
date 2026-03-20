import torch
from torchvision import transforms, models
from PIL import Image

# Load model
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("tree_classifier.pth"))
model.eval()

# Class names
classes = ["fruit_bearing", "non_fruit_bearing"]

# Transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# Load image
img_path = input("Enter image path: ")
image = Image.open(img_path).convert("RGB")
image = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(image)
    pred = output.argmax(1).item()

print(f"Prediction: {classes[pred]}")