import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image

# Load model
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("tree_classifier.pth", map_location="cpu"))
model.eval()

classes = ["fruit_bearing", "non_fruit_bearing"]

# Transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# UI
st.title("🌳 Tree Classifier 🌳")
st.write("Upload an image to check if the tree is fruit-bearing")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, pred = torch.max(probabilities, 0)

    st.subheader(f"Prediction: {classes[pred]}")
    st.subheader(f"Confidence: {confidence.item()*100:.2f}%")