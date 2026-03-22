import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Tree Classifier",
    page_icon="🌳",
    layout="centered"
)

# -------------------------
# LOAD MODEL
# -------------------------
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("tree_classifier.pth", map_location="cpu"))
model.eval()

classes = ["fruit_bearing", "non_fruit_bearing"]

# -------------------------
# TRANSFORM
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------
# UI HEADER
# -------------------------
st.markdown("""
    <h1 style='text-align: center; color: #2e7d32;'>Tree Classifier</h1>
    <p style='text-align: center;'>Upload an image to check if the tree is fruit-bearing</p>
""", unsafe_allow_html=True)

st.markdown("---")

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title("ℹ️ About")
st.sidebar.write("This AI model classifies whether a tree is fruit-bearing or not.")

# -------------------------
# FILE UPLOAD
# -------------------------
uploaded_file = st.file_uploader(
    "Upload a tree image",
    type=["jpg", "png", "jpeg"]
)

# -------------------------
# PREDICTION
# -------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width="stretch")

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        with st.spinner("Analyzing image..."):
            output = model(img)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence, pred = torch.max(probabilities, 0)

    confidence_value = confidence.item()

    # -------------------------
    # NOT SURE THRESHOLD
    # -------------------------
    THRESHOLD = 0.70

    if confidence_value < THRESHOLD:
        label = "not_sure"
    else:
        label = classes[pred]

    # -------------------------
    # COLOR LOGIC
    # -------------------------
    if label == "fruit_bearing":
        color = "#4CAF50"
    elif label == "non_fruit_bearing":
        color = "#F44336"
    else:
        color = "#9E9E9E"  # gray for not sure

    # -------------------------
    # DISPLAY TEXT
    # -------------------------
    display_label = (
        "Not Sure" if label == "not_sure"
        else label.replace("_", " ").title()
    )

    # -------------------------
    # RESULT BOX
    # -------------------------
    st.markdown(f"""
        <div style="
            padding: 20px;
            border-radius: 12px;
            background-color: {color};
            color: white;
            text-align: center;
            margin-top: 20px;
        ">
            <h2>{display_label}</h2>
            <h4>Confidence: {confidence_value * 100:.2f}%</h4>
        </div>
    """, unsafe_allow_html=True)

    # Optional message
    if label == "not_sure":
        st.info("The model is not confident. Try another image.")

    # -------------------------
    # PROBABILITY UI
    # -------------------------
    st.markdown("---")
    st.subheader("Prediction Confidence")

    fruit_prob = probabilities[0].item()
    nonfruit_prob = probabilities[1].item()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Fruit Bearing**")
        st.progress(fruit_prob)
        st.write(f"{fruit_prob*100:.2f}%")

    with col2:
        st.markdown("**Non Fruit Bearing**")
        st.progress(nonfruit_prob)
        st.write(f"{nonfruit_prob*100:.2f}%")

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")

st.markdown("""
    <div style='text-align: center; color: gray; font-size: 14px; margin-top: 20px;'>
        © 2026 Xavier B. Olaer | BSCpE 3A
    </div>
""", unsafe_allow_html=True)