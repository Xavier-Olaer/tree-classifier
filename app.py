import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image

# =========================
# TREE INFO DATABASE
# =========================
tree_info = {
    # FRUIT BEARING
    "avocado": {
        "type": "Fruit Bearing",
        "description": "A tropical tree that produces creamy, nutrient-rich avocados.",
        "fun_fact": "Avocados are technically berries!"
    },
    "banana": {
        "type": "Fruit Bearing",
        "description": "A fast-growing plant that produces bananas.",
        "fun_fact": "Banana plants are actually giant herbs, not trees!"
    },
    "coconut": {
        "type": "Fruit Bearing",
        "description": "A tropical tree known for producing coconuts.",
        "fun_fact": "Coconuts can travel across oceans and still grow!"
    },
    "guava": {
        "type": "Fruit Bearing",
        "description": "A small tree that produces sweet and aromatic guava fruits.",
        "fun_fact": "Guava has more Vitamin C than oranges."
    },
    "guyabano": {
        "type": "Fruit Bearing",
        "description": "Also known as soursop, it produces large, spiky fruits.",
        "fun_fact": "Guyabano is often used in juices and desserts."
    },
    "jackfruit": {
        "type": "Fruit Bearing",
        "description": "A large tropical tree that produces the biggest tree fruit.",
        "fun_fact": "Jackfruit can weigh up to 50 kg!"
    },
    "kaimito": {
        "type": "Fruit Bearing",
        "description": "Also called star apple, known for its sweet, milky pulp.",
        "fun_fact": "Kaimito has a star shape when cut in half."
    },
    "lanzones": {
        "type": "Fruit Bearing",
        "description": "A tropical tree that produces small, sweet fruits in clusters.",
        "fun_fact": "Lanzones is a popular fruit in Southeast Asia."
    },
    "mango": {
        "type": "Fruit Bearing",
        "description": "A tropical tree famous for its sweet mango fruits.",
        "fun_fact": "Mango is known as the 'King of Fruits'."
    },
    "santol": {
        "type": "Fruit Bearing",
        "description": "A tropical fruit tree with thick rind and sweet pulp.",
        "fun_fact": "Santol is sometimes called cotton fruit."
    },
    "starfruit": {
        "type": "Fruit Bearing",
        "description": "A tree that produces star-shaped fruits.",
        "fun_fact": "Starfruit slices look like stars."
    },
    "chicos": {
        "type": "Fruit Bearing",
        "description": "Also known as sapodilla, produces sweet brown fruits.",
        "fun_fact": "Chico fruit tastes like caramel."
    },

    # NON-FRUIT BEARING
    "acacia": {
        "type": "Non-Fruit Bearing",
        "description": "A large tree commonly used for shade and timber.",
        "fun_fact": "Acacia trees can survive in very dry climates."
    },
    "balete": {
        "type": "Non-Fruit Bearing",
        "description": "A large fig tree often associated with folklore.",
        "fun_fact": "Balete trees are known in Filipino myths."
    },
    "banaba": {
        "type": "Non-Fruit Bearing",
        "description": "A tree known for its medicinal leaves.",
        "fun_fact": "Banaba leaves are used for diabetes treatment."
    },
    "bangkal": {
        "type": "Non-Fruit Bearing",
        "description": "A tree often found in swampy areas.",
        "fun_fact": "Bangkal wood is used in local construction."
    },
    "bani": {
        "type": "Non-Fruit Bearing",
        "description": "A coastal tree known for its durability.",
        "fun_fact": "Bani trees can survive salty environments."
    },
    "eucalyptus": {
        "type": "Non-Fruit Bearing",
        "description": "A fast-growing tree known for its aromatic leaves.",
        "fun_fact": "Koalas feed mainly on eucalyptus leaves 🐨."
    },
    "gmelina": {
        "type": "Non-Fruit Bearing",
        "description": "A fast-growing tree used for lumber.",
        "fun_fact": "Gmelina is widely used in furniture making."
    },
    "mahogany": {
        "type": "Non-Fruit Bearing",
        "description": "A hardwood tree valued for its timber.",
        "fun_fact": "Mahogany wood is used in high-end furniture."
    },
    "molave": {
        "type": "Non-Fruit Bearing",
        "description": "A strong and durable native tree.",
        "fun_fact": "Molave wood resists termites."
    },
    "narra": {
        "type": "Non-Fruit Bearing",
        "description": "The national tree of the Philippines.",
        "fun_fact": "Narra wood is very strong and valuable."
    },
    "talisay": {
        "type": "Non-Fruit Bearing",
        "description": "A tropical tree often used for shade.",
        "fun_fact": "Talisay leaves turn red before falling 🍂."
    }
}

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
model = models.efficientnet_b0()
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load("tree_classifier_efficientnet.pth", map_location="cpu"))
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
    <h1 style='text-align: center; color: #2e7d32;'>🌳 Tree Classifier</h1>
    <p style='text-align: center; font-size:16px;'>
        Upload a tree image to check if it is fruit-bearing
    </p>
""", unsafe_allow_html=True)

st.markdown("---")

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title("ℹ️ About")
st.sidebar.write(
    "This AI model classifies whether a tree is fruit-bearing or not using deep learning."
)

st.sidebar.markdown("---")
st.sidebar.write("**Tip:** Use clear tree images for better results.")

# -------------------------
# FILE UPLOAD
# -------------------------
uploaded_file = st.file_uploader(
    "📤 Upload a tree image",
    type=["jpg", "png", "jpeg"]
)

# -------------------------
# CLASSIFICATION
# -------------------------
if uploaded_file is not None:

    # TREE TYPE DETECTION (FROM FILENAME)
    filename = uploaded_file.name.lower()
    tree_type = filename.split("_")[0]

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", width="stretch")

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        with st.spinner("🔍 Analyzing image..."):
            output = model(img)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence, pred = torch.max(probabilities, 0)

    confidence_value = confidence.item()

    # -------------------------
    # THRESHOLD
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
        emoji = "🌿"
    elif label == "non_fruit_bearing":
        color = "#F44336"
        emoji = "🌳"
    else:
        color = "#9E9E9E"
        emoji = "❓"

    # -------------------------
    # DISPLAY LABEL
    # -------------------------
    display_label = (
        "Not Sure" if label == "not_sure"
        else label.replace("_", " ").title()
    )

    # =========================
    # RESULT BOX DISPLAY
    # =========================
    st.markdown(f"""
        <div style="
            padding: 25px;
            border-radius: 15px;
            background-color: {color};
            color: white;
            text-align: center;
            margin-top: 20px;
        ">
            <h2>{emoji} Classification: {display_label}</h2>
            <h4>Confidence: {confidence_value * 100:.2f}%</h4>
        </div>
    """, unsafe_allow_html=True)

    if label == "not_sure":
        st.warning("⚠️ The model is not confident. Try a clearer image.")

    # =========================
    # 🧠 SMART CONSISTENCY CHECK
    # =========================
    st.markdown("---")
    st.subheader("🌳 Tree Analysis")

    info = tree_info.get(tree_type)

    if info:
        st.write(f"**🌳 Detected Tree (Dataset):** {tree_type.title()}")
        st.write(f"**🌿 Expected Type:** {info['type']}")

        st.write(f"📖 {info['description']}")
        st.info(f"💡 Fun Fact: {info['fun_fact']}")

    else:
        st.warning("⚠️ Tree type not recognized from filename.")

    # -------------------------
    # PROBABILITY UI
    # -------------------------
    st.markdown("---")
    st.subheader("📊 Classification Breakdown")

    fruit_prob = probabilities[0].item()
    nonfruit_prob = probabilities[1].item()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🌿 Fruit Bearing**")
        st.progress(fruit_prob)
        st.caption(f"{fruit_prob*100:.2f}%")

    with col2:
        st.markdown("**🌳 Non Fruit Bearing**")
        st.progress(nonfruit_prob)
        st.caption(f"{nonfruit_prob*100:.2f}%")

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")

st.markdown("""
    <div style='text-align: center; color: gray; font-size: 14px; margin-top: 20px;'>
        © 2026 Xavier B. Olaer | BSCpE 3A
    </div>
""", unsafe_allow_html=True)