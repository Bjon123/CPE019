# App.py
import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# ----------------------------
# STREAMLIT CONFIG
# ----------------------------
st.set_page_config(
    page_title="Car Brand Classifier",
    page_icon="🚗",
    layout="centered"
)

# ----------------------------
# GOOGLE DRIVE MODEL DOWNLOAD (gdown)
# ----------------------------
MODEL_FILE = "Model.h5"
MODEL_DRIVE_URL = "https://drive.google.com/uc?id=1ekNm6RW1sffhQQ94xhHqpsRN5tgIVpAx"  # replace with your file ID

if not os.path.exists(MODEL_FILE):
    st.info("Downloading model from Google Drive...")
    import gdown
    gdown.download(MODEL_DRIVE_URL, MODEL_FILE, quiet=False)
    st.success("Model downloaded successfully!")

# ----------------------------
# LOAD CLASS NAMES
# ----------------------------
with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    num_classes = len(class_names)
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ----------------------------
# IMAGE TRANSFORMS
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225)
    )
])

# ----------------------------
# UI HEADER
# ----------------------------
st.markdown(
    """
    <h1 style="text-align:center; color:#4CAF50;">
        Car Brand Classifier
    </h1>
    <p style="text-align:center; font-size:18px;">
        Upload an image of a car and the AI will predict its brand.
    </p>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# UPLOAD IMAGE
# ----------------------------
file = st.file_uploader("Upload Car Image (JPG or PNG)", type=["jpg","jpeg","png"])

# ----------------------------
# PREDICTION FUNCTION
# ----------------------------
def predict(image):
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.softmax(outputs, dim=1).numpy()[0]
    return probabilities

# ----------------------------
# MAIN LOGIC
# ----------------------------
if file:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    probabilities = predict(image)
    pred_idx = np.argmax(probabilities)
    pred_label = class_names[pred_idx]
    pred_conf = probabilities[pred_idx]*100

    # Result Card
    st.markdown(
        f"""
        <div style="padding:20px; border-radius:10px; background-color:#f0f8f5; 
                    text-align:center; border:2px solid #4CAF50;">
            <h2 style="color:#2E7D32;">Prediction: {pred_label}</h2>
            <h3 style="color:#388E3C;">Confidence: {pred_conf:.2f}%</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Top 3 predictions
    top3_idx = probabilities.argsort()[-3:][::-1]
    st.markdown("**Top 3 Predictions:**")
    for i in top3_idx:
        st.write(f"{class_names[i]}: {probabilities[i]*100:.2f}%")

    # Confidence Chart
    st.subheader("Confidence Score Chart")
    fig, ax = plt.subplots(figsize=(10,4))
    ax.bar(class_names, probabilities, color="#4CAF50")
    ax.set_ylabel("Confidence")
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    st.pyplot(fig)

else:
    st.info("Please upload an image above to begin classification.")
