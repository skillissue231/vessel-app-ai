
import streamlit as st
import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from inference import predict
from train import train_model

st.set_page_config(page_title="CAM Assay Vessel Detection", layout="centered")
st.title("🧠 CAM Assay Vessel Detection App")

# Encoder selection
encoder = st.selectbox("Choose U-Net Encoder", ["resnet18", "resnet34", "resnet50"])

# Dataset selection
dataset_choice = st.selectbox("Select Dataset", ["cam", "retina"])
dataset_path = f"data/{dataset_choice}"

# Training settings
with st.expander("🔧 Training Settings"):
    epochs = st.slider("Epochs", min_value=5, max_value=100, value=40)
    batch_size = st.slider("Batch Size", min_value=1, max_value=16, value=4)
    learning_rate = st.number_input("Learning Rate", value=0.0003, format="%.5f")

if st.button("Start Training"):
    config = {
        "encoder": encoder,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": learning_rate,
        "pretrained": True
    }

    train_model(config, dataset_path=dataset_path)

    st.success(f"✅ Training completed on {dataset_choice} dataset!")

# Prediction section
st.markdown("## 📷 Predict from Image")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
model_path = st.text_input("Path to .pth model", value=f"checkpoints/{encoder}/epoch_{epochs}.pth")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Transform image to tensor
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    image_tensor = preprocess(image)

    if st.button("Predict"):
        if os.path.exists(model_path):
            st.success(f"Loaded model: {model_path}")
            prediction = predict(image_tensor, model_path, encoder)
            st.image(prediction, caption="Predicted Vessel Mask", use_container_width=True, clamp=True)
        else:
            st.error(f"Model not found at: {model_path}")

# Footer
st.markdown("---")
st.caption("CAM Vessel Segmentation Demo App - Powered by Streamlit")
