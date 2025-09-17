# app.py
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from utils import load_model, preprocess_image, predict, generate_saliency

# Load model
model, device = load_model()

# Streamlit UI
st.set_page_config(page_title="Fundus Image Quality Assessment", layout="wide")
st.title("👁️ Fundus Image Quality Assessment Dashboard")
st.write("Upload retinal fundus images to check if they are gradable for AI-assisted screening.")

tab1, tab2 = st.tabs(["📄 Single Image", "📂 Multiple Images"])

# ------------------------------
# Single Image Upload
# ------------------------------
with tab1:
    uploaded_file = st.file_uploader(
        "Upload a fundus image (jpg/png)", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=False,
        key="single_upload"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        saliency_map = generate_saliency(model, image, device=device, cmap_name="hsv", overlay=True)
        # --- Display side by side ---
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption={uploaded_file.name}, width=400)

        with col2:
            st.image(saliency_map, caption=f"Saliency Map", width=400)

        input_tensor = preprocess_image(image)
        pred_class, confidence, gradable_status, reason = predict(model, device, input_tensor)

        # Display results
        if gradable_status == "Gradable":
            st.success(f"✅ **{pred_class} (Gradable)** (Confidence: {confidence*100:.1f}%)")
        else:
            st.error(f"❌ **{pred_class} (Ungradable)** (Confidence: {confidence*100:.1f}%)")
            st.warning(f"Reason: {reason}")


# ------------------------------
# Multiple Image Upload
# ------------------------------
with tab2:
    uploaded_files = st.file_uploader(
        "Upload multiple fundus images (jpg/png)", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True,
        key="multi_upload"
    )

    if uploaded_files:
        results = []
        for file in uploaded_files:
            image = Image.open(file).convert("RGB")
            # saliency_map = generate_saliency(model, image)  # Saliency map
            saliency_map = generate_saliency(model, image, device=device, cmap_name="hsv", overlay=True)
            # --- Display side by side ---
            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption={file.name}, width=400)

            with col2:
                st.image(saliency_map, caption="Saliency Map", width=400)

            input_tensor = preprocess_image(image)
            pred_class, confidence, gradable_status, reason = predict(model, device, input_tensor)

            if gradable_status == "Gradable":
                st.success(f"✅ {file.name}: **{pred_class} (Gradable)** ({confidence*100:.1f}%)")
            else:
                st.error(f"❌ {file.name}: **{pred_class} (Ungradable)** ({confidence*100:.1f}%)")
                st.warning(f"Reason: {reason}")

            results.append([file.name, pred_class, gradable_status, f"{confidence*100:.1f}%", reason])

        st.subheader("📊 Summary of Uploaded Images")
        df_results = pd.DataFrame(results, columns=["Filename", "Predicted", "Gradable", "Confidence", "Reason"])
        st.table(df_results)
