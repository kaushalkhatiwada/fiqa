import streamlit as st
import numpy as np
from PIL import Image
import time
import random

# ------------------------------
# Mock model functions (replace with real model later)
# ------------------------------
def classify_image(image):
    classes = ["Gradable", "Ungradable"]
    pred_class = random.choice(classes)
    confidence = round(random.uniform(0.7, 0.99), 2)
    
    if pred_class == "Ungradable":
        reasons = ["Blurry / Out-of-focus", 
                   "Low illumination / Overexposure", 
                   "Small field of view", 
                   "Obstructions (eyelids, glare)"]
        reason = random.choice(reasons)
    else:
        reason = None
    return pred_class, confidence, reason


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Fundus Image Quality Assessment", layout="wide")

st.title("👁️ Fundus Image Quality Assessment Dashboard")
st.write("Upload retinal fundus images to check if they are gradable for AI-assisted screening.")

# Upload Section
uploaded_files = st.file_uploader(
    "Upload fundus images (jpg/png)", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded_files:
    results = []
    for file in uploaded_files:
        # Load image
        image = Image.open(file).convert("RGB")
        
        # Display image
        st.image(image, caption=f"Uploaded: {file.name}", width=300)
        
        # Run fake model
        with st.spinner("Analyzing image quality..."):
            time.sleep(1)  # simulate processing
            pred_class, confidence, reason = classify_image(image)
        
        # Display results
        if pred_class == "Gradable":
            st.success(f"✅ {file.name} is **Gradable** (Confidence: {confidence})")
        else:
            st.error(f"❌ {file.name} is **Ungradable** (Confidence: {confidence})")
            st.warning(f"Reason: {reason}")
        
        results.append((file.name, pred_class, confidence, reason))
    
    # Bulk summary table
    st.subheader("📊 Summary of Uploaded Images")
    st.table(results)
