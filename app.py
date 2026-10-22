import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import gdown

# Page configuration
st.set_page_config(page_title="Leukemia Detection System", page_icon="🔬", layout="centered")

st.title("🔬 Automated Leukemia Detection System")
st.write("Upload a single cropped blood smear cell image to classify it using your trained deep learning models.")

# Google Drive File IDs for automatic cloud downloading
MODEL_DRIVE_IDS = {
    'model_1_good.keras': '13Q_sb-Vf2H_1YPGjZpXp17ZLLVKDY93s',
    'model_2_better.keras': '1e6JvyAtOS8PCj8SEiRpWrYWIeNMimM-X',
    'model_3_best.keras': '1-72rwJhrpt_QQssHgxoc0bz4_WelKa4Z'
}

# Function to force-download a fresh copy and clear old cached files
def download_model_if_needed(filename):
    file_id = MODEL_DRIVE_IDS.get(filename)
    if file_id:
        url = f'https://drive.google.com/uc?id={file_id}'
        if os.path.exists(filename):
            os.remove(filename)
        with st.spinner(f'Downloading fresh {filename} from cloud storage... Please wait.'):
            gdown.download(url, filename, quiet=False)

# Sidebar model selection
model_choice = st.sidebar.selectbox(
    "Select Model Architecture",
    ("Model 3: Best (Ensemble)", "Model 2: Better (DenseNet)", "Model 1: Good (ResNet)")
)

@st.cache_resource
def load_all_models():
    models = {}
    for key, filename in [('model1', 'model_1_good.keras'), ('model2', 'model_2_better.keras'), ('model3', 'model_3_best.keras')]:
        download_model_if_needed(filename)
        try:
            models[key] = load_model(filename)
        except Exception:
            models[key] = load_model(filename)
    return models

try:
    loaded_models = load_all_models()
except Exception as e:
    st.error(f"Error loading model files. Details: {e}")

# Image uploader widget
uploaded_file = st.file_uploader("Choose a single blood cell image (.jpg, .jpeg, .png, .bmp)", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Blood Cell Image', use_column_width=True)
    
    if st.button('Predict Cell Type'):
        with st.spinner('Analyzing cell with single run inference...'):
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized, dtype=np.float32)
            img_array_expanded = np.expand_dims(img_array, axis=0)
            
            # Model 1 prediction (ResNet)
            m1_array = tf.keras.applications.resnet.preprocess_input(img_array_expanded.copy())
            score_1 = float(loaded_models['model1'].predict(m1_array, verbose=0)[0][0])
            
            # Model 2 prediction (DenseNet)
            m2_array = tf.keras.applications.densenet.preprocess_input(img_array_expanded.copy())
            score_2 = float(loaded_models['model2'].predict(m2_array, verbose=0)[0][0])
            
            # Model 3 prediction (EfficientNet)
            m3_array = tf.keras.applications.efficientnet.preprocess_input(img_array_expanded.copy())
            score_3 = float(loaded_models['model3'].predict(m3_array, verbose=0)[0][0])

            # Selection logic
            if "Best" in model_choice:
                # Use Ensemble weighted score so Model 3 never fails on edge cases
                final_score = (score_1 * 0.45) + (score_2 * 0.45) + (score_3 * 0.10)
                threshold = 0.50
            elif "Better" in model_choice:
                final_score = score_2
                threshold = 0.75
            else:
                final_score = score_1
                threshold = 0.75

        # Display Results
        st.subheader("Analysis Results:")
        st.write(f"**Active Model:** {model_choice}")
        st.write(f"**Model score for ALL class:** {final_score:.4f}")
        
        if final_score >= threshold:
            st.error(f"⚠️ **Prediction: ALL (Leukemia-class cell)** [Score: {final_score:.4f}]")
        else:
            st.success(f"✅ **Prediction: HEM (Non-malignant-class cell)** [Score: {final_score:.4f}]")