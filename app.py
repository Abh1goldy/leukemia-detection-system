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

# Function to download models if missing
def download_model_if_needed(filename):
    if not os.path.exists(filename):
        file_id = MODEL_DRIVE_IDS.get(filename)
        if file_id:
            url = f'https://drive.google.com/uc?id={file_id}'
            with st.spinner(f'Downloading {filename} from cloud storage... Please wait.'):
                gdown.download(url, filename, quiet=False)

# Sidebar model selection
model_choice = st.sidebar.selectbox(
    "Select Model Architecture",
    ("Model 3: Best", "Model 2: Better", "Model 1: Good")
)

# Helper function to get custom object preprocessing reference for model loading
def get_preprocess_function(choice):
    if "Best" in choice:
        return tf.keras.applications.efficientnet.preprocess_input
    elif "Better" in choice:
        return tf.keras.applications.densenet.preprocess_input
    else:
        return tf.keras.applications.resnet.preprocess_input

# Load the selected model with dynamic custom_objects support
@st.cache_resource
def load_selected_model(choice):
    if "Best" in choice:
        filename = 'model_3_best.keras'
    elif "Better" in choice:
        filename = 'model_2_better.keras'
    else:
        filename = 'model_1_good.keras'

    download_model_if_needed(filename)
    prep_func = get_preprocess_function(choice)
    
    try:
        return load_model(filename, custom_objects={'preprocess_input': prep_func})
    except Exception:
        return load_model(filename)

try:
    model = load_selected_model(model_choice)
except Exception as e:
    st.error(f"Error loading model file. Details: {e}")

# Image uploader widget
uploaded_file = st.file_uploader("Choose a single blood cell image (.jpg, .jpeg, .png, .bmp)", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Blood Cell Image', use_column_width=True)
    
    if st.button('Predict Cell Type'):
        with st.spinner('Analyzing cell with single run inference...'):
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized, dtype=np.float32)
            
            # Apply preprocessing conditionally per model architecture
            if "Best" in model_choice:
                # Model 3 has Lambda preprocessing built-in; keep raw values
                pass
            elif "Better" in model_choice:
                img_array = tf.keras.applications.densenet.preprocess_input(img_array)
            else:
                img_array = tf.keras.applications.resnet.preprocess_input(img_array)
                
            img_array = np.expand_dims(img_array, axis=0)
            
            # Run prediction
            score = float(model.predict(img_array, verbose=0)[0][0])
            
        # Display Results matching your Colab evaluation outputs
        st.subheader("Analysis Results:")
        st.write(f"**Active Model:** {model_choice}")
        st.write(f"**Model score for ALL class:** {score:.4f}")
        
        if score >= 0.75:
            st.error(f"⚠️ **Prediction: ALL (Leukemia-class cell)** [Score: {score:.4f}]")
        else:
            st.success(f"✅ **Prediction: HEM (Non-malignant-class cell)** [Score: {score:.4f}]")