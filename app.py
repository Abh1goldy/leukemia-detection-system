import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(page_title="Leukemia Detection System", page_icon="🔬", layout="centered")

st.title("🔬 Automated Leukemia Detection System")
st.write("Upload a single cropped blood smear cell image to classify it using your trained deep learning models.")

# Sidebar model selection
model_choice = st.sidebar.selectbox(
    "Select Model Architecture",
    ("Model 3: MobileNetV2", "Model 2: Deeper Custom CNN", "Model 1: Baseline Custom CNN")
)

# Load the selected model with caching
@st.cache_resource
def load_selected_model(choice):
    if "Model 3" in choice:
        return load_model('model_3_best.keras')
    elif "Model 2" in choice:
        return load_model('model_2_best.keras')
    else:
        return load_model('model_1_basic.keras')

try:
    model = load_selected_model(model_choice)
except Exception as e:
    st.error(f"Error loading model file. Make sure your .keras files are in the same folder as app.py! Details: {e}")

# Image uploader widget
uploaded_file = st.file_uploader("Choose a single blood cell image (.jpg, .jpeg, .png, .bmp)", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Blood Cell Image', use_column_width=True)
    
    # Single run prediction button
    if st.button('Predict Cell Type'):
        with st.spinner('Analyzing cell with single run inference...'):
            # All models were trained on 224x224 with preprocess_input based on your notebook setup
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized, dtype=np.float32)
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
                
            img_array = np.expand_dims(img_array, axis=0)
            
            # Run prediction
            prediction = float(model.predict(img_array, verbose=0)[0][0])
            
        # Display Results
        st.subheader("Analysis Results:")
        st.write(f"**Active Model:** {model_choice}")
        st.write(f"**Raw Prediction Score:** {prediction:.4f}")
        
        # Corrected threshold logic: High score = ALL, Low score = HEM
        if prediction >= 0.80:
            st.error(f"⚠️ **Result: ALL (Leukemia - Malignant Cell)** [Confidence: {prediction * 100:.2f}%]")
        #elif prediction >= 0.75:
           # st.warning(f"⚠️ **Result: Borderline / Uncertain Cell** [Score: {prediction:.4f}] — *Recommended for Pathologist Review*")
        else:
            st.success(f"✅ **Result: HEM (Non-malignant / Normal Cell)** [Confidence: {(1 - prediction) * 100:.2f}%]")