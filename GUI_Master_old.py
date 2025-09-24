import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
import datetime
import io

# Set page to wide mode and hide the default Streamlit sidebar and its scrollbar
st.set_page_config(layout="wide")

# Custom styles for a modern, attractive look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Roboto', sans-serif;
    }
    
    .stApp {
        /* Modern, neutral gradient background */
        background-image: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
        color: #2c3e50; /* A dark, professional gray */
    }
    
    .stButton>button {
        background-color: #27ae60; /* A calming green for action */
        color: white;
        font-size: 16px;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        background-color: #229954;
        transform: scale(1.05);
    }
    
    /* Container for the Upload section - a soft, warm background */
    .st-emotion-cache-1c7y3q.stContainer:nth-of-type(1) {
        background-color: #ffffff; /* White for cleanliness */
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        margin-top: 20px;
        margin-bottom: 20px;
    }
    
    /* Container for the Analysis section - a slightly different background */
    .st-emotion-cache-1c7y3q.stContainer:nth-of-type(2) {
        background-color: #f8f9fa; /* A very light gray for separation */
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .st-emotion-cache-1r6ftj { /* This is for st.info */
        background-color: #e8f6f3; /* Light teal for success/info */
        color: #16a085;
        border-left: 5px solid #1abc9c;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .st-emotion-cache-q82v5q { /* This is for st.warning */
        background-color: #fef9e7; /* Light yellow for warnings */
        color: #f39c12;
        border-left: 5px solid #f1c40f;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    h1 {
        color: #34495e; /* Dark professional gray */
        text-align: center;
        font-weight: 700;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }

    /* Target the main content block for padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 5%;
        padding-right: 5%;
    }
    
    /* Ensure scrollbar is hidden on specific elements */
    .st-emotion-cache-1r6ftj {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)

# Load model once
@st.cache_resource
def load_lung_model():
    """Loads the pre-trained Keras model."""
    try:
        model = load_model('lung_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

model = load_lung_model()

# --- Main App Content ---
st.markdown("<h1 style='text-align: center;'>🫁 LungScan AI: Deep Learning-Powered Lung CT Scan Analysis</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #555;'>A tool for preliminary screening of lung conditions.</h3>", unsafe_allow_html=True)
st.markdown("---")

# Use a container for the input section to make it look cleaner
with st.container(border=True):
    st.header("Upload & Patient Information")
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Upload Lung CT Scan Image", type=["jpg", "jpeg", "png"])
    
    with col2:
        user = st.text_input("Enter Patient Name", value="Jane Doe")

st.info("💡 **Instructions:** Upload a clear Lung CT scan image (JPG, JPEG, or PNG). The AI will analyze the image to provide a preliminary result.")
st.warning("⚠️ **Disclaimer:** This tool provides indicative results only. Always consult a qualified medical professional for an accurate diagnosis.")

detection_result = None

def test_model_proc(img: Image.Image):
    """Processes the image and predicts the lung condition."""
    if model is None:
        return "Model not loaded. Please try again."

    IMAGE_SIZE = 64
    try:
        img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.array(img).astype("float32") / 255.0
        img_array = img_array.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)

        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        
        confidence = prediction[0][class_index] * 100

        if class_index == 0:
            return f"Benign Lung Cancer Detected (Small) - {confidence:.2f}% Confidence"
        elif class_index == 1:
            return f"Malignant Lung Cancer Detected (Large) - {confidence:.2f}% Confidence"
        elif class_index == 2:
            return f"Normal Lung Condition Detected - {confidence:.2f}% Confidence"
        else:
            return "Unknown condition"
    except Exception as e:
        return f"Error during analysis: {e}"

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        
        st.markdown("---")
        
        # Container for the results display
        with st.container(border=True):
            st.header("Analysis")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("🖼️ Uploaded CT Scan")
                st.image(image, caption="CT Scan Image", use_container_width=True)

            with col2:
                st.subheader("🔬 Analysis Result")
                if st.button("🚀 Run Analysis"):
                    with st.spinner("Analyzing image... This may take a moment."):
                        detection_result = test_model_proc(image)
                        if "Error" in detection_result or "Unknown" in detection_result:
                            st.error(f"❌ Analysis failed: {detection_result}")
                        else:
                            st.success(f"✅ **Analysis Complete!**")
                            
                            st.markdown("---")
                            st.subheader("📋 Final Report")
                            st.markdown(f"**Patient Name:** {user}")
                            st.markdown(f"**Date of Analysis:** {datetime.datetime.now().strftime('%B %d, %Y')}")
                            st.markdown(f"**Analysis Result:** **{detection_result}**")
                            st.markdown("---")
                            
                            st.info("Remember to consult a doctor for a professional diagnosis. This report is for informational purposes only.")
                            
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
