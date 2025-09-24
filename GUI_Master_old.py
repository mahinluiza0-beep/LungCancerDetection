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
    
    .main {
        background-color: #f0f2f6;
        color: #333333;
        padding: 20px 50px;
    }
    
    .stButton>button {
        background-color: #007ACC;
        color: white;
        font-size: 16px;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
        transition: all 0.3s ease;
        border: none;
    }
    
    .stButton>button:hover {
        background-color: #005f99;
        transform: scale(1.05);
    }
    
    .custom-container {
        background-color: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-top: 20px;
        margin-bottom: 20px;
    }
    
    .st-emotion-cache-1r6ftj {
        background-color: #e6f3ff;
        color: #004d99;
        border-radius: 10px;
        padding: 10px;
    }
    
    .st-emotion-cache-q82v5q {
        background-color: #fcf8e3;
        color: #8a6d3b;
        border-radius: 10px;
        padding: 10px;
    }
    
    h1 {
        color: #003366;
        text-align: center;
        font-weight: 700;
        text-shadow: 1px 1px 2px #cccccc;
    }

    /* Hide the default sidebar and its scroller */
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
