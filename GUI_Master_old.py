import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
import io

# Set page to wide mode
st.set_page_config(layout="wide")

# Custom styles for a modern, attractive look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Roboto', sans-serif;
    }
    
    .stApp {
        background-color: #f0f2f6;
        color: #333333;
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
    
    .st-emotion-cache-1c7y3q {
        background-color: white;
        border-radius: 15px;
        padding: 20px;
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
    
    .sidebar .st-emotion-cache-1j43zcr {
        background-color: #003366;
    }
    
    .sidebar .st-emotion-cache-1j43zcr h2 {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Load model once
@st.cache_resource
def load_lung_model():
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

st.sidebar.header("📂 Upload & Patient Information")
uploaded_file = st.sidebar.file_uploader("Upload Lung CT Scan Image", type=["jpg", "jpeg", "png"])
user = st.sidebar.text_input("Enter Patient Name", value="Jane Doe")

st.info("💡 **Instructions:** Upload a clear Lung CT scan image (JPG, JPEG, or PNG) on the left sidebar. The AI will analyze the image to provide a preliminary result.")
st.warning("⚠️ **Disclaimer:** This tool provides indicative results only. Always consult a qualified medical professional for an accurate diagnosis.")

detection_result = None
stored_image = None

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
                        st.markdown(f"**Result:** {detection_result}")
            
            stored_image = image

            # Display "Generate Report" button only after analysis
            if detection_result:
                st.markdown("---")
                st.subheader("📄 Generate Report")
                if st.button("Download PDF Report"):
                    pdf_file = generate_report(user, detection_result, stored_image)
                    st.download_button(
                        "Click to Download",
                        data=pdf_file,
                        file_name=f"Lung_Scan_Report_{user}_{datetime.datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        help="Generates and downloads a detailed PDF report of the analysis."
                    )
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

def generate_report(user, detection_result, image):
    """Generates a professional PDF report."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    
    # Border and title
    c.setStrokeColor(colors.black)
    c.setLineWidth(1.5)
    c.rect(20, 20, letter[0] - 40, letter[1] - 40)
    
    c.setFont("Helvetica-Bold", 24)
    c.setFillColor(colors.darkblue)
    c.drawString(160, 750, "LungScan AI Diagnostics Report")
    c.setFont("Helvetica", 12)
    c.setFillColor(colors.gray)
    c.drawString(160, 730, "Automated Lung Imaging Analysis")
    
    c.line(50, 720, 550, 720)

    # Patient and report details
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.black)
    c.drawString(50, 690, "Patient Name:")
    c.setFont("Helvetica", 14)
    c.drawString(200, 690, user)

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 670, "Date of Report:")
    c.setFont("Helvetica", 14)
    c.drawString(200, 670, datetime.datetime.now().strftime('%B %d, %Y'))
    
    # Analysis results
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 630, "Analysis Summary:")
    c.setFont("Helvetica", 14)
    c.drawString(200, 630, detection_result)
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 600, "Image Used for Analysis:")
    
    # Image inclusion
    img_reader = ImageReader(image)
    c.drawImage(img_reader, 50, 380, width=200, height=200)

    # Disclaimer
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(colors.red)
    c.drawString(50, 150, "Important Medical Disclaimer:")
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.black)
    disclaimer_text = """This report is generated by an AI model and is intended for informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. The results should be interpreted by a qualified healthcare professional in conjunction with other clinical data. The creators of this application are not liable for any clinical decisions made based on this report."""
    textobject = c.beginText(50, 135)
    textobject.setFont("Helvetica", 10)
    for line in disclaimer_text.split('\n'):
        textobject.textLine(line)
    c.drawText(textobject)
    
    c.save()
    buffer.seek(0)
    return buffer
