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

# Load model once
@st.cache_resource
def load_lung_model():
    return load_model('lung_model.h5')

model = load_lung_model()

# ---- CUSTOM CSS ----
page_bg_img = """
<style>
    /* Background image with overlay */
    .stApp {
        background: linear-gradient(rgba(255,255,255,0.85), rgba(255,255,255,0.85)), 
                    url("https://images.unsplash.com/photo-1581093588401-7c5bbfcbfab5?auto=format&fit=crop&w=1470&q=80");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Header styling */
    h1, h2, h3 {
        color: #003366;
        font-weight: 700;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #007ACC;
        color: white;
        font-weight: 600;
        font-size: 16px;
        border-radius: 12px;
        padding: 12px 28px;
        transition: background-color 0.3s ease;
        box-shadow: 2px 4px 8px rgba(0, 0, 0, 0.2);
    }
    div.stButton > button:hover {
        background-color: #005f99;
        cursor: pointer;
    }

    /* Sidebar headers */
    .css-1d391kg {
        font-size: 20px;
        font-weight: 700;
        color: #004080;
        margin-bottom: 10px;
    }

    /* Info and warning boxes with custom colors */
    .stAlert > div[data-testid="stAlertContent"] {
        font-size: 15px;
        font-weight: 600;
    }

    /* Columns spacing */
    .css-1lcbmhc {
        gap: 40px;
    }
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# ---- PAGE HEADER ----
st.markdown("<h1 style='text-align: center; margin-bottom: 5px;'>🫁 Lung Cancer Detection AI</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #005f99; margin-top: 0;'>Powered by Deep Learning</h3>", unsafe_allow_html=True)
st.markdown("---")

# ---- SIDEBAR ----
st.sidebar.header("🔍 Upload & Patient Info")
uploaded_file = st.sidebar.file_uploader("Upload Lung CT Scan", type=["jpg", "jpeg", "png"])
user = st.sidebar.text_input("Patient Name", value="John Doe")

st.sidebar.markdown(
    """
    <small style='color:#444;'>
    Please upload a clear CT scan image in JPG, JPEG, or PNG format.<br>
    Detection results are indicative and not a substitute for professional medical advice.
    </small>
    """, unsafe_allow_html=True)

detection_result = None
stored_image = None

# ---- MODEL INFERENCE FUNCTION ----
def test_model_proc(img: Image.Image):
    IMAGE_SIZE = 64
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(img).astype("float32") / 255.0

    if len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)
    elif img.shape[-1] != 3:
        return "Invalid image format"

    img = img.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)

    prediction = model.predict(img)
    plant = np.argmax(prediction)

    if plant == 0:
        return "Benign Lung Cancer Detected (Small)"
    elif plant == 1:
        return "Malignant Lung Cancer Detected (Large)"
    elif plant == 2:
        return "Normal Lung Condition Detected"
    else:
        return "Unknown condition"

# ---- MAIN APP LAYOUT ----
if uploaded_file:
    image = Image.open(uploaded_file)
    stored_image = image

    col1, col2 = st.columns([1,1])

    with col1:
        st.image(image, caption="Uploaded CT Scan", use_container_width=True, clamp=True)

    with col2:
        if st.button("Run Detection"):
            with st.spinner("Analyzing the scan..."):
                detection_result = test_model_proc(image)
                st.success(f"✅ Detection Result: {detection_result}")

        if detection_result:
            # Highlight result in a colored box
            st.markdown(
                f"""
                <div style='
                    background-color: #e6f2ff; 
                    border-left: 6px solid #007ACC; 
                    padding: 15px; 
                    border-radius: 8px;
                    margin-top: 15px;
                    font-weight: 600;
                    font-size: 18px;
                '>
                {detection_result}
                </div>
                """, unsafe_allow_html=True)

            if st.button("Generate PDF Report"):
                pdf_file = generate_report(user, detection_result, stored_image)
                st.download_button(
                    "📄 Download PDF Report",
                    data=pdf_file,
                    file_name=f"lung_report_{user.replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    key="download-pdf"
                )
else:
    st.info("Please upload a Lung CT scan image to get started.")

# ---- REPORT GENERATION FUNCTION ----
def generate_report(user, detection_result, image):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setStrokeColor(colors.black)
    c.setLineWidth(1.5)
    c.rect(20, 20, letter[0] - 40, letter[1] - 40)

    c.setFont("Helvetica-Bold", 18)
    c.drawString(180, 750, "LungScan Diagnostics Center")
    c.setFont("Helvetica", 12)
    c.drawString(180, 735, "Comprehensive Lung Imaging Report")
    c.line(50, 720, 550, 720)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 700, "Patient Name:")
    c.drawString(250, 700, user)

    c.drawString(50, 680, "Date of Report:")
    c.drawString(250, 680, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    c.drawString(50, 660, "Scan Result:")
    c.setFont("Helvetica", 12)
    c.drawString(250, 660, detection_result)

    img_reader = ImageReader(image)
    c.drawImage(img_reader, 50, 420, width=200, height=200)

    c.drawString(50, 100, "Thank you for using LungScan AI")
    c.drawString(50, 80, "Report generated automatically. Please consult a doctor for confirmation.")

    c.save()
    buffer.seek(0)
    return buffer
