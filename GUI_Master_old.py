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

# Custom styles
st.markdown("""
    <style>
    .stButton>button {
        background-color: #007ACC;
        color: white;
        font-size: 16px;
        border-radius: 10px;
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: navy;'>🫁 Lung Cancer Detection using Deep Learning</h1>", unsafe_allow_html=True)
st.markdown("---")

st.sidebar.header("Upload & Patient Info")
uploaded_file = st.sidebar.file_uploader("Upload Lung CT Scan", type=["jpg", "jpeg", "png"])
user = st.sidebar.text_input("Enter Patient Name", value="John Doe")

st.info("Upload a clear Lung CT scan image in JPG, JPEG, or PNG format.")
st.warning("Results are indicative only. Consult a doctor for diagnosis.")

detection_result = None
stored_image = None

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

if uploaded_file:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Run Detection"):
        with st.spinner("Analyzing..."):
            detection_result = test_model_proc(image)
            st.success(f"✅ Detection Complete: {detection_result}")

    stored_image = image

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

if uploaded_file and detection_result:
    if st.button("Generate PDF Report"):
        pdf_file = generate_report(user, detection_result, stored_image)
        st.download_button("Download Report", data=pdf_file, file_name=f"lung_report_{user}.pdf", mime="application/pdf")
