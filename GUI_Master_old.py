import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
import os
import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors

# Load model once
@st.cache_resource
def load_lung_model():
    return load_model('lung_model.h5')

model = load_lung_model()

st.title("Lung Cancer Detection using Deep Learning")

# Upload image
uploaded_file = st.file_uploader("Upload a Lung CT Scan", type=["jpg", "jpeg", "png"])

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

# Show and process image
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Detection"):
        with st.spinner("Analyzing..."):
            detection_result = test_model_proc(image)
            st.success(f"Result: {detection_result}")

    # Save the uploaded image for report
    stored_image = image

# Report generation
def generate_report(user, detection_result, image):
    filename = f"lung_report_{user}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    temp_img_path = "temp_uploaded_image.jpg"
    image.save(temp_img_path)

    c = canvas.Canvas(filename, pagesize=letter)
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

    c.drawImage(temp_img_path, 50, 420, width=200, height=200)

    c.drawString(50, 100, "Thank you for using LungScan AI")
    c.drawString(50, 80, "Report generated automatically. Please consult a doctor for confirmation.")

    c.save()
    os.remove(temp_img_path)
    return filename

if uploaded_file and detection_result:
    user = st.text_input("Enter Patient Name", value="John Doe")
    if st.button("Generate PDF Report"):
        pdf_file = generate_report(user, detection_result, stored_image)
        with open(pdf_file, "rb") as f:
            st.download_button("Download Report", data=f, file_name=pdf_file, mime="application/pdf")
