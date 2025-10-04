# Lung Cancer Detection Using ResNet50 CNN

## Overview
Lung cancer remains one of the leading causes of mortality worldwide, emphasizing the **urgent need for early and accurate detection** to improve patient outcomes. This project leverages **Convolutional Neural Networks (CNNs)**, specifically the **ResNet50 architecture**, to detect lung cancer from **Computed Tomography (CT) scans**. 

ResNet50, with its deep residual connections, excels at analyzing complex medical images and extracting critical features necessary for precise diagnosis. The project highlights **meticulous data preparation** and evaluates performance using metrics such as **accuracy, sensitivity, and specificity**.

---

## Features
- **Deep Learning Model:** Employs ResNet50 for robust image feature extraction.
- **Medical Image Analysis:** Detects lung cancer from CT scan images.
- **High Accuracy:** Optimized to achieve strong performance metrics.
- **Data Preprocessing:** Handles normalization, resizing, and augmentation to improve model robustness.
- **Evaluation:** Uses accuracy, sensitivity, specificity, and other key performance indicators.

---

## Dataset
- **Source:** Kaggle
- **Format:** CT scan images in `.png` / `.jpg`.
- **Classes:**  
  - `Benign`  
  - `Malignant`  

> Special attention is given to class imbalance to ensure accurate model predictions.

---

## Technologies Used
- **Programming Language:** Python
- **Deep Learning Framework:** TensorFlow / Keras
- **Libraries:** OpenCV, NumPy, Pandas, Matplotlib, Scikit-learn

---

## Model Performance
- **Training Accuracy:** ~90%  
- **Testing Accuracy:** ~70%  

> CNN-based approaches, particularly ResNet50, have shown strong performance in lung cancer detection, often achieving accuracies exceeding 90%.

---

## Challenges & Future Work
- **Class Imbalance:** Needs careful handling through augmentation and weighted loss.  
- **Model Generalizability:** Ensuring robustness across diverse patient populations and imaging conditions.   
- **Data Limitations:** Strategies to handle small or biased datasets to enhance reliability in real-world settings.

---



**Keywords:** Image Processing, Feature Extraction, Classification, Model Training, Cancer Detection, ResNet50
