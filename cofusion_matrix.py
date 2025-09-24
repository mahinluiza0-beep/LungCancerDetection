# -*- coding: utf-8 -*-
"""
Created on Wed May 7 19:06:56 2025

@author: luiza
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

def generate_confusion_matrix(model, test_set):
    # Step 1: Predict the classes for the test set
    test_set.reset()  # Ensure the test_set is reset before predicting
    predictions = model.predict(test_set, steps=len(test_set), verbose=1)
    
    # Step 2: Get the true labels from the test_set
    true_labels = test_set.classes
    
    # Step 3: Get the predicted class indices
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Step 4: Generate the confusion matrix
    cm = confusion_matrix(true_labels, predicted_classes)
    
    # Step 5: Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=test_set.class_indices.keys(),
                yticklabels=test_set.class_indices.keys())
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    # Step 6: Print classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_classes, target_names=list(test_set.class_indices.keys())))

# ------------------------
# Paths and Data Loading
# ------------------------

basepath = "C:\\Users\\luiza\\Downloads\\lung cancer 100%-20240810T091253Z-001\\lung cancer 100_"

# Use augmentation even on test data to test robustness (or keep simple if you want)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load the test set
test_set = test_datagen.flow_from_directory(
    basepath + '/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Do NOT shuffle for correct label matching
)

# Load the trained model
classifier = load_model(basepath + '/lung_model.h5')

# Generate confusion matrix for the test set
generate_confusion_matrix(classifier, test_set)

# ------------------------
# (Optional) TIP:
# To improve accuracy even more,
# consider retraining the model using:
# - More layers (CNN)
# - Dropout to reduce overfitting
# - Augmentation on training data
# - More epochs
# ------------------------
