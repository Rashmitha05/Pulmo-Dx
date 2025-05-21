import numpy as np
import cv2
import tensorflow as tf
import os

# Load the trained model from disk
model_path = 'model/my_image_classifier_ntc_93_94_95_Densenet_201.h5'
model = tf.keras.models.load_model(model_path)

def predict_image(image_path):
    if not os.path.exists(image_path):
        return "Error: Image file not found."
    
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Unable to read the image."
    
    img = cv2.resize(img, (224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Define class labels
    class_labels = ["Bacterial Pneumonia", "Corona Virus Disease", "Normal", "Tuberculosis", "Viral Pneumonia"]

    # Prediction
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions, axis=1)
    predicted_disease = class_labels[predicted_index[0]]
    return predicted_disease
