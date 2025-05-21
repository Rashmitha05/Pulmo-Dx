import tensorflow as tf
from tensorflow.keras.applications import DenseNet201

# Load the DenseNet201 model
base_model = DenseNet201(weights='imagenet', include_top=False)

# Rename layers to remove forward slashes
for layer in base_model.layers:
    layer._name = layer.name.replace('/', '_')

# Create a new model with renamed layers
model = tf.keras.models.Sequential()
model.add(base_model)

# Save the model
model.save('my_image_classifier.h5')
