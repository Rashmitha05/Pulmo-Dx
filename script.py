import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from collections import defaultdict

# Paths
main_path = "Lung Disease Dataset"
train_path = os.path.join(main_path, "train")
val_path = os.path.join(main_path, "val")
test_path = os.path.join(main_path, "test")

# Function to get image paths
def get_image_paths(path, sub_dirs):
    images = []
    for sub_dir in sub_dirs:
        images += glob.glob(os.path.join(path, sub_dir, "*.jpeg"))
        images += glob.glob(os.path.join(path, sub_dir, "*.jpg"))
        images += glob.glob(os.path.join(path, sub_dir, "*.png"))
    return images

# Getting image paths for each class
train_classes = ["Bacterial Pneumonia", "Corona Virus Disease", "Normal", "Tuberculosis", "Viral Pneumonia"]
corona_virus_disease_train_images = get_image_paths(train_path, ["Corona Virus Disease"])
normal_train_images = get_image_paths(train_path, ["Normal"])
tuberculosis_train_images = get_image_paths(train_path, ["Tuberculosis"])
viral_pneumonia_train_images = get_image_paths(train_path, ["Viral Pneumonia"])
bacterial_pneumonia_train_images = get_image_paths(train_path, ["Bacterial Pneumonia"])

corona_virus_disease_val_images = get_image_paths(val_path, ["Corona Virus Disease"])
normal_val_images = get_image_paths(val_path, ["Normal"])
tuberculosis_val_images = get_image_paths(val_path, ["Tuberculosis"])
viral_pneumonia_val_images = get_image_paths(val_path, ["Viral Pneumonia"])
bacterial_pneumonia_val_images = get_image_paths(val_path, ["Bacterial Pneumonia"])

corona_virus_disease_test_images = get_image_paths(test_path, ["Corona Virus Disease"])
normal_test_images = get_image_paths(test_path, ["Normal"])
tuberculosis_test_images = get_image_paths(test_path, ["Tuberculosis"])
viral_pneumonia_test_images = get_image_paths(test_path, ["Viral Pneumonia"])
bacterial_pneumonia_test_images = get_image_paths(test_path, ["Bacterial Pneumonia"])

# Creating DataFrame for count plot
data = pd.DataFrame({
    'class': ['Corona Virus Disease'] * len(corona_virus_disease_train_images) +
             ['Normal'] * len(normal_train_images) +
             ['Tuberculosis'] * len(tuberculosis_train_images) +
             ['Viral Pneumonia'] * len(viral_pneumonia_train_images) +
             ['Bacterial Pneumonia'] * len(bacterial_pneumonia_train_images)
})

# Plotting count plot
plt.figure(figsize=(10, 8))
ax = sns.catplot(x="class", data=data, kind="count")
unique_categories = data['class'].unique()
ax.set_xticklabels(unique_categories, rotation=45, ha="right")
plt.ylabel('Count', fontsize=12)
plt.title('Number of cases', fontsize=14)
plt.savefig("case_distribution.png", bbox_inches='tight')
plt.show()

# Function to plot images
def plot_images(image_paths, title):
    fig, axes = plt.subplots(nrows=1, ncols=min(len(image_paths), 6), figsize=(15, 10), subplot_kw={'xticks':[], 'yticks':[]})
    for i, ax in enumerate(axes.flat):
        if i < len(image_paths):
            img = cv2.imread(image_paths[i])
            img = cv2.resize(img, (512, 512))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 512 / 10), -4, 128)
            ax.imshow(img, cmap='gray')
            ax.set_title(title)
    fig.tight_layout()
    plt.show()

# Plotting images
plot_images(normal_train_images, "Normal")
plot_images(corona_virus_disease_train_images, "Corona Virus Disease")
plot_images(tuberculosis_train_images, "Tuberculosis")
plot_images(viral_pneumonia_train_images, "Viral Pneumonia")
plot_images(bacterial_pneumonia_train_images, "Bacterial Pneumonia")

# Function to count class images
def count_class_images(data_dir):
    if os.path.exists(data_dir):
        classes = os.listdir(data_dir)
        class_counts = defaultdict(int)
        for class_name in classes:
            class_path = os.path.join(data_dir, class_name)
            class_counts[class_name] = len(os.listdir(class_path))
        return class_counts
    else:
        print(f"Directory '{data_dir}' not found.")
        return None

# Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.5, 1.5],
    channel_shift_range=20
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# Function to create data generator
def create_data_generator(data_dir, class_counts, mode='train', shuffle=True):
    if class_counts is None:
        return None
    generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=shuffle,
        classes=list(class_counts.keys())
    )
    return generator

# Counting class images
train_class_counts = count_class_images(train_path)
val_class_counts = count_class_images(val_path)
test_class_counts = count_class_images(test_path)

# Creating data generators
train_generator = create_data_generator(train_path, train_class_counts, mode='train')
val_generator = create_data_generator(val_path, val_class_counts, mode='val')
test_generator = create_data_generator(test_path, test_class_counts, mode='test')

# Model setup
base_model = DenseNet201(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(len(train_classes), activation='softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=x)
model.summary()

# Model compilation
initial_learning_rate = 1e-3
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=755,
    decay_rate=0.9,
    staircase=True
)

METRICS = [
    'accuracy',
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')
]

model.compile(
    optimizer=Adam(learning_rate=lr_schedule),
    loss='categorical_crossentropy',
    metrics=METRICS
)

callbacks = EarlyStopping(monitor='accuracy', patience=5)

# Model training
steps_per_epoch = len(train_generator) // 16
validation_steps = len(val_generator) // 16

history = model.fit(
    train_generator,
    epochs=20,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    validation_data=val_generator,
    class_weight=None,
    callbacks=[callbacks]
)

# Model evaluation
test_loss, test_acc, test_precision, test_recall = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc:.4f}, Test precision: {test_precision:.4f}, Test recall: {test_recall:.4f}')

# Plot training history
fig, ax = plt.subplots(1, 4, figsize=(20, 3))
ax = ax.ravel()
metrics = ['accuracy', 'precision', 'recall', 'loss']

for i, met in enumerate(metrics):
    ax[i].plot(history.history[met])
    ax[i].plot(history.history['val_' + met])
    ax[i].set_title('Model {}'.format(met))
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel(met)
    ax[i].legend(['train', 'val'])

# Save the plot as an image
fig.savefig("model_performance.png", bbox_inches='tight')

# Save the model
model.save('my_image_classifier_ntc_93_94_95_Densenet_201.h5')

# Image prediction 
image_path = 'Tuberculosis-103.png'

if not os.path.exists(image_path):
    print("Error: Image file not found.")
else:
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Unable to read the image.")
    else:
        img = cv2.resize(img, (224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Get class labels dictionary from the training generator
        class_labels = train_generator.class_indices
        print("Class Labels:")                              
        for class_name, index in class_labels.items():
            print(f"{class_name}: {index}")

        # Prediction
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions, axis=1)
        disease_labels = list(class_labels.keys())
        predicted_disease = disease_labels[predicted_index[0]]
        print("Predicted index:", predicted_index)
        print("Disease labels:", disease_labels)
        print("Predicted disease:", predicted_disease)

# Classification report
y_true = test_generator.classes
y_pred = model.predict(test_generator)
y_pred = np.argmax(y_pred, axis=1)

# Generate classification report
report = classification_report(y_true, y_pred, target_names=train_classes)
print(report)