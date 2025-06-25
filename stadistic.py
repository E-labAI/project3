import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("modelo_entrenado_resnet50.keras")

# Get class names from train folder
train_ds = tf.keras.utils.image_dataset_from_directory(
    "train",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)
class_names = train_ds.class_names

# Function to predict an image
def predict_image(image):
    img = tf.image.resize(image, (224, 224))
    img = tf.expand_dims(img, 0)  # batch size = 1
    prediction = model(img, training=False)
    score = tf.nn.softmax(prediction[0])
    return np.argmax(score), float(tf.reduce_max(score))  # predicted class, probability

# Select 10 random images from the training set for visualization
plt.figure(figsize=(20, 12))
selected_images = []
train_dir = "train"

# Get 10 random images (2 per class if possible)
for i, class_name in enumerate(class_names):
    class_path = os.path.join(train_dir, class_name)
    if os.path.isdir(class_path):
        images = os.listdir(class_path)
        if len(images) >= 2:
            selected = random.sample(images, min(2, len(images)))
            for img_file in selected:
                selected_images.append((class_name, img_file))
        else:
            for img_file in images:
                selected_images.append((class_name, img_file))
                
# Ensure we have exactly 10 images
selected_images = random.sample(selected_images, min(10, len(selected_images)))

# Plot predictions for selected images
for idx, (true_class, img_file) in enumerate(selected_images, 1):
    img_path = os.path.join(train_dir, true_class, img_file)
    image = Image.open(img_path).convert("RGB")
    image_array = np.array(image)
    
    pred_index, prob = predict_image(image_array)
    pred_class = class_names[pred_index]
    
    plt.subplot(2, 5, idx)
    plt.imshow(image)
    plt.title(f"True: {true_class}\nPred: {pred_class}\nProb: {prob:.2f}")
    plt.axis('off')

plt.suptitle("Model Predictions on Training Samples", fontsize=16)
plt.tight_layout()
plt.show()

# Original evaluation code on test set
test_dir = "test"
results = {name: {"correct": 0, "total": 0} for name in class_names}

for class_name in os.listdir(test_dir):
    class_path = os.path.join(test_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    
    for img_file in os.listdir(class_path):
        try:
            img_path = os.path.join(class_path, img_file)
            image = Image.open(img_path).convert("RGB")
            image = np.array(image)

            pred_index, prob = predict_image(image)
            pred_class = class_names[pred_index]

            # Update metrics
            results[class_name]["total"] += 1
            if pred_class == class_name:
                results[class_name]["correct"] += 1
        except:
            print(f"Error with image: {img_file}")

# Calculate accuracy per class
labels = []
accuracies = []

for class_name, stats in results.items():
    total = stats["total"]
    correct = stats["correct"]
    accuracy = correct / total * 100 if total > 0 else 0
    labels.append(class_name)
    accuracies.append(accuracy)
    print(f"{class_name}: {accuracy:.2f}%")

# Plot results
plt.figure(figsize=(10, 6))
plt.bar(labels, accuracies, color='skyblue')
plt.xlabel("Classes")
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy by Class")
plt.ylim(0, 100)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()