import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt

# Data loading
train_ds = tf.keras.utils.image_dataset_from_directory(
    "train",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),  # ResNet50 expects 224x224 images
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "valid",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "test",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

num_classes = 100

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2),
])

# Transfer learning with ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model

# Build model
inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.resnet50.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30  # More epochs for better learning
)

# Save the model
model.save("modelo_entrenado_resnet50.keras")

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_ds)
print("Precisión en test:", test_acc)

# Plot training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Entrenamiento')
plt.plot(val_acc, label='Validación')
plt.title('Precisión')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Entrenamiento')
plt.plot(val_loss, label='Validación')
plt.title('Pérdida')
plt.legend()
plt.show()

# Gradio interface
def classify_image(image):
    # Preprocess the image
    image = tf.image.resize(image, (224, 224))
    image = tf.expand_dims(image, axis=0)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    
    # Make prediction
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    # Get class names (assuming they're available)
    class_names = train_ds.class_names if hasattr(train_ds, 'class_names') else [str(i) for i in range(num_classes)]
    
    return {class_names[i]: float(predictions[0][i]) for i in range(num_classes)}

# Create Gradio interface
interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(shape=(224, 224)),
    outputs=gr.Label(num_top_classes=5),
    title="Clasificador de mariposas y polillas (ResNet50)",
    description="Sube una imagen de un lepidóptero y el modelo te dirá la especie."
)

# Launch the interface
interface.launch()