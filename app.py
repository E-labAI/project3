import gradio as gr
import tensorflow as tf
import numpy as np


# Load the trained model (now using ResNet50-based model)
model = tf.keras.models.load_model("modelo_entrenado_resnet50.keras")

# Load dataset just to get class names (with correct image size)
train_ds = tf.keras.utils.image_dataset_from_directory(
    "train",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),  # Changed to 224x224 to match ResNet50
    batch_size=32
)

class_names = train_ds.class_names  # List of classes from the dataset

# Prediction function adjusted for ResNet50 preprocessing
def predict_image(image):
    # Resize and preprocess the image for ResNet50
    img = tf.image.resize(image, (224, 224))
    img = tf.expand_dims(img, 0)  # Create batch of size 1
    img = tf.keras.applications.resnet50.preprocess_input(img)
    
    # Make prediction
    prediction = model.predict(img)
    score = tf.nn.softmax(prediction[0])
    
    # Return top predictions
    return {class_names[i]: float(score[i]) for i in range(len(class_names))}

# Crear un tema personalizado (por ejemplo, con colores suaves y contrastes agradables)
custom_theme = gr.themes.Base(
    primary_hue="indigo",  # Color principal del botón, barra superior, etc.
    neutral_hue="gray",    # Color base de fondo y texto
    font=["sans-serif"],   # Fuente
).set(
    background_fill_primary="#f0f4f8",  # Color de fondo claro
    border_color_primary="#4b5563",     # Bordes de secciones
    text_color_primary="#111827",       # Texto principal
    button_primary_background_fill="#4f46e5",  # Botón principal
    button_primary_text_color="#ffffff"
)

# Create interface
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=5),  # Show top 5 predictions
    title="🦋 Clasificador de mariposas y polillas (ResNet50)",
    description="Sube una imagen de un lepidóptero y el modelo te dirá si es una mariposa o una polilla.",
    theme=custom_theme
)

interface.launch()