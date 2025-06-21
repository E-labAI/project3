import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# Cargar modelo entrenado
model = tf.keras.models.load_model("modelo_entrenado_resnet50.keras")

# Obtener nombres de clases desde carpeta train
train_ds = tf.keras.utils.image_dataset_from_directory(
    "imgtrain",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)
class_names = train_ds.class_names

# Función para predecir una imagen
def predict_image(image):
    img = tf.image.resize(image, (224, 224))
    img = tf.expand_dims(img, 0)  # batch size = 1
    prediction = model(img, training=False)
    score = tf.nn.softmax(prediction[0])
    return np.argmax(score), float(tf.reduce_max(score))  # clase predicha, probabilidad

# Automatizar evaluación en carpeta test/
test_dir = "imgtest"
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

            # Actualizar métricas
            results[class_name]["total"] += 1
            if pred_class == class_name:
                results[class_name]["correct"] += 1
        except:
            print(f"Error con imagen: {img_file}")

# Calcular precisión por clase
labels = []
accuracies = []

for class_name, stats in results.items():
    total = stats["total"]
    correct = stats["correct"]
    accuracy = correct / total * 100 if total > 0 else 0
    labels.append(class_name)
    accuracies.append(accuracy)
    print(f"{class_name}: {accuracy:.2f}%")

# Graficar resultados
plt.figure(figsize=(10, 6))
plt.bar(labels, accuracies, color='skyblue')
plt.xlabel("Clases")
plt.ylabel("Precisión (%)")
plt.title("Precisión del modelo por clase")
plt.ylim(0, 100)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()