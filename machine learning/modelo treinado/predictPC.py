import numpy as np
import tensorflow as tf
from PIL import Image
import sys
import os

diretorio_base = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(diretorio_base, "ia_treinada", "model.tflite")
labels_path = os.path.join(diretorio_base, "ia_treinada", "labels.txt")
teste_path = os.path.join(diretorio_base, "imgs Teste", "solopantanoso3.jpg")


tflite = tf.lite


def carregar_labels(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]

# Função para pré-processar imagem
def preprocessar_image(image_path, input_shape, input_dtype):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((input_shape[1], input_shape[2]))
    image = np.array(image)

    if input_dtype == np.float32:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(input_dtype)

    return np.expand_dims(image, axis=0)

def main():
    global teste_path
    if len(sys.argv) > 1:
        teste_path = sys.argv[1]

    # Carrega modelo
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    # Processa imagem
    input_data = preprocessar_image(teste_path, input_shape, input_dtype)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Saída
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.squeeze(output_data)

    # Carregar labels
    labels = carregar_labels(labels_path)

    # Resultado
    top_index = np.argmax(prediction)
    top_label = labels[top_index]
    confidence = prediction[top_index] * 100

    print(f"Classe: {top_label} | Probabilidade: {confidence:.2f}%")

if __name__ == "__main__":
    main()