import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image
import cv2  # OpenCV
import time
import os

model_path = "ia_treinada/model.tflite"
labels_path = "ia_treinada/labels.txt"
image_path = "captura.jpg"

# Carrega labels
def carregar_labels(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]

# Captura imagem da câmera e salva
def capturar_imagem(path="captura.jpg"):
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise Exception("Não foi possível acessar a câmera.")
    
    time.sleep(2)  # Espera a câmera estabilizar
    ret, frame = cam.read()
    cam.release()

    if not ret:
        raise Exception("Falha ao capturar imagem da câmera.")

    cv2.imwrite(path, frame)
    print(f"Imagem salva em {path}")

# Pré-processa imagem
def preprocessar_image(image_path, input_shape, input_dtype, quant_params=None):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((input_shape[1], input_shape[2]))
    image = np.array(image)

    if input_dtype == np.uint8 and quant_params:
        scale, zero_point = quant_params
        image = (image / 255.0) / scale + zero_point
        image = np.clip(image, 0, 255).astype(np.uint8)
    elif input_dtype == np.float32:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(input_dtype)

    return np.expand_dims(image, axis=0)

def main():
    capturar_imagem(image_path)

    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    # Detecta se modelo é quantizado
    quant_params = input_details[0].get('quantization', None)
    if quant_params == (0.0, 0):
        quant_params = None  # Não é quantizado

    input_data = preprocessar_image(image_path, input_shape, input_dtype, quant_params)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Desscalar saída se quantizada
    if output_details[0]['dtype'] == np.uint8:
        scale, zero_point = output_details[0]['quantization']
        output_data = scale * (output_data.astype(np.float32) - zero_point)

    prediction = np.squeeze(output_data)
    labels = carregar_labels(labels_path)

    top_index = np.argmax(prediction)
    top_label = labels[top_index]
    confidence = prediction[top_index] * 100

    print(f"Classe: {top_label} | Probabilidade: {confidence:.2f}%")

if __name__ == "__main__":
    main()