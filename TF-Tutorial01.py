# Tutorial -> https://www.tensorflow.org/tutorials/keras/classification

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# 0: Obtención del dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 1: Pretratamiento de los datos (en este caso una normalización con una distribución de mediana 0 y desviación 1)
train_images = train_images / 255.0
test_images = test_images / 255.0

# 2: Creación del modelo
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# 3: Función de optimización
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4: Entrenamiento         
model.fit(train_images, train_labels, epochs=10)

# 5: Evaluación del modelo
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# 6: Hacer predicciones
predictions = model.predict(test_images) # Devuelve la última capa, en este caso un vector con la posibilidad de pertenecer a cada clase
cont = 1
for pr in predictions:
    print(("Imagen {} ha predicho {} y lo correcto es {}").format(cont, np.argmax(pr), test_labels[cont - 1]))
    cont += 1