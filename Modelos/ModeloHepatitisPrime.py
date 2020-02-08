import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0, "C:\\Users\\mario\\Documents\\GitHub\\TensorFlow-Learning\\Preprocesamiento")
import PreprocesamientoHepPrime as proc
from tensorflow import keras

# 1: Obtención y procesamiento de datos
procesador = proc.ProcessorHepPrime("C:\\Users\\mario\\Documents\\GitHub\\TensorFlow-Learning\\Datos\\hepatitis_csv.csv", 0.2)
(samples_data, samples_labels) = procesador.get_data()

# 2: Creación de la red neuronal
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='tanh'),
    keras.layers.Dense(32, activation='tanh'),
    keras.layers.Dense(16, activation='tanh'),
    keras.layers.Dense(8, activation='tanh'),
    keras.layers.Dense(4, activation='tanh'),
    keras.layers.Dense(procesador.num_class, activation='softmax')
])

# 3: Función de optimización y metrica a optimizar
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4: Entrenamiento del modelo         
model.fit(
    x = samples_data, 
    y = samples_labels, 
    epochs = 20,
    validation_split = procesador.test_percentage
    )

# *WIP* 5: Hacer predicciones, evaluar el modelo y visualización de los resultados
predictions = model.predict(samples_data) # Devuelve la última capa, en este caso un vector con la posibilidad de pertenecer a cada clase
cont = 1
for pr in predictions:
    #print(("Muestra {} ha predicho {} y lo correcto es {}").format(cont, np.argmax(pr), int(samples_labels[cont - 1])))
    cont += 1