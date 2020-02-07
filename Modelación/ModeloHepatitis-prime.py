# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

class PreprocessorHepa:
    input_data = []                     # Array bidimensional o lista de arrays que contendra los datos obtenidos del fichero de entrada
    num_class = 2                       # Entero que indica el número de diferentes clases a predecir
    delimiter = ','                     # Elemento delimitador para separar cada dato del fichero de entrada
    test_percentage = 0.1               # Decimal que representa el porcentaje de muestras para evaluar el modelo [0.1 - 0.9]
    
    variables_name = []                 # *WIP* Array que contiene los nombres de las variables
    ar_dict = {                         # Diccionario utilizado en el preprocesamiento de datos
        "False": 0.0,
        "True": 1.0,
        "male": 0.0,
        "female": 1.0,
        "live\n": 0.0,
        "die\n": 1.0
    }

    def __init__(self, path : str, test_percentage : float):
        self.test_percentage = test_percentage         
        with open(path, 'r') as file_r:
            self.variables_name = file_r.readline().split(',')
            self.input_data = list(map(lambda x: x.split(self.delimiter), file_r.readlines()))
            input_aux = []
            for aux in self.input_data:
                sample = self.__map_data(aux)
                if sample != []:
                    input_aux.append(sample)
            self.input_data = input_aux
            self.__normalize_data()

    # Devuelve una tupla de dos tensores, los datos y la clases a predecir
    def get_data(self):
        t_data = tf.constant(                                       # Tensor de rango 2 que contiene los datos
            value = self.input_data, 
            shape = [len(self.input_data), len(self.input_data[0])]
            )  
        inp_aux = np.array(self.input_data)
        t_label = tf.constant(                                      # Tensor de rango 2 que contiene las predicciones
            value = inp_aux[:, len(inp_aux[0]) - 1]
            )

        # Devolvemos una tupla de dos tensores donde cada tensor ha sido transformado en uno concatenando sus elementos en el eje 0
        return (tf.concat(t_data, 0), tf.concat(t_label, 0))

    # Preprocesa los datos. Elimina las muestras corruptas
    def __map_data(self, sample):
        res = []
        for aux in sample:
            if aux == '':
                return []
            elif aux in self.ar_dict:
                res.append(self.ar_dict[aux])
            else:
                res.append(aux)
        return res

    # Preprocesa los datos. Normaliza las variables continuas
    def __normalize_data(self):
        normalize_pos = [0, 13, 14, 15, 16, 17]         # Array que contiene las posiciones de las variables continuas
        normalize_max = [-1] * len(normalize_pos)       # Array que contiene el máximo valor de cada variable continua
        normalize_avg = [0] * len(normalize_pos)        # *WIP* Array que contiene la media de cada varriable continua
        res = []                                        # Array que contendra los datos normalizados

        # Calculamos el máximo y (*WIP* la media) de cada variable continua
        for i in range(len(self.input_data)):
            for j in range(len(normalize_pos)):
                aux = self.input_data[i]
                # *WIP* normalize_avg[j] = normalize_avg[j] + aux[normalize_pos[j]]
                if float(aux[normalize_pos[j]]) > normalize_max[j]:
                    normalize_max[j] = float(aux[normalize_pos[j]])
            
        # *WIP* Calculamos la media de cada variable a normalizar
        # for i in range(len(normalize_pos)):
            #normalize_avg[j] = normalize_avg[j] /  len(self.input_data)

        # Normalizamos cada variable continua. A efectos practicos convertimos cada variable continua en otra con valores [0, 1]
        for i in range(len(self.input_data)):
            for j in range(len(normalize_pos)):
                aux = self.input_data[i]
                aux[normalize_pos[j]] = round(float(normalize_pos[j]) / normalize_max[j], 4)
            res.append(aux)
                
        self.input_data = res


# 1: Obtención y procesamiento de datos
procesador = PreprocessorHepa("Preprocesamiento\hepatitis_csv.csv", 0.15)
(samples_data, samples_labels) = procesador.get_data()

# 2: Creación del modelo
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='tanh'),
    keras.layers.Dense(32, activation='tanh'),
    keras.layers.Dense(16, activation='tanh'),
    keras.layers.Dense(8, activation='tanh'),
    keras.layers.Dense(4, activation='tanh'),
    keras.layers.Dense(procesador.num_class, activation='softmax')
])

# 3: Función de optimización
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4: Entrenamiento         
model.fit(
    x = samples_data, 
    y = samples_labels, 
    epochs = 20,
    validation_split = procesador.test_percentage
    )

# 5: Hacer predicciones
predictions = model.predict(samples_data) # Devuelve la última capa, en este caso un vector con la posibilidad de pertenecer a cada clase
cont = 1
for pr in predictions:
    #print(("Muestra {} ha predicho {} y lo correcto es {}").format(cont, np.argmax(pr), int(samples_labels[cont - 1])))
    cont += 1