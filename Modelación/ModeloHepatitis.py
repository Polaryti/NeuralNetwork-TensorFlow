# TensorFlow and tf.keras
import random
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

class PreprocessorHepa:
    input_data = []                                         # Array bidimensional o lista de arrays que contendra los datos obtenidos del fichero de entrada
    num_class = 2                                           # Entero que indica el número de diferentes clases a predecir
    delimiter = ','                                         # Elemento delimitador para separar cada dato del fichero de entrada
    test_percentage = 0.1                                   # Decimal que representa el porcentaje de muestras para evaluar el modelo [0.1 - 0.9]
    
    variables_name = [] # Atributo extra con el nombre de las variables, a eliminar
    ar_dict = {
        "False": 0.0,
        "True": 1.0,
        "male": 0.0,
        "female": 1.0,
        "live\n": 0.0,
        "die\n": 1.0
    }

    def __init__(self, path : str, test_percentage : float):
        if test_percentage < 0.1 or test_percentage > 0.9:
            raise ValueError
        self.test_percentage = test_percentage         
        with open(path, 'r') as file_r:
            self.variables_name = file_r.readline().split(',')
            self.input_data = list(map(lambda x: x.split(self.delimiter), file_r.readlines()))
            random.shuffle(self.input_data)
            input_aux = []
            for aux in self.input_data:
                sample = self.__map_data(aux)
                if sample != []:
                    input_aux.append(sample)
            self.input_data = input_aux
            self.__normalize_data()

                

    # Separa los datos por clases y devuevle una lista de 4 arrays con los datos bien
    def get_data(self):
        data_by_class = []                                  # Array auxiliar que contendra los datos separados por clase a predecir
        target_by_class = []                                # Array auxiliar que contendra la clase a predecir
        for n in range(self.num_class):                          # Añadimos a los anteriores arrays tantos arrays como número de clases a predecir
            data_by_class.append([])                            # Añadimos un array vacio
            target_by_class.append([])                          # Añadimos un array vacio
        
        for aux in self.input_data:                              # Añadimos a los sub-arrays de los arrays anteriores las muestras separados de la clase a predecir
            data = aux[0:len(aux) - 1]                       # Array auxiliar que contendra los datos de cada muestra
            target = aux[len(aux) - 1]                         # Entero que indicara la clase a predecir
            data_by_class[int(target)].append(data)                  # Añadimos los datos al sub-array correspondiente
            target_by_class[int(target)].append(target)              # Añadimos la clase a predecir al sub-array correspondiente

        train_vectors = []                                  # Array auxiliar que contendra las muestras para entrenar el modelo
        train_labels = []                                   # Array auxiliar que contendra las clases a predecir para entrenar el modelo
        test_vectors = []                                   # Array auxiliar que contendra las muestras para evaluar el modelo
        test_labels = []                                    # Array axuiliar que contendra las clases a predecir para evaluar el modelo
        for m in range(self.num_class):
            # Tupla de cuatro elementos que contiene en cada elemento un tensor
            (train_vector, train_label, test_vector, test_label) = self.__convert_to_tensors(data_by_class[m], target_by_class[m])
            train_vectors.append(train_vector)                  # Añadimos la muestra de entrenamiento
            train_labels.append(train_label)                    # Añadimos la clase a predecir de entrenamiento
            test_vectors.append(test_vector)                    # Añadimos la muestra de evaluación
            test_labels.append(test_label)                      # Añadimos la clase a predecir de evaluación

        # Devolvemos una tupla de cuatro tensores donde cada tensor ha sido transformado en uno concatenando sus elementos en el eje 0
        return (tf.concat(train_vectors, 0), tf.concat(train_labels, 0), tf.concat(test_vectors, 0), tf.concat(test_labels, 0))


    # Convierte la data en tensores y los separa
    def __convert_to_tensors(self, data, targets):
        num_samples = len(data)                                     # Entero que indica el número de muestras 
        num_test_samples = round(num_samples * self.test_percentage)     # Entero que indica el número de muestras para evaluar el modelo
        num_train_samples = num_samples - num_test_samples          # Entero que indica el número de muestras para entrenar el modelo
        sample_dimesion = len(data[0])                              # Entero que indica el número de variables por muestra

        t_data = tf.constant(                                       # Tensor de rango 2 para almacenar las muestras
            value = data, 
            shape = [num_samples, sample_dimesion]
            )  

        # t_label = tf.one_hot(                                       # Tensor de rango 1 que transforma el número de la clase a predecir a un formato one-hot ({0, 1, 2} : 1 -> [0, 1, 0])
        #     indices = tf.constant(                                      # Tensor de rango 1 que contiene las clases a predecir
        #         value = targets
        #         #dtype = 'int32'                                         
        #         ),
        #     depth = self.num_class
        #     )
        
        t_label = tf.constant(
            value = targets,
        )


        train_vector = tf.slice(                                    # Tensor de rango 2 que contiene las muestras de entrenamiento
            input_ = t_data,
            begin = [0, 0],
            size = [num_train_samples, sample_dimesion]
        )

        test_vector = tf.slice(                                     # Tensor de rango 2 que contiene las muestras de evaluación
            input_ = t_data,
            begin = [num_train_samples, 0],
            size = [num_test_samples, sample_dimesion]
        ) 

        train_label = tf.constant(
            value = targets[0:num_train_samples],
            shape = (num_train_samples,)
        )

        test_label = tf.constant(
            value = targets[num_train_samples:num_samples],
            shape = (num_test_samples,)
        )

        # train_label = tf.slice(                                     # Tensor de rango 1 que contiene las clases a predecir de entrenamiento
        #     input_ = t_label,
        #     begin = [0, 0],
        #     size = (num_train_samples,)
        # )

        # test_label = tf.slice(                                      # Tensor de rango 1 que contiene las clases a predecir de evaluación
        #     input_ = t_label,
        #     begin = [0, 0],
        #     size = (num_test_samples,)
        # )                       

        return (train_vector, train_label, test_vector, test_label)
    
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

    def __normalize_data(self):
        res = []

        max_age = -1
        avg_age = 0
        #bilirubin,alk_phosphate,sgot,albumin,protime
        max_bil = -1
        avg_bil = 0
        max_alk = -1
        avg_alk = 0
        max_sgot = -1
        avg_sgot = 0
        max_alb = -1
        avg_alb = 0
        max_pro = -1
        avg_pro = 0

        for aux in self.input_data:
            avg_age = avg_age + float(aux[0])
            if float(aux[0]) > max_age:
                max_age = float(aux[0])
            avg_bil = avg_bil + float(aux[13])
            if float(aux[13]) > max_bil:
                max_bil = float(aux[13])
            avg_alk = avg_alk + float(aux[14])
            if float(aux[14]) > max_alk:
                max_alk = float(aux[14])
            avg_sgot = avg_sgot + float(aux[15])
            if float(aux[15]) > max_sgot:
                max_sgot = float(aux[15])
            avg_alb = avg_alb + float(aux[16])
            if float(aux[16]) > max_alb:
                max_alb = float(aux[16])
            avg_pro = avg_pro + float(aux[17])
            if float(aux[17]) > max_pro:
                max_pro = float(aux[17])

        for aux in self.input_data:
            aux[0] = round(float(aux[0]) / max_age, 4)
            aux[13] = round(float(aux[13]) / max_bil, 4)
            aux[14] = round(float(aux[14]) / max_alk, 4)
            aux[15] = round(float(aux[15]) / max_sgot, 4)
            aux[16] = round(float(aux[16]) / max_alb, 4)
            aux[17] = round(float(aux[17]) / max_pro, 4)
            res.append(aux)

        self.input_data = res

# 1: Obtención y procesamiento de datos
procesador = PreprocessorHepa("Preprocesamiento\hepatitis_csv.csv", 0.1)
(train_data, train_label, test_data, test_label) = procesador.get_data()

# 2: Creación del modelo
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(procesador.num_class, activation='softmax')
])

# 3: Función de optimización
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4: Entrenamiento         
model.fit(
    x = train_data, 
    y = train_label, 
    epochs = 20,
    validation_split = 0.1
    )

# 5: Evaluación del modelo
test_loss, test_acc = model.evaluate(test_data,  test_label, verbose=2)
print('\nTest accuracy:', test_acc)

# 6: Hacer predicciones
predictions = model.predict(test_data) # Devuelve la última capa, en este caso un vector con la posibilidad de pertenecer a cada clase
cont = 1
for pr in predictions:
    print(pr)
    pr = [round(pr[0]), round(pr[1])]
    pred = -1
    if pr[0] > pr[1]:
        pred = 0
    else:
        pred = 1
    print(("Muestra {} ha predicho {} y lo correcto es {}").format(cont, pred, test_label[cont - 1]))
    cont += 1
