#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf

# ESTO SOLO VALE PARA UN INPUT DONDE EL ULTIMO ELEMENTO SEA LA CLASE A PREDECIR
# El fichero contiene una primera linea con el nombre de las clases
# Luego 10 datos con 4 variables cada uno separado por |

class Preprocessor:
    input_data = []                                         # Array bidimensional o lista de arrays que contendra los datos obtenidos del fichero de entrada
    num_class = -1                                          # Entero que indica el número de diferentes clases a predecir
    delimiter = '|'                                         # Elemento delimitador para separar cada dato del fichero de entrada
    test_percentage = 0.1                                   # Decimal que representa el porcentaje de muestras para evaluar el modelo [0.1 - 0.9]
    
    variables_name = [] # Atributo extra con el nombre de las variables, a eliminar

    def __init__(self, path : str, test_percentage : float):
        try:
            self.test_percentage = test_percentage          
            file = open(path)
            line = file.readline().split(',')
            variables_name = line
            while line:
                input_data.append(file.readline().split(delimiter))
        finally:
            file.close()

    # Separa los datos por clases y devuevle una lista de 4 arrays con los datos bien
    def get_data():
        data_by_class = []                                  # Array auxiliar que contendra los datos separados por clase a predecir
        target_by_class = []                                # Array auxiliar que contendra la clase a predecir
        for n in range(num_class):                          # Añadimos a los anteriores arrays tantos arrays como número de clases a predecir
            data_by_class.append([])                            # Añadimos un array vacio
            target_by_class.append([])                          # Añadimos un array vacio
        
        for aux in input_data:                              # Añadimos a los sub-arrays de los arrays anteriores las muestras separados de la clase a predecir
            data = aux[0:(aux.count - 1)]                       # Array auxiliar que contendra los datos de cada muestra
            target = aux[aux.count - 1]                         # Entero que indicara la clase a predecir
            data_by_class[target].append(data)                  # Añadimos los datos al sub-array correspondiente
            target_by_class[target].append(target)              # Añadimos la clase a predecir al sub-array correspondiente

        train_vectors = []                                  # Array auxiliar que contendra las muestras para entrenar el modelo
        train_labels = []                                   # Array auxiliar que contendra las clases a predecir para entrenar el modelo
        test_vectors = []                                   # Array auxiliar que contendra las muestras para evaluar el modelo
        test_labels = []                                    # Array axuiliar que contendra las clases a predecir para evaluar el modelo
        for m in range(num_class):
            # Tupla de cuatro elementos que contiene en cada elemento un tensor
            (train_vector, train_label, test_vector, test_label) = __convert_to_tensors(data_by_class[m], target_by_class[m])
            train_vectors.append(train_vector)                  # Añadimos la muestra de entrenamiento
            train_labels.append(train_label)                    # Añadimos la clase a predecir de entrenamiento
            test_vectors.append(test_vector)                    # Añadimos la muestra de evaluación
            test_labels.append(test_label)                      # Añadimos la clase a predecir de evaluación

        # Devolvemos una tupla de cuatro tensores donde cada tensor ha sido transformado en uno concatenando sus elementos en el eje 0
        return (tf.concat(train_vector, 0), tf.concat(train_labels, 0), tf.concat(test_vectors, 0), tf.concat(test_labels, 0))


    # Convierte la data en tensores y los separa
    def __convert_to_tensors(data, targets):
        num_samples = len(data)                                     # Entero que indica el número de muestras 
        num_test_samples = round(num_samples * test_percentage)     # Entero que indica el número de muestras para evaluar el modelo
        num_train_samples = num_samples - num_test_samples          # Entero que indica el número de muestras para entrenar el modelo
        sample_dimesion = len(data[0])                              # Entero que indica el número de variables por muestra

        t_data = tf.constant(                                       # Tensor de rango 2 para almacenar las muestras
            value = data, 
            shape = [num_samples, sample_dimesion]
            )  

        t_label = tf.one_hot(                                       # Tensor de rango 1 que transforma el número de la clase a predecir a un formato one-hot ({0, 1, 2} : 1 -> [0, 1, 0])
            indices = tf.constant(                                      # Tensor de rango 1 que contiene las clases a predecir
                value = targets,
                dtype = 'int32'                                         
                ),
            depth = num_class
            )

        train_vector = tf.slice(                                    # Tensor de rango 2 que contiene las muestras de entrenamiento
            input_ = t_data,
            begin = [0, 0],
            size = [num_train_samples, sample_dimesion]
        )

        test_vector = tf.slice(                                     # Tensor de rango 2 que contiene las muestras de evaluación
            input = t_data,
            begin = [num_train_samples, 0],
            size = [num_test_samples, sample_dimesion]
        ) 

        train_label = tf.slice(                                     # Tensor de rango 1 que contiene las clases a predecir de entrenamiento
            input_ = t_label,
            begin = [0, 0],
            size = [num_train_samples, num_class]
        )

        test_label = tf.slice(                                      # Tensor de rango 1 que contiene las clases a predecir de evaluación
            input = t_label,
            begin = [0, 0],
            size = [num_test_samples, num_class]
        )                       

        return (train_vector, train_label, test_vector, test_label)
    

