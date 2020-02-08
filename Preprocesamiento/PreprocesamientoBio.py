import tensorflow as tf
import numpy as np

class ProcesadorBio:
    input_data = []                     # Array bidimensional o lista de arrays que contendra los datos obtenidos del fichero de entrada
    num_class = 2                       # Entero que indica el número de diferentes clases a predecir
    delimiter = ','                     # Elemento delimitador para separar cada dato del fichero de entrada
    test_percentage = 0.1               # Decimal que representa el porcentaje de muestras para evaluar el modelo [0.1 - 0.9]
    
    variables_name = []                 # *WIP* Array que contiene los nombres de las variables

    def __init__(self, path : str, test_percentage : float):
        self.test_percentage = test_percentage         
        with open(path, 'r') as file_r:
            self.variables_name = file_r.readline().split(',')
            self.input_data = list(map(lambda x: x.split(self.delimiter), file_r.readlines()))
            for i in range(len(self.input_data)):
                self.input_data[i] = list(map(float, self.input_data[i]))

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
