import tensorflow as tf
from tensorflow import keras

# *Preprocesador en desarrolor*
# Tipo de input:
#   
# Tipo de output:
#   - Dataset donde cada elemento es:
#       - features: 
#       - labels:
class CSVParse:
    train_data_path = '' # File path of training data
    test_data_path = '' # File path of test data
    separator = ',' # Character to split the CSV file
    predict_name = '' # Column of prediction (only for training data)

    def __init__(self, train_data : str, test_data : str, delimeter : str, prediction : str):
        self.train_data_path = train_data
        self.test_data_path = test_data
        self.separator = delimeter
        self.predict_name = prediction

    
    def __gen_dataset(file_path : str, **kwargs):
        dataset = tf.data.experimental.make_csv_dataset(
            file_pattern = file_path,
            batch_size = 4, # Tamaño de tandas a enviar n_tandas = n_muestras / batch_size
            label_name = predict_name,
            num_epochs = 1, # A revisar, puede que es lo que decia Alfons de enviar 
            field_delim = separator,
            **kwargs
        )
        return dataset

    def get_datasets():
        return (__gen_dataset(train_data_path), __gen_dataset(test_data_path))

    def show_batch(dataset):
        for batch, label in dataset.take(1):
            for key, value in batch.items():
                print("{:20s}: {}".format(key,value.numpy()))