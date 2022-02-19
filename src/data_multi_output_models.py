import os
import generate_test_data as data_generator
import data_parser
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import itertools
from tqdm import tqdm
import tensorflow
from tensorflow import optimizers
from tensorflow import metrics
from models import LSTM_single, GRU_single
import tensorflow.keras.models as keras_models
import encode_decode as compressor


def create_data(file_names, size, step_size=8):
    return data_generator.test_data_generator(file_names, size, step_size=step_size)


def create_one_hot_categories(char_size, out_length):
    return [[int("".join(t))] for t in itertools.product("".join([str(i) for i in range(char_size)]), repeat=out_length)]



def train_models(file_list, out_size=4, epochs=2, out_dir='out_default', input_size=32):

    model_size = input_size
    output_path = os.path.join('..', 'output', out_dir)
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    alphabet_size = len(data_parser.LETTER_TO_INT)   # predetermined on 5

    print('Generating data...')
    X = create_data(file_list, input_size, out_size)

    print('Creating and OneHotEncoding y...')
    # make ground truth from X
    y = np.array(X[1:], dtype='<U1')[:, -out_size:]
    y = np.array([[int("".join(arr))] for arr in list(y)])
    # one hot encode y
    categories = create_one_hot_categories(alphabet_size, out_size)

    y = OneHotEncoder(sparse=False).fit(categories).transform(y)

    # last set has no successor
    X = X[:-1]

    ###
    # LSTM
    ###

    # SINGLE
    print(f'Creating LSTM_single model with inputsize {input_size} and lstmsize {model_size}...')
    # make a model
    model = LSTM_single(input_size, alphabet_size**out_size, out_dim=64, lstm_size=model_size)
    model.compile(optimizer=optimizers.Adam(), loss=tensorflow.keras.losses.BinaryCrossentropy(),
                  metrics=metrics.BinaryAccuracy())

    print('Fitting model...')
    # fit the model
    model.fit(x=X, y=y, epochs=epochs)

    model_name = f'LSTM_single_i{input_size}_l{model_size}'
    model_save_path = os.path.join(output_path, model_name)
    keras_models.save_model(model, model_save_path)
    print(model_name)
    return categories, model


# this trains a model that predict STEP_SIZE characters as output in each step
if __name__ == '__main__':
    step_size = 2
    file_name = 'small_test.txt'
    # file_name = 'NW_018654718.1'
    categ, model = train_models([file_name], epochs=2, out_dir='out_TEST_NW', out_size=step_size)
    # categ = create_one_hot_categories(5, step_size)
    # model = keras_models.load_model(os.path.join('..', 'output', 'out_TEST_NW', 'LSTM_single_i32_l32'))
    len_data = compressor.encode_multi(os.path.join('..', 'parsed_data', file_name), 32, 5, model, categ, step_size=step_size, out_name="TEST.dz")
    output = compressor.decode_multi('TEST.dz', 32, 5, model, len_data, categ, step_size)
    print("".join([str(x) for x in output]))

