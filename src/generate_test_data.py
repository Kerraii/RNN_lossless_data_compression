import os
from tqdm import tqdm
import tensorflow
from tensorflow import optimizers
from tensorflow import metrics
from sklearn.preprocessing import OneHotEncoder
from models import *
import numpy as np
import data_parser
import tensorflow.keras.models as keras_models


def float_to_percent(float_list, precision=3):
    return [np.format_float_positional(el*100, precision=precision) + '%' for el in float_list]


def test_data_generator(file_names, size, step_size=1):
    ints = []
    for file in file_names:
        file_path = os.path.join('..', 'parsed_data', file)
        with open(file_path, 'r') as f:
            ints += [int(number) for number in tqdm(list(f.readlines()[0]))]
    return np.array([ints[current_y-size:current_y] for current_y in tqdm(range(size, len(ints), step_size))])


def train_model(file_name, size, epochs=10):
    print('Generating data...')
    X = test_data_generator([file_name], size)

    alphabet_size = len(data_parser.LETTER_TO_INT)   # predetermined on 5

    print('Creating and OneHotEncoding y...')
    # make ground truth from X
    y = np.array(X[1:])[:, -1]
    y = y.reshape(len(y), 1)

    # one hot encode y
    y = OneHotEncoder(sparse=False).fit([[i] for i in range(alphabet_size)]).transform(y)

    # last set has no successor
    X = X[:-1]

    print('Creating model...')
    # make a model
    model = LSTM_single(size, alphabet_size, out_dim=64, lstm_size=32)
    model.compile(optimizer=optimizers.Adam(), loss=tensorflow.keras.losses.BinaryCrossentropy(),
                  metrics=metrics.BinaryAccuracy())

    print('Fitting model...')
    # fit the model
    # TODO other X/y
    model.fit(x=X, y=y, epochs=epochs)

    # test = np.array([X[0]])
    # print(test.shape)
    # print(test)
    # print(test.dtype)
    # output = model.predict(test)
    # print(output)

    return model


# this trains all models LSTM, GRU, simple_RNN with different stepsizes (32, 64, 128) and different layers (single,
# Bi_double). Double has been left out because we do not have enough computing power to train all models.
def train_all_models(file_list, epochs=10, out_dir='out_default'):
    model_names = []
    size_range = [32, 64, 128]
    output_path = os.path.join('..', 'output', out_dir)
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    alphabet_size = len(data_parser.LETTER_TO_INT)  # predetermined on 5

    for input_size in size_range:
        print('Generating data...')
        X = test_data_generator(file_list, input_size)

        print('Creating and OneHotEncoding y...')
        # make ground truth from X
        y = np.array(X[1:])[:, -1]
        y = y.reshape(len(y), 1)
        # one hot encode y
        y = OneHotEncoder(sparse=False).fit([[i] for i in range(alphabet_size)]).transform(y)

        # last set has no successor
        X = X[:-1]

        model_size = input_size
        ###
        # LSTM
        ###

        # SINGLE
        print(f'Creating LSTM_single model with inputsize {input_size} and lstmsize {model_size}...')
        # make a model
        model = LSTM_single(input_size, alphabet_size, out_dim=64, lstm_size=model_size)
        model.compile(optimizer=optimizers.Adam(), loss=tensorflow.keras.losses.BinaryCrossentropy(),
                      metrics=metrics.BinaryAccuracy())

        print('Fitting model...')
        # fit the model
        model.fit(x=X, y=y, epochs=epochs)

        model_name = f'LSTM_single_i{input_size}_l{model_size}'
        model_save_path = os.path.join(output_path, model_name)
        keras_models.save_model(model, model_save_path)
        model_names.append(model_name)
        print(model_name)

        # # DOUBLE
        # print(f'Creating LSTM_double model with inputsize {input_size} and lstmsize {model_size}...')
        # # make a model
        # model = LSTM_double(input_size, alphabet_size, out_dim=64, lstm_size=model_size)
        # model.compile(optimizer=optimizers.Adam(), loss=tensorflow.keras.losses.BinaryCrossentropy(),
        #               metrics=metrics.BinaryAccuracy())
        #
        # print('Fitting model...')
        # # fit the model
        # model.fit(x=X, y=y, epochs=epochs)
        #
        # model_name = f'LSTM_double_i{input_size}_l{model_size}'
        # model_save_path = os.path.join(output_path, model_name)
        # keras_models.save_model(model, model_save_path)
        # model_names.append(model_name)
        # print(model_name)

        # BI
        print(f'Creating LSTM_bi model with inputsize {input_size} and lstmsize {model_size}...')
        # make a model
        model = biLSTM(input_size, alphabet_size, out_dim=64, lstm_size=model_size)
        model.compile(optimizer=optimizers.Adam(), loss=tensorflow.keras.losses.BinaryCrossentropy(),
                      metrics=metrics.BinaryAccuracy())

        print('Fitting model...')
        # fit the model
        model.fit(x=X, y=y, epochs=epochs)

        model_name = f'LSTM_bi_i{input_size}_l{model_size}'
        model_save_path = os.path.join(output_path, model_name)
        keras_models.save_model(model, model_save_path)
        model_names.append(model_name)
        print(model_name)

        ###
        # simple_RNN
        ###
        # SINGLE
        print(f'Creating simpleRNN_single model with inputsize {input_size} and rnnsize {model_size}...')
        # make a model
        model = simpleRNN_single(input_size, alphabet_size, out_dim=64, rnn_size=model_size)
        model.compile(optimizer=optimizers.Adam(), loss=tensorflow.keras.losses.BinaryCrossentropy(),
                      metrics=metrics.BinaryAccuracy())

        print('Fitting model...')
        # fit the model
        model.fit(x=X, y=y, epochs=epochs)

        model_name = f'simpleRNN_single_i{input_size}_l{model_size}'
        model_save_path = os.path.join(output_path, model_name)
        keras_models.save_model(model, model_save_path)
        model_names.append(model_name)
        print(model_name)

        # # DOUBLE
        # print(f'Creating simpleRNN_double model with inputsize {input_size} and rnnsize {model_size}...')
        # # make a model
        # model = simpleRNN_double(input_size, alphabet_size, out_dim=64, rnn_size=model_size)
        # model.compile(optimizer=optimizers.Adam(), loss=tensorflow.keras.losses.BinaryCrossentropy(),
        #               metrics=metrics.BinaryAccuracy())
        #
        # print('Fitting model...')
        # # fit the model
        # model.fit(x=X, y=y, epochs=epochs)
        #
        # model_name = f'simpleRNN_double_i{input_size}_l{model_size}'
        # model_save_path = os.path.join(output_path, model_name)
        # keras_models.save_model(model, model_save_path)
        # model_names.append(model_name)
        # print(model_name)

        # BI
        print(f'Creating simpleRNN_bi model with inputsize {input_size} and rnnsize {model_size}...')
        # make a model
        model = bisimpleRNN(input_size, alphabet_size, out_dim=64, rnn_size=model_size)
        model.compile(optimizer=optimizers.Adam(), loss=tensorflow.keras.losses.BinaryCrossentropy(),
                      metrics=metrics.BinaryAccuracy())

        print('Fitting model...')
        # fit the model
        model.fit(x=X, y=y, epochs=epochs)

        model_name = f'simpleRNN_bi_i{input_size}_l{model_size}'
        model_save_path = os.path.join(output_path, model_name)
        keras_models.save_model(model, model_save_path)
        model_names.append(model_name)
        print(model_name)

        ###
        # GRU
        ###
        # SINGLE
        print(f'Creating GRU_single model with inputsize {input_size} and GRUsize {model_size}...')
        # make a model
        model = GRU_single(input_size, alphabet_size, out_dim=64, gru_size=model_size)
        model.compile(optimizer=optimizers.Adam(), loss=tensorflow.keras.losses.BinaryCrossentropy(),
                      metrics=metrics.BinaryAccuracy())

        print('Fitting model...')
        # fit the model
        model.fit(x=X, y=y, epochs=epochs)

        model_name = f'GRU_single_i{input_size}_l{model_size}'
        model_save_path = os.path.join(output_path, model_name)
        keras_models.save_model(model, model_save_path)
        model_names.append(model_name)
        print(model_name)

        # # DOUBLE
        # print(f'Creating GRU_double model with inputsize {input_size} and GRUsize {model_size}...')
        # # make a model
        # model = GRU_double(input_size, alphabet_size, out_dim=64, gru_size=model_size)
        # model.compile(optimizer=optimizers.Adam(), loss=tensorflow.keras.losses.BinaryCrossentropy(),
        #               metrics=metrics.BinaryAccuracy())
        #
        # print('Fitting model...')
        # # fit the model
        # model.fit(x=X, y=y, epochs=epochs)
        #
        # model_name = f'GRU_double_i{input_size}_l{model_size}'
        # model_save_path = os.path.join(output_path, model_name)
        # keras_models.save_model(model, model_save_path)
        # model_names.append(model_name)
        # print(model_name)

        # BI
        print(f'Creating GRU_bi model with inputsize {input_size} and GRUsize {model_size}...')
        # make a model
        model = biGRU(input_size, alphabet_size, out_dim=64, gru_size=model_size)
        model.compile(optimizer=optimizers.Adam(), loss=tensorflow.keras.losses.BinaryCrossentropy(),
                      metrics=metrics.BinaryAccuracy())

        print('Fitting model...')
        # fit the model
        model.fit(x=X, y=y, epochs=epochs)

        model_name = f'GRU_bi_i{input_size}_l{model_size}'
        model_save_path = os.path.join(output_path, model_name)
        keras_models.save_model(model, model_save_path)
        model_names.append(model_name)
        print(model_name)
