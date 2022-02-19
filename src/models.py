from keras.layers import Embedding, Dense, LSTM, Bidirectional, SimpleRNN, GRU
from tensorflow.keras.models import Sequential


# https://www.tensorflow.org/guide/keras/rnn
def LSTM_single(input_length, alphabet_size, out_dim=64, lstm_size=32):
    model = Sequential()
    model.add(Embedding(alphabet_size, out_dim, input_length=input_length))
    model.add(LSTM(units=lstm_size, recurrent_activation='sigmoid'))
    model.add(Dense(out_dim, activation='relu'))
    model.add(Dense(alphabet_size, activation='softmax'))
    return model


def LSTM_double(input_length, alphabet_size, out_dim=64, lstm_size=32):
    model = Sequential()
    model.add(Embedding(alphabet_size, out_dim, input_length=input_length))
    model.add(LSTM(units=lstm_size, recurrent_activation='sigmoid', return_sequences=True))
    model.add(LSTM(units=lstm_size, recurrent_activation='sigmoid'))
    model.add(Dense(out_dim, activation='relu'))
    model.add(Dense(alphabet_size, activation='softmax'))
    return model


def biLSTM(input_length, alphabet_size, out_dim=64, lstm_size=32):
    model = Sequential()
    model.add(Embedding(alphabet_size, out_dim, input_length=input_length))
    model.add(Bidirectional(LSTM(units=lstm_size, recurrent_activation='sigmoid', return_sequences=True)))
    model.add(Bidirectional(LSTM(units=lstm_size, recurrent_activation='sigmoid')))
    model.add(Dense(out_dim, activation='relu'))
    model.add(Dense(alphabet_size, activation='softmax'))
    return model


def simpleRNN_single(input_length, alphabet_size, out_dim=64, rnn_size=32):
    model = Sequential()
    model.add(Embedding(alphabet_size, out_dim, input_length=input_length))
    model.add(SimpleRNN(units=rnn_size))
    model.add(Dense(out_dim, activation='relu'))
    model.add(Dense(alphabet_size, activation='softmax'))
    return model


def simpleRNN_double(input_length, alphabet_size, out_dim=64, rnn_size=32):
    model = Sequential()
    model.add(Embedding(alphabet_size, out_dim, input_length=input_length))
    model.add(SimpleRNN(units=rnn_size, return_sequences=True))
    model.add(SimpleRNN(units=rnn_size))
    model.add(Dense(out_dim, activation='relu'))
    model.add(Dense(alphabet_size, activation='softmax'))
    return model


def bisimpleRNN(input_length, alphabet_size, out_dim=64, rnn_size=32):
    model = Sequential()
    model.add(Embedding(alphabet_size, out_dim, input_length=input_length))
    model.add(Bidirectional(SimpleRNN(units=rnn_size, return_sequences=True)))
    model.add(Bidirectional(SimpleRNN(units=rnn_size)))
    model.add(Dense(out_dim, activation='relu'))
    model.add(Dense(alphabet_size, activation='softmax'))
    return model


def GRU_single(input_length, alphabet_size, out_dim=64, gru_size=32):
    model = Sequential()
    model.add(Embedding(alphabet_size, out_dim, input_length=input_length))
    model.add(GRU(units=gru_size, recurrent_activation='sigmoid'))
    model.add(Dense(out_dim, activation='relu'))
    model.add(Dense(alphabet_size, activation='softmax'))
    return model


def GRU_double(input_length, alphabet_size, out_dim=64, gru_size=32):
    model = Sequential()
    model.add(Embedding(alphabet_size, out_dim, input_length=input_length))
    model.add(GRU(units=gru_size, recurrent_activation='sigmoid', return_sequences=True))
    model.add(GRU(units=gru_size, recurrent_activation='sigmoid'))
    model.add(Dense(out_dim, activation='relu'))
    model.add(Dense(alphabet_size, activation='softmax'))
    return model


def biGRU(input_length, alphabet_size, out_dim=64, gru_size=32):
    model = Sequential()
    model.add(Embedding(alphabet_size, out_dim, input_length=input_length))
    model.add(Bidirectional(GRU(units=gru_size, recurrent_activation='sigmoid', return_sequences=True)))
    model.add(Bidirectional(GRU(units=gru_size, recurrent_activation='sigmoid')))
    model.add(Dense(out_dim, activation='relu'))
    model.add(Dense(alphabet_size, activation='softmax'))
    return model

