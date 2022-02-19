import tensorflow.keras.models as keras_models
from encode_decode import encode, decode, encode_only_fair_probs, encode_only_equal_probs, decode_only_fair_probs, decode_only_equal_probs
import os
import re
from tqdm import tqdm
import numpy as np


def encode_using_models(file_name, model_dir):
    file_path = os.path.join('..', 'parsed_data', file_name)

    out_dir = os.path.join('..', 'output', model_dir)
    model_paths = [f.path for f in os.scandir(out_dir) if f.is_dir()]

    for model_save_path in tqdm(model_paths):
        model = keras_models.load_model(model_save_path)
        print(f'ENCODING WITH {model_save_path}')
        m_size = int(re.search('i[0-9]+', model_save_path).group()[1:])
        _ = encode(file_path, m_size, alphabet_size=5, model=model,
                   out_name=os.path.join(f'{model_save_path}.dz'))


def decode_using_model(model_save_path, file_path_compressed, file_path_orig):
    with open(file_path_orig, 'r') as f:
        data_length = len(np.array([int(x) for x in f.readlines()[0]]))

    for _ in tqdm(range(1)):
        model = keras_models.load_model(model_save_path)
        print(f'ENCODING WITH {model_save_path}')
        m_size = int(re.search('i[0-9]+', model_save_path).group()[1:])
        _ = decode(file_path_compressed, m_size, 5, model, data_length)


if __name__ == '__main__':
    TIME_ENCODE = True
    TIME_DECODE = False
    base_path_model = os.path.join('..', 'output', 'out_NW_018654718_1')
    original_file = os.path.join('..', 'parsed_data', "NW_018654718.1")

    if TIME_ENCODE:
        # encode_using_models("NW_018654718.1", 'out_NW_018654718_1__')
        # encode_using_models("NW_018654718.1", 'out_NW_018654718_1_ep2')
        # encode_using_models("NW_018654718.1", 'out_NW_018654718_1_ep1')
        encode_using_models("NW_018654718.1", 'out_5_files')

    # some timing test for decoding
    # default false because it takes a VERY LONG time
    if TIME_DECODE:
        # +- 12 hours
        decode_using_model(os.path.join(base_path_model, 'GRU_single_i32_l32'),
                           os.path.join(base_path_model, 'GRU_single_i32_l32.dz'),
                           original_file)
        # +- 13 hours
        decode_using_model(os.path.join(base_path_model, 'GRU_single_i64_l64'),
                           os.path.join(base_path_model, 'GRU_single_i64_l64.dz'),
                           original_file)
        # +- 20 hours
        decode_using_model(os.path.join(base_path_model, 'GRU_single_i128_l128'),
                           os.path.join(base_path_model, 'GRU_single_i128_l128.dz'),
                           original_file)
        # +- 12 hours
        decode_using_model(os.path.join(base_path_model, 'LSTM_single_i32_l32'),
                           os.path.join(base_path_model, 'LSTM_single_i32_l32.dz'),
                           original_file)
        # +- 13 hours
        decode_using_model(os.path.join(base_path_model, 'LSTM_single_i64_l64'),
                           os.path.join(base_path_model, 'LSTM_single_i64_l64.dz'),
                           original_file)
        # +- 18 hours
        decode_using_model(os.path.join(base_path_model, 'LSTM_single_i128_l128'),
                           os.path.join(base_path_model, 'LSTM_single_i128_l128.dz'),
                           original_file)
        # +- 12 hours
        decode_using_model(os.path.join(base_path_model, 'simpleRNN_single_i32_l32'),
                           os.path.join(base_path_model, 'simpleRNN_single_i32_l32.dz'),
                           original_file)
        # +- 14 hours
        decode_using_model(os.path.join(base_path_model, 'simpleRNN_single_i64_l64'),
                           os.path.join(base_path_model, 'simpleRNN_single_i64_l64.dz'),
                           original_file)
        # +- 20 hours
        decode_using_model(os.path.join(base_path_model, 'simpleRNN_single_i128_l128'),
                           os.path.join(base_path_model, 'simpleRNN_single_i128_l128.dz'),
                           original_file)

    # range 1 is simples timing
    for _ in tqdm(range(1)):
        len_data = encode_only_equal_probs(original_file, 5, os.path.join('..', 'output', 'out_NW_018654718_1', 'equal_probs.dz'))

    for _ in tqdm(range(1)):
        decode_only_equal_probs(os.path.join('..', 'output', 'out_NW_018654718_1', 'equal_probs.dz'), 5, len_data)

    for _ in tqdm(range(1)):
        len_data, f_probs = encode_only_fair_probs(original_file, 5, os.path.join('..', 'output', 'out_NW_018654718_1', 'fair_probs.dz'))

    for _ in tqdm(range(1)):
        decode_only_fair_probs(os.path.join('..', 'output', 'out_NW_018654718_1', 'fair_probs.dz'), f_probs, len_data)

