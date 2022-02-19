from arithmetic_coder import ArithmeticEncoder, ArithmeticDecoder, finishing
import numpy as np
from tqdm import tqdm
import collections


def equal_probs(alphabet_size):
    return np.array([1/alphabet_size]*alphabet_size, dtype=np.float64)


def fair_probs(data, alphabet_size):
    counter = collections.Counter(data)
    output = [0.0]*alphabet_size
    len_data = len(data)
    for item, amount in dict(counter).items():
        output[int(item)] = amount/len_data
    return np.array(output)


def test():
    # encode
    string = b"01320330034034040402430243023410234023402432013203300340340404024302430234102340234024320132033003403404040243024302341023402340243201320330034034040402430243023410234023402432013203300340340404024302430234102340234024320132033003403404040243024302341023402340243201320330034034040402430243023410234023402432013203300340340404024302430234102340234024320132033003403404040243024302341023402340243201320330034034040402430243023410234023402432013203300340340404024302430234102340234024320132033003403404040243024302341023402340243201320330034034040402430243023410234023402432013203300340340404024302430234102340234024320132033003403404040243024302341023402340243201320330034034040402430243023410234023402432013203300340340404024302430234102340234024320132033003403404040243024302341023402340243201320330034034040402430243023410234023402432013203300340340404024302430234102340234024320132033003403404040243024302341023402340243201320330034034040402430243023410234023402432013203300340340404024302430234102340234024320132033003403404040243024302341023402340243201320330034034040402430243023410234023402432013203300340340404024302430234102340234024320132033003403404040243024302341023402340243201320330034034040402430243023410234023402432013203300340340404024302430234102340234024320132033003403404040243024302341023402340243201320330034034040402430243023410234023402432013203300340340404024302430234102340234024320132033003403404040243024302341023402340243201320330034034040402430243023410234023402432013203300340340404024302430234102340234024320132033003403404040243024302341023402340243201320330034034040402430243023410234023402432013203300340340404024302430234102340234024320132033003403404040243024302341023402340243201320330034034040402430243023410234023402432013203300340340404024302430234102340234024320132033003403404040243024302341023402340243201320330034034040402430243023410234023402432013203300340340404024302430234102340234024320132033003403404040243024302341023402340243201320330034034040402430243023410234023402432013203300340340404024302430234102340234024320132033003403404040243024302341023402340243201320330034034040402430243023410234023402432013203300340340404024302430234102340234024320132033003403404040243024302341023402340243201320330034034040402430243023410234023402432013203300340340404024302430234102340234024320132033003403404040243024302341023402340243201320330034034040402430243023410234023402432013203300340340404024302430234102340234024320132033003403404040243024302341023402340243201320330034034040402430243023410234023402432013203300340340404024302430234102340234024320132033003403404040243024302341023402340243201320330034034040402430243023410234023402432013203300340340404024302430234102340234024320132033003403404040243024302341023402340243201320330034034040402430243023410234023402432013203300340340404024302430234102340234024320132033003403404040243024302341023402340243201320330034034040402430243023410234023402432013203300340340404024302430234102340234024320132033003403404040243024302341023402340243201320330034034040402430243023410234023402432013203300340340404024302430234102340234024320132033003403404040243024302341023402340243201320330034034040402430243023410234023402432013203300340340404024302430234102340234024320132033003403404040243024302341023402340243201320330034034040402430243023410234023402432013203300340340404024302430234102340234024320132033003403404040243024302341023402340243201320330034034040402430243023410234023402432013203300340340404024302430234102340234024320132033003403404040243024302341023402340243201320330034034040402430243023410234023402432013203300340340404024302430234102340234024320132033003403404040243024302341023402340243201320330034034040402430243023410234023402432013203300340340404024302430234102340234024320132033003403404040243024302341023402340243201320330034034040402430243023410234023402432013203300340340404024302430234102340234024320132033003403404040243024302341023402340243201320330034034040402430243023410234023402432013203300340340404024302430234102340234024320132033003403404040243024302341023402340243201320330034034040402430243023410234023402432013203300340340404024302430234102340234024320132033003403404040243024302341023402340243201320330034034040402430243023410234023402432013203300340340404024302430234102340234024320132033003403404040243024302341023402340243201320330034034040402430243023410234023402432013203300340340404024302430234102340234024320132033003403404040243024302341023402340243201320330034034040402430243023410234023402432013203300340340404024302430234102340234024320132033003403404040243024302341023402340243201320330034034040402430243023410234023402432013203300340340404024302430234102340234024320132033003403404040243024302341023402340243201320330034034040402430243023410234023402432013203300340340404024302430234102340234024320132033003403404040243024302341023402340243201320330034034040402430243023410234023402432"
    # probabilities will be estimated by a neural network
    probs = np.zeros(256, dtype=np.float64)
    for c in string:
        probs[c]+=1
    probs /= len(string)
    with finishing(ArithmeticEncoder("test.dz")) as enc:  # alternative: call enc.finish() when done encoding
        for c in string:
            enc.encode_symbol(probs, c)
    # decode
    newstring = []
    with finishing(ArithmeticDecoder("test.dz")) as dec:  # alternative: call dec.finish() when done decoding
        for _ in string:
            newstring.append(chr(dec.decode_symbol(probs)))
    print("".join(newstring))


def sliding_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


OFFSET = ord('0')


# default encode using a RNN model
def encode(file_name, m_size, alphabet_size, model, out_name="out.dz"):
    with open(file_name, 'r') as f:
        data = np.array([int(x) for x in f.readlines()[0]])
    eq_probs = equal_probs(alphabet_size)

    with finishing(ArithmeticEncoder(out_name)) as enc:
        # first model_size chars have equal probs
        for c in data[:m_size]:
            enc.encode_symbol(eq_probs, ord(str(c))-OFFSET)
        # others probs will be predicted by the model
        probs = model.predict(sliding_window(data[:-1], m_size), batch_size=32)
        for c, prob in zip(data[m_size:], probs):
            enc.encode_symbol(prob, c)

    return len(data)


# default decode using a RNN model
def decode(f_name, m_size, alphabet_size, model, len_data):
    output = []
    with finishing(ArithmeticDecoder(f_name)) as dec:

        # first model_size chars had equal probs
        eq_probs = equal_probs(alphabet_size)
        for _ in range(m_size):
            output.append(int(dec.decode_symbol(eq_probs)))

        # others probs were predicted by the model
        for _ in tqdm(range(m_size, len_data), total=len_data-m_size):
            prob = model.predict([output[-m_size:]]*32, batch_size=32)
            output.append(dec.decode_symbol(prob[0]))

    return output


# encode with equal prob for comparison
def encode_only_equal_probs(file_name, alphabet_size, out_name="out.dz"):
    with open(file_name, 'r') as f:
        data = np.array([int(x) for x in f.readlines()[0]])
    eq_probs = equal_probs(alphabet_size)

    with finishing(ArithmeticEncoder(out_name)) as enc:
        for c in data:
            enc.encode_symbol(eq_probs, ord(str(c))-OFFSET)
    return len(data)


# decode with equal prob for comparison
def decode_only_equal_probs(f_name, alphabet_size, len_data):
    output = []
    with finishing(ArithmeticDecoder(f_name)) as dec:
        eq_probs = equal_probs(alphabet_size)
        for _ in range(len_data):
            output.append(int(dec.decode_symbol(eq_probs)))
    return output


# encode with fair prob for comparison
def encode_only_fair_probs(file_name, alphabet_size, out_name="out.dz"):
    with open(file_name, 'r') as f:
        data = np.array([int(x) for x in f.readlines()[0]])
    f_probs = fair_probs(data, alphabet_size)

    with finishing(ArithmeticEncoder(out_name)) as enc:
        for c in data:
            enc.encode_symbol(f_probs, ord(str(c)) - OFFSET)
    return len(data), f_probs


# decode with fair prob for comparison
def decode_only_fair_probs(f_name, f_probs, len_data):
    output = []
    with finishing(ArithmeticDecoder(f_name)) as dec:
        for _ in range(len_data):
            output.append(int(dec.decode_symbol(f_probs)))
    return output


# encode with a model that predicts STEP_SIZE characters in each time_step
def encode_multi(file_name, m_size, alphabet_size, model, categories, step_size=4, out_name="out.dz"):
    with open(file_name, 'r') as f:
        data = np.array([int(x) for x in f.readlines()[0]])
    eq_probs = equal_probs(alphabet_size)
    offset = len(data[m_size:]) % step_size
    start = m_size + offset
    with finishing(ArithmeticEncoder(out_name)) as enc:
        # first model_size chars have equal probs
        for c in data[:start]:
            enc.encode_symbol(eq_probs, ord(str(c))-OFFSET)
        # others probs will be predicted by the model
        probs = model.predict(sliding_window(data[offset:-step_size], m_size)[::step_size], batch_size=1)
        for i, prob in enumerate(probs):
            d = int("".join(str(x) for x in data[start + i * step_size:start + (i + 1) * step_size]))
            enc.encode_symbol(prob, categories.index([d]))
    return len(data)


# decode with a model that predicts STEP_SIZE characters in each time_step
def decode_multi(f_name, m_size, alphabet_size, model, len_data, categories, step_size):
    output = []
    with finishing(ArithmeticDecoder(f_name)) as dec:

        start = m_size + (len_data-m_size) % step_size
        # first model_size chars had equal probs
        eq_probs = equal_probs(alphabet_size)
        for _ in range(start):
            output.append(int(dec.decode_symbol(eq_probs)))

        # others probs were predicted by the model
        for _ in range(start, len_data, step_size):
            prob = model.predict([output[-m_size:]], batch_size=1)
            index = dec.decode_symbol(prob[0])
            for x in str(categories[index][0]).zfill(step_size):
                output.append(int(x))

    return output


if __name__ == "__main__":
    test()