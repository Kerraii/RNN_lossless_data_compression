import data_parser
import os
from generate_test_data import train_all_models


# parses the data and trains different models
# these models can then be used to compress and decompress, the code for this can be executed in time_encode_decode.py
if __name__ == '__main__':
    # parsing has to be done only one time, make sure the dataset is in the data folder
    PARSE = True

    if PARSE:
        # name dataset
        file = 'GRCh38_latest_genomic.fna'
        # path to dataset
        file_path = os.path.join('..', 'data', file)
        # parse dataset
        _ = data_parser.parse_input_fasta(file_path)

    m_size = 64
    epochs = 10

    # Total = 5 random files
    files = [
        'NT_113948.1',      # 91 kB
        'NW_017852932.1',   # 260 kB
        'NT_187443.1',      # 90 kB
        'NW_018654712.1',   # 133 kB
        'NT_187582.1'       # 67 kB
    ]

    # train on one big file
    _ = train_all_models(['NW_018654718.1'], epochs=epochs, out_dir='out_NW_018654718_1')

    # train on multiple files (only a few models)
    _ = train_all_models(file_list=files, epochs=epochs, out_dir='out_5_files')

    # lower epochs for training on one file
    _ = train_all_models(['NW_018654718.1'], epochs=2, out_dir='out_NW_018654718_1_ep2')
    _ = train_all_models(['NW_018654718.1'], epochs=1, out_dir='out_NW_018654718_1_ep1')

