import os
from Bio import SeqIO
import re


LETTER_TO_INT = {
    'N': 0,
    'A': 1,
    'C': 2,
    'G': 3,
    'T': 4,
}


def get_int_to_letter():
    return {value: key for key, value in LETTER_TO_INT.items()}


# PARSES THE DATA
def parse_input_fasta(file_name):
    # check if file exists
    if not os.path.exists(os.path.join('..', 'data', file_name)):
        print(f'FILE NOT FOUND: {file_name}')
        return 0
    input_path = os.path.join('..', 'data', file_name)

    # check if parsed dir exists if not make it
    parsed_dir = os.path.join('..', 'parsed_data')
    if not os.path.exists(parsed_dir):
        os.mkdir(parsed_dir)

    # check if raw dir exists if not make it
    raw_dir = os.path.join('..', 'raw')
    if not os.path.exists(raw_dir):
        os.mkdir(raw_dir)

    # check if parsed dir of input exists
    if not os.path.exists(os.path.join(parsed_dir, file_name)):
        os.mkdir(os.path.join(parsed_dir, file_name))

    count = 0
    for index, record in enumerate(SeqIO.parse(input_path, "fasta")):
        seq_id = record.id
        print(f'Parsing file {str(index+1)}: {seq_id}')
        seq_seq = str(record.seq).upper()

        # we will only use letters A C G T and N
        seq_seq = re.sub(r'[BDEFH-MO-SU-Z]', 'N', seq_seq)

        # write raw input to raw dir
        with open(os.path.join(raw_dir, seq_id), 'w') as f:
            f.write(seq_seq)

        print(f'Replacing letters in file {str(index + 1)}: {seq_id}')
        # now replace letters with int
        for key, value in LETTER_TO_INT.items():
            seq_seq = re.sub(key, str(value), seq_seq)

        # write parsed input to parsed dir
        with open(os.path.join(parsed_dir, seq_id), 'w') as f:
            f.write(seq_seq)

        print(f'Completed file {str(index + 1)}: {seq_id} \n')
        count += 1

    print(f'\nDone parsing, {str(count)} files were parsed.')
    return count


if __name__ == '__main__':
    # 639
    file = 'GRCh38_latest_genomic.fna'
    file_path = os.path.join('..', 'data', file)
    files_parsed = parse_input_fasta(file_path)

