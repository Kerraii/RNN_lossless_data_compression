## Lossless data compression for genome sequences using recurrent neural networks

Project by Arnoud De Jonge and Ewout Vlaeminck

### Getting the Data

Make sure the data file is in the data folder. The data used can directly be downloaded from the following link: https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/annotation/GRCh38_latest/refseq_identifiers/GRCh38_latest_genomic.fna.gz

This is the Reference Genome Sequence FASTA (GRCh38) found on: https://www.ncbi.nlm.nih.gov/projects/genome/guide/human/index.shtml

Note that we do only use a few sequences from the data set that contains 639 genome sequences.

### Parsing the data and creating the models
This can be done by executing the main.py

### Using the models

this can be done in time_encode_decode.py

### Code files
arithmetic_code.py: contains the code for arithmetic encoding.

data_multi_output_models.py: contains the code for generating models that predict M characters in each timestep. 
Note that this is not complete, it still encodes all probabilities at once which will result in a memory error when using big input files.

data_parser.py: parses the data.

encode_decode.py: code that does the encoding/decoding in different ways.

generate_test_data.py: splits the data file into sequences and also has a train models function that trains all different models (with a single output character in each time step).

main.py: parses data and trains & saves different models.

models.py: the models used in the project.

time_encode_decode.py: code to time the encoding and decoding. This can also serve as an example on how to use the saved model.
