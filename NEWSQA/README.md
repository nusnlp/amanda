# Experiments on NewsQA Dataset #

This directory contains the code and necessary files for experimentation on NewsQA dataset.

### Software Requirements ###

* Python 2.7
* Numpy
* Theano
* h5py
* nltk
* tqdm
* pandas
* Keras (with Theano backend)

### Data Requirements ###

Clone and follow the steps in the Requirements section of 
[NewsQA repository](https://github.com/Maluuba/newsqa).
Clone it with this commit id: eef43f75b298021b17a1a4812bee6fd2b546b89f

Then package the dataset.

Split the data based on their scripts:
```bash
cd newsqa
python maluuba/newsqa/split_dataset.py
```

Now, you should have a directory maluuba/newsqa/split_data which contains train.tsv, dev.tsv and test.tsv
We convert the csv file to json for ease of use.
Execute:

```bash
cd ..
mkdir data
python prep-data/data_prep_json.py newsqa/maluuba/newsqa/split_data/train.csv data/train-v1.1.json
python prep-data/data_prep_json.py newsqa/maluuba/newsqa/split_data/dev.csv data/dev-v1.1.json
python prep-data/data_prep_json.py newsqa/maluuba/newsqa/split_data/test.csv data/test-v1.1.json
```

### Data Preparation ###

For preparing the tokenized data and indexing, refer to the `src/run_dataprep.sh` file
Steps are self explanatory.
Execute:

```bash
./src/run_dataprep.sh data/train-v1.1.json data/dev-v1.1.json data/test-v1.1.json
```

### Training ###

You can see the training options by executing:

```bash
python src/train_newsqa.py -h
```

For training with the default configuration exwcute:

```bash
python src/train_newsqa.py
```

Alternatively, you can download our trained model from [here](https://tinyurl.com/ybdvpxcr/newsqa/newsqa_amanda.hdf5).


### Testing ###

Similarly, you can see the testing options.

```bash
python src/test_newsqa.py -h
```

Test with the default configuration:

```bash
python src/test_newsqa.py -w /path/to/the/model
```
