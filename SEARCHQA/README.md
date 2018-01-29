# Experiments on SearchQA Dataset #

This directory contains the code and necessary files for experimentation on SearchQA dataset.

### Software Requirements ###

* Python 2.7
* Python 3.5
* Numpy
* Theano
* h5py
* nltk
* tqdm
* pandas
* Keras (with Theano backend)

### Data Requirements ###

```bash
mkdir data
```

Clone the [SearchQA repository](https://github.com/nyu-dl/SearchQA). 
Follow the steps in the description to download the data.
Place the three files `{train,val,test}.txt` in `data/` directory. 

We convert the data to json for ease of use.
Execute:

```bash
python src/convert_data.py -trainf data/train.txt -valf data/val.txt -testf data/test.txt -out data/
```

### Data Preparation ###

For preparing the tokenized data and indexing, refer to the `src/run_dataprep.sh` file
Steps are self explanatory.
Execute:

```bash
./src/run_dataprep.sh data/train.json data/dev.json data/test.json
```

### Training ###

You can see the training options by executing:

```bash
python src/train_searchqa.py -h
```

For training with the default configuration exwcute:

```bash
python src/train_searchqa.py
```

Alternatively, you can download our trained model from [here](https://tinyurl.com/ybdvpxcr/searchqa/searchqa_amanda.hdf5).


### Testing ###

Similarly, you can see the testing options.

```bash
python src/test_searchqa.py -h
```

Test with the default configuration:

```bash
python src/test_searchqa.py -w path/of/the/model
```
