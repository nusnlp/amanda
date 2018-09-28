# Experiments on TriviaQA Dataset #

This directory contains the code and necessary files for experimentation on TriviaQA dataset.

### Software Requirements ###

* Python 2.7
* Python 3.5
* Numpy
* h5py
* nltk
* tqdm
* pandas
* Theano-0.9
* Keras (with Theano backend)-1.1.0
* cudnn-v5

### Data Requirements ###

Download the TriviaQA version 1.0 RC data from [here](http://nlp.cs.washington.edu/triviaqa/) and unzip it.
After unzipping you should have a `qa/` directory and an `evidence/` directory.

`utils/` and the evaluation script `src/triviaqa_evaluation.py` are borrowed from [official TriviaQA repository](https://github.com/mandarjoshi90/triviaqa).

### Data Preparation ###

Now, convert to the data for ease of use.

```bash
src/convert_data.sh qa/ squad-like-data/ evidence/wikipedia/ evidence/web/
```

Create a symbolic link of the `tokenization` and create the data directories.

```bash
ln -s ../SEARCHQA/tokenization/ ./
```
Then run the `src/run_dataprep.sh` for preparing all the data.

```bash
src/run_dataprep.sh squad-like-data/ data/
```

### Training ###

You can see the training options by executing:

```bash
python src/train_triviaqa.py -h
```

Alternatively, you can download our trained [wikimodel](https://tinyurl.com/ybdvpxcr/triviaqa/triviaqa_wiki_amanda.hdf5) and [webmodel](https://tinyurl.com/ybdvpxcr/triviaqa/triviaqa_web_amanda.hdf5).

### Testing ###

Similarly, you can see the testing options.

```bash
python src/test_triviaqa_web.py -h
```

