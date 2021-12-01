# CS3602 Project 1: Smoothing Algorithm for N-Gram LM

> Implements a basic Katz-backoff algorithm for an N-Gram Language Model.

- [CHN](./README_CHN.md)

## Quick Start

### Environment

#### Libraries

- `Python 3.8.5`
- `nltk 3.5`
- `kenlm`

#### External Data

- Training the LM requires the `COCA_20000` vocabulary list from [this repository](https://github.com/mahavivo/english-wordlists).

#### Suggested File Structure

To run the code with default configuration, the file directory should look like

- `hw1_dataset`
  - `train_set.txt`
  - `dev_set.txt`
  - `test_set.txt`
  - `COCA_20000.txt`
- `main.py`
- other python scripts

#### How do I run?

The `main.py` includes the full process of training and evaluating an N-Gram language model.

```cmd
python main.py
```

For other commandline parameters, please use `python main.py --help` or check the [documentation](#commandline-args) below.

## Structure

- `main.py` is the training and evaluation process of the language model, which is implemented in `ngram.py`.
- `ngram.py` contains the implementation of the NGram language model.
- `discount.py` contains the implementation of Good-Turing discount method.
- `dataloader.py` includes functions for loading data from text files.
- `utils.py` contains various util functions and constants.
- `prototype.py` and `test.ipynb` are prototype and drafts for this project.

## Commandline Args

The `main.py` takes in the following arguments.

- `--output` `-o` specifies the path and name for the output `.arpa` language model.
  - Default `./lm.apra`
- `--path` `-p` specifies the path to the dataset.
  - The dataset should include train, dev and test texts and the COCA wordlist.
  - Default `./hw1_dataset`
- `--vocabulary` `-v` specifies which wordlist to use for determining `<unk>`s
  - Choices `NLTK` and `COCA`
    - `NLTK`: English wordlist from `nltk.corpus.words.words()`
    - `COCA`: COCA 20000 wordlist
  - Default `COCA`
- `--tokenizer` `-t` specifies the tokenization strategy
  - Choices include `NLTK` and `NAIVE`
    - `NLTK`: `nltk` tokenizer
    - `NAIVE`: Tokenization by splitting spaces
  - Default `NLTK`

**Note:** `nltk`-related parameters may requiring downloading extra data from `nltk`.
