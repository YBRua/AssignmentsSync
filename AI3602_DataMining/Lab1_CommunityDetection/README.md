# AI3602 Data Mining Assignment 1

> Implementation of Louvain Algorithm for graph clustering

## Requirements

- `Python 3.8`
- `numpy 1.21.2`
- `scipy 1.5.2`
- `pandas 1.1.3`

## How to Run

### Quick Start

The program should be executed at folder `<StudentNumber>_EXP1/.`. One can run the algorithm by executing.

```shell
studentnumber_EXP1/ > python ./src/main.py
```

The output will be stored in `./src/res.csv` by default. With the random seed `HK416A5`, it should achieve approximately 0.85 acc on the given ground truths.

Note that this implementation randomly shuffles the visiting order of nodes. The result may vary greatly depending on the shuffling result. The shuffling usually produces fair results but it does crash sometimes (acc ranging from 0.6 to 0.8).

**NOTE**: `src/main.py` loads datasets and stores results by *relative paths*. Please make sure **the root directory when executing the code is `<StudentNumber>_EXP1/.`**, or otherwise the relative paths would fail ~~and everything explodes~~. If one wants to change the paths of input dataset and output results, please check the [commandline arguments](#commandline-arguments) below.

## Commandline Arguments

Currently `src/main.py` takes two optional arguments to redirect input and output paths

- `--dataset_path` or `-d` is the path to the dataset csv file (which should be `edges_update.csv`)
  - default value is `./data/edges_update.csv`
- `--output` or `-o` is the path of the output csv file (where `res.csv` will be dumped to)
  - default value is `./src/res.csv`
