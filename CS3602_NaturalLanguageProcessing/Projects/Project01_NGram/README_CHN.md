# CS3602 Project 1: NGram平滑算法

> 实现了一个简单的 Katz-backoff 回退算法

## 一键运行

### 环境配置

#### 第三方库依赖

- `Python 3.8.5`
- `nltk 3.5`
- `kenlm`

#### 其他数据需求

- 可能需要使用[此仓库](https://github.com/mahavivo/english-wordlists) 的`COCA_20000` 词表

#### 参考文件结构

如果要按默认命令行参数运行，建议文件目录如下

- `hw1_dataset`
  - `train_set.txt`
  - `dev_set.txt`
  - `test_set.txt`
  - `COCA_20000.txt`
- `main.py`
- 其他 `Python` 代码

#### 如何运行？

`main.py` 包含了训练和测试语言模型的全部过程，在使用[参考文件结构](#参考文件结构)的前提下可以直接无参数运行

```cmd
python main.py
```

可以使用 `python main.py --help` 查看命令行参数，也可以参见下文[命令行参数](#命令行参数)

## 文件结构

- `main.py` 是完整的语言模型训练和测试流程
- `ngram.py` 包含了NGram语言模型（以及Katz回退算法）的实现
- `discount.py` 包含了GoodTuring回退算法的实现
- `dataloader.py` 包含了数据集文件的读取函数
- `utils.py` 包含了一些工具函数和常量

## 命令行参数

`main.py` 接受如下参数

- `--output` `-o` 指定输出的 `.arpa` 文件的路径和文件名.
  - Default `./lm.apra`
- `--path` `-p` 指定了数据集文件的路径.
  - 数据集中需要包括训练、开发、测试文本，以及COCA词表
  - Default `./hw1_dataset`
- `--vocabulary` `-v` 指定了使用的词表，词表将决定训练集中哪些单词被标注为 `<unk>`
  - Choices include `NLTK` and `COCA`
    - `NLTK`: 从 `nltk.corpus.words.words()` 获取的英文常见单词表（可能需要额外下载）
    - `COCA`: COCA 20000 词表
  - Default `COCA`
- `--tokenizer` `-t` 指定了分词方式
  - Choices include `NLTK` and `NAIVE`
    - `NLTK`: 使用 `nltk` 提供的分词功能（可能需要额外下载）
    - `NAIVE`: 使用空格分词
  - Default `NLTK`
