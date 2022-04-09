# CS3602自然语言处理 Project 3 SLU

## 提交文件架构

- 在原基线文件结构的基础上，我们额外增加了若干文件
- `model`
  - `slu_ontology_guided_tagging.py` 是Project提出的模型
  - `slu_pinyin_tagging.py` 是Project早期阶段提出的一个不太成功的尝试。详见报告附录。
- `scripts`
  - `eval_test.py` 是在测试集上运行模型的测试脚本
  - `ontology_guided.py` 是Project提出的模型的测试脚本
  - `pinyin.py` 是增加了拼音Embedding的基线模型的测试脚本。详见报告附录。
- `utils`
  - `ontology.py` 是封装的 `Ontology` 类，用于给本项目实现的模型使用。
- 此外，我们提交了预训练的模型，保存在 `model.bin` 中。

## 环境配置

- 在原基线环境的基础上，需要额外安装 `pypinyin` 库

```sh
conda create -n slu python=3.6
source activate slu
pip install torch==1.7.1
pip install pypinyin
```

## 运行

我们为训练和测试分别提供了脚本。**训练脚本不支持在测试集上的测试功能，测试脚本也不支持在训练集上的训练功能。**

### 训练

使用下面的代码一键复现训练过程

```sh
python ./scripts/ontology_guided.py --pinyin --transcript
```

#### 训练脚本参数

- 我们保留了原基线支持的所有参数。此外，`ontology_guided.py` 还支持额外参数
  - `--model_save` 指定了模型参数保存的文件名，默认值是 `model.bin`
  - `--pinyin` 指定是否需要开启基于拼音的距离计算
  - `--transcript` 指定是否在训练时额外使用训练集中的 `manual_transcript`

### 测试

使用以下代码一键运行测试。

- 脚本运行时，要求 `<dataroot>` 目录下存在 `test_unlabelled.json`，且会把输出写入目录下的 `test.json`
- 测试脚本**默认使用基于拼音的距离对模型解码进行修正**。
- 测试脚本**只支持batch_size为1！否则输出结果的顺序将会不正确！**

```sh
python ./scripts/eval_test.py
```

#### 测试脚本参数

测试脚本支持以下参数。如无特殊说明，则参数功能与训练脚本一致。

- 通用设置
  - `--dataroot`
  - `--word2vec_path`
  - `--device` 默认为CPU，使用GPU需要手动指定 `--device 0` 等
  - `--batch_size` **测试脚本要求batch size为1以保证解码输出和输入的顺序的一致性**
  - `--model_save` 预训练的模型的保存路径，默认同样为 `model.bin`，脚本将从这里加载模型。
- 模型结构超参数
  - `--encoder_cell`
  - `--dropout`
  - `--embed_size`
  - `--hidden_size`
  - `--num_layer`
