# AI2651 Project 1 Task 1

## English
- using short-time features of speech signals
- using linear/threshold classifiers

### How to Run
- Setup
  - Root directory should be `./2651Project1_VAD`, instead of here.
  - Please make sure that datasets `data` and `wavs` are under root directory. (Or please change the paths manually in the code)
  - Please make sure that `time_domain_features.csv` and `freq_domain_features.csv` are available under root directory. If they do not exist, please run [`short_time_analysis.py`](./short_time_analysis.py) or [`run_on_dev_set.py`](./run_on_dev_set.py)). These files are essential for initializing the threshold classifier.
- Files
  - [`run_on_dev_set.py`](./run_on_dev_set.py) includes the process of feature extraction, data analysis and classifier validation.
  - [`run_on_test_set.py`](./run_on_test_set.py) runs the classifier on test set and outputs the labels.
  - [`parameter_tune.py`](./parameter_tune.py) includes the process of stochastic parameter tuning.

## 中文
- 语音信号短时特征
- 简单线性/阈值分类器

[GITHUB](https://github.com/YBRua/2651SpeechRecognition/tree/main/P1_VoiceActivityDetection)
### 如何运行
- 在开始之前：
  - 请保证本项目中的所有文件相对位置正确
  - 请确保data和wavs两个文件夹在当前目录下（或者在代码中重新配置路径）
  - 请确保当前目录下有 `time_domain_features.csv` 和 `freq_domain_features.csv` 两个csv文件（如果没有，请先运行[`short_time_analysis.py`](./short_time_analysis.py) 或 [`run_on_dev_set.py`](./run_on_dev_set.py))。这是因为分类器的阈值并没有硬编码在代码中，而是需要根据上述两个csv文件确定阈值。
- 要验证特征提取流程和模型在开发集上的表现，请运行[`run_on_dev_set.py`](./run_on_dev_set.py)。该文件包括了特征提取、数据分析、构建分类器以及在开发集上验证分类器的全部流程。
  - 该程序会进行两次循环
  - 第一次循环提取特征
  - 第二次循环在开发集上验证模型
  - ~~所以看到tqdm跑了两次不是bug，是特性~~
- 要使用分类器对测试集上的语音文件进行分类，请运行[`run_on_test_set.py`](./run_on_test_set.py)。
- 要复现调参过程，请运行[`parameter_tune.py`](./parameter_tune.py)。由于实现时疏忽，忘记设定固定的随机数种子，每次运行的结果可能有差异。另外请注意完整运行这部分代码需要很长时间。

### 其他文件
- classifiers文件夹下存放了分类器的实现代码。目前只有[阈值分类器](./classifiers/basic.py)。
- [`short_time_features.py`](./short_time_features.py)存放了提取各项特征的函数。
- [`short_time_analysis.py`](./short_time_analysis.py)提取了开发集上500条音频文件的时域和频域特征，并统计了Voiced和Unvoiced两类帧的特征。分析的结果会被导出到两个.csv文件中，用于构建分类器。
- [`classification.py`](./classification.py)保存了分类任务需要使用的一些工具函数。
- [`parameter_tune.py`](./parameter_tune.py)是报告中随机参数调整的代码。由于实现时疏忽，忘记设定固定的随机数种子，每次运行的结果可能有差异。另外请注意完整运行这部分代码需要很长时间。

### 分类器权重
#### 默认权重
```python
ScoreWeight(
    mag=2,
    enr=2,
    zcr=0,
    low=1,
    med=4,
    high=1,
    state_weight=2,
    primary_passes=5,
    secondary_passes=6
)
```
#### 调参后得到的较优权重
```python
ScoreWeight(
    mag=2.3660515227,
    enr=2.20055434,
    zcr=0.4418205904,
    low=0.3016455199,
    med=4.3315168325,
    high=1.7074504699,
    state_weight=3.4955110421,
    primary_passes=5.6886246655,
    secondary_passes=6.6389220191
)
```