# 大作业2：Word2vec

## 环境需求

* Python版本大于等于3.6
* numpy
* scipy

## 任务

请补全`word2vec.py`中的`train_one_step`函数，并成功运行`python main.py`

main.py中包含3个测试点：
```
test1: 用于调试模型，如果实现正确，那么最终的loss将会停留在1.0左右，且'i','he','she'三者的相似性较高。
test2: 用实现的模型在data/treebank.txt上训练10个epoch。此部分最终的loss将会降至7.0左右，耗时约1.5h，请合理安排训练时间。
test3: 用test2训练的模型测试效果，如果spearman相关系数高于0.3且pearson相关系数高于0.4，则通过测试。
```

## 作业提交要求

1. PDF格式的报告中包含CBOW模型的前向传播和参数后向更新公式推导，包含关键代码实现和程序的重要输出结果。
2. 整个项目代码。如有其他特殊说明，请在报告中写明。
3. 请把报告和代码打包成zip文件上传。
