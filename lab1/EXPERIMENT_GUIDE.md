# 自然语言处理实验一：实验指导

## 1. 实验目标

本实验比较三类文本表示方法在 NYT 三分类新闻数据上的效果：

1. Bag-of-Words：二值词袋、词频词袋；
2. Word Embedding：预训练 GloVe、AG News Word2Vec、NYT Word2Vec；
3. 预训练语言模型：`google-bert/bert-base-uncased` 微调。

所有分类实验统一在同一个 NYT Test Set 上报告 Accuracy 和 Macro-F1。重点不是追求最高分，而是保证方法实现正确、比较条件公平，并解释不同表示为什么会有不同表现。

## 2. 实验要求核对

- Task 1 原本写有“三种 Bag-of-Words”，但其后实际只列出了 Binary Bag-of-Words 和 Word Frequency。根据老师补充说明，这里按两种方法执行。
- Task 1 和 Task 2 的 Logistic Regression 使用相同基础配置；Task 2 的词向量维度严格为 100。
- Task 3 使用 `bert-base-uncased`，`max_length=64`，训练 3 个 epoch。
- 最终测试集只用于最终评价，不用测试集反复选择模型。
- 提交内容包括代码、README、运行脚本和 PDF 实验报告。

## 3. 数据与划分

### 3.1 数据

- `nyt.csv`：`text,label`，共 11,519 条，三个类别。
- `ag.csv`：只有 `text`，共 90,000 条，仅用于训练 Word2Vec。

NYT 类别分布明显不均衡：sports 约占 75%，politics 和 business 各约占 12.5%。因此 Accuracy 和 Macro-F1 必须同时报告，并重点分析少数类。

### 3.2 划分

采用简单、可复现的分层随机划分：

- Training Set：80%；
- Validation Set：10%；
- Test Set：10%；
- 随机种子：42；
- 两次调用 `train_test_split`：先分出 20% 的临时集，再将临时集对半划分为 validation 和 test。

分层的作用是保证三个集合中的类别比例接近，避免少数类在测试集里过少。所有任务共用同一个 `split_data()` 函数和同一个随机种子。

## 4. 统一预处理

BoW 和 Word2Vec 使用同一套分词规则：

1. 转为小写；
2. 使用 NLTK `TreebankWordTokenizer`；
3. 删除不含字母或数字的纯标点 token；
4. 保留数字和包含字母数字的混合 token；
5. 空文档保留为空 token 列表，并在实验日志中统计。

BERT 使用其自身的 WordPiece Tokenizer，因为 Transformer 模型必须使用与预训练模型匹配的分词器。这里不能强行复用 BoW 分词结果。

关键词表和统计信息只能从 Training Set 获得。NYT Word2Vec 仅在 Training Set 的文本上训练，防止测试文本参与表示学习。

## 5. 通用分类设置

Task 1 和 Task 2 使用：

```text
LogisticRegression(
    C=1.0,
    max_iter=2000,
    solver="lbfgs",
    random_state=42
)
```

六个实验使用同一份 NYT 划分。每个实验输出：

- Test Accuracy；
- Test Macro-F1；
- 每类 Precision、Recall、F1；
- 混淆矩阵；
- 部分错分样本；
- 运行时间和关键中间统计量。

## 6. 逐实验执行顺序

### Experiment 1：Task 1 Bag-of-Words

1. 读取 NYT；
2. 构建固定划分；
3. 在 Training Set 上统计词表；
4. 生成 Binary BoW 和 Word Frequency 矩阵；
5. 分别训练 Logistic Regression；
6. 保存指标、混淆矩阵、分类报告和错分案例；
7. 在项目 README 中记录本实验背景、配置、结果和分析。

### Experiment 2：Task 2 Word2Vec

1. 完成 GloVe 的 100 维词向量加载和平均池化；
2. 使用 AG News 训练 100 维 Word2Vec；
3. 使用 NYT Training Set 训练 100 维 Word2Vec；
4. 三种表示分别训练 Logistic Regression；
5. 记录 OOV 比例、有效词数和分类结果；
6. 在 README 中追加本实验配置、结果和分析。

### Experiment 3：Task 3 BERT

1. 加载 `bert-base-uncased` 和对应 Tokenizer；
2. 使用 `max_length=64`；
3. 在 Training Set 上微调 3 个 epoch；
4. 在 Validation Set 上监控效果；
5. 在 Test Set 上计算 Accuracy、Macro-F1 和类别指标；
6. 在 README 中追加 BERT 配置、训练曲线、结果和分析。

## 7. 分析与报告要求

最终报告以比较分析为主体：

1. 六种方法的 Accuracy 和 Macro-F1 总表；
2. Accuracy 与 Macro-F1 差异，说明类别不平衡的影响；
3. 三种 Word Embedding 的来源差异和 OOV 情况；
4. BERT 相对于静态表示的提升及可能原因；
5. 每个方法的混淆矩阵与典型错误案例；
6. `max_length=64` 的截断统计和潜在影响；
7. 当前方法的局限和进一步改进方向。

## 8. 复现原则

- 固定随机种子；
- 记录运行环境、库版本和运行命令；
- 保存中间统计和最终结果，不手工修改指标；
- 使用同一个数据划分做公平比较；
- 先分别完成任务，最后再统一整理格式和汇总 PDF 报告。
