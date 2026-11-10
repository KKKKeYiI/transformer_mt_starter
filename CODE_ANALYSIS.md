# minimal_transformer_mt.py 详细代码解析

这份解析对应恢复后的字符级版本。整个项目目前确实把数据读取、词表、数据集、模型、训练、推理和 BLEU 评价都放在一个文件里。

## 整体结构

文件按顺序包含：

1. 导入库
2. 设置控制台输出编码
3. 特殊符号常量
4. 读取 TSV 数据
5. 字符级词表
6. 数据集类
7. batch 补齐函数
8. 位置编码
9. Transformer 模型
10. 逐字翻译函数
11. 批量生成函数
12. BLEU 评价函数
13. DataLoader 构建函数
14. 训练函数
15. 命令行参数

## 1. 导入库

```python
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from sacrebleu.metrics import BLEU
```

- `torch`：PyTorch 主库。
- `nn`：神经网络模块，例如 `Embedding`、`Transformer`、`Linear`、`CrossEntropyLoss`。
- `pad_sequence`：把不同长度的序列补齐。
- `DataLoader` 和 `Dataset`：批量读取训练数据。
- `BLEU`：sacreBLEU 的 BLEU 计算类。

## 2. 设置输出编码

```python
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
```

作用是把 Python 输出改成 UTF-8，避免 Windows 终端打印中文时报 GBK 编码错误。

## 3. 特殊符号

```python
PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
```

- `<pad>`：补齐短句。
- `<bos>`：目标句开始。
- `<eos>`：目标句结束。
- `<unk>`：词表中不存在的字。

## 4. read_tsv

位于第 27 行。参数 `path` 是 TSV 文件路径。每行按 Tab 拆成两列，第一列是英文，第二列是中文。空行跳过，列数不对会报错。

## 5. normalize

位于第 43 行。当前是字符级词表，所以：

```python
"I like tea" -> "ilovetea"
```

它把英文转小写并去掉空格，为逐字符编码做准备。

## 6. CharVocab：词表

位于第 48 行。这就是“构建词表”的代码。

### `__init__`

它统计训练集中出现过的所有字符，去重并排序，然后加上 4 个特殊符号：

```python
tokens = [PAD, BOS, EOS, UNK, *chars]
```

它建立两个映射：

- `itos`：编号到字符，例如 `itos[5] = "a"`。
- `stoi`：字符到编号，例如 `stoi["a"] = 5`。

同时保存 `pad_id`、`bos_id`、`eos_id`、`unk_id`，供后面的模型和数据集使用。

### `__len__`

返回词表大小，也就是模型 Embedding 和输出层的词汇数量。

### `encode`

把文本转成编号列表。`add_eos=True` 时会在末尾追加 `<eos>` 编号。

### `decode`

把编号列表还原成文本，并跳过特殊符号。

## 7. ParallelDataset：样本构造

位于第 73 行。每条样本包含源句编号列表和目标句编号列表。

参数：

- `pairs`：句对列表。
- `src_vocab`：源语言词表。
- `tgt_vocab`：目标语言词表。
- `max_len`：单个句子的最大字符数。

`__getitem__` 做的事情：

1. 把英文按字符编号，并截断到 `max_len`。
2. 把中文按字符编号，再在前后加上 `<bos>` 和 `<eos>`。
3. 返回两个 PyTorch 张量。

## 8. make_collate_fn：补齐 batch

位于第 102 行。一个 batch 里的句子长度不同，`pad_sequence` 用 `pad_id` 把短句补到最长句的长度。

## 9. PositionalEncoding：位置编码

位于第 115 行。Transformer 本身不知道顺序，所以给每个位置加上位置向量。

参数：

- `d_model`：词向量的维度。
- `max_len`：能支持的最大长度。
- `dropout`：随机失活比例。

它使用正弦和余弦函数生成位置编码，并把编码注册为 buffer，表示它不是可训练参数。

## 10. ToyTransformer：模型主体

位于第 132 行。

### 构造参数

- `src_vocab_size`：源语言词表大小。
- `tgt_vocab_size`：目标语言词表大小。
- `src_pad_id`：源语言 `<pad>` 编号。
- `tgt_pad_id`：目标语言 `<pad>` 编号。
- `max_len`：句子最大长度。

### 嵌入矩阵

```python
self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=src_pad_id)
self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_pad_id)
```

这就是“构建嵌入矩阵”的代码。`nn.Embedding` 是一张查找表，每个词表编号对应一个 `d_model` 维向量。`padding_idx` 表示 `<pad>` 对应的向量会固定为零，不参与更新。

### 模型固定参数

- `d_model = 128`：词向量和隐藏层维度。
- `nhead = 8`：多头注意力头数。
- `num_encoder_layers = 3`：Encoder 层数。
- `num_decoder_layers = 3`：Decoder 层数。
- `dim_feedforward = 512`：前馈网络维度。
- `dropout = 0.1`：随机失活比例。
- `batch_first = True`：张量形状为 `[batch, seq, d_model]`。

### 输出层

```python
self.output = nn.Linear(d_model, tgt_vocab_size)
```

把 Transformer 输出的向量映射到目标语言词表大小，得到每个字的得分。

### forward

它先生成三种掩码：

- 源句 padding mask：忽略补齐位置。
- 目标句 padding mask：忽略目标补齐位置。
- causal mask：禁止 Decoder 看到未来的字。

然后依次做 Embedding、位置编码、Transformer，最后线性映射。

## 11. translate：逐字生成

位于第 190 行。这是 Greedy Search。

参数：

- `model`：训练好的模型。
- `source`：英文句子。
- `src_vocab`、`tgt_vocab`：两个词表。
- `device`：CPU 或 GPU。
- `max_src_len`：源句最大长度。
- `max_new_tokens`：最多生成多少个中文字符。

过程：

1. 放入 `<bos>`。
2. 模型预测下一个字的得分。
3. 选择得分最高的字。
4. 把该字接到已生成序列。
5. 重复，直到出现 `<eos>` 或超过最大长度。

## 12. greedy_generate

位于第 214 行。对多条源句分别调用 `translate`。

## 13. evaluate：BLEU 评价

位于第 239 行。

参数：

- `model`：模型。
- `pairs`：测试句对。
- `max_src_len`：源句最大长度。
- `max_new_tokens`：最大生成长度。
- `bleu_tokenizer`：sacreBLEU 使用的分词方式，当前默认 `"zh"`。
- `max_eval_samples`：最多评价多少句，`None` 表示全部评价。

它生成译文后，用 `BLEU(...).corpus_score(...)` 计算语料级 BLEU，并返回 BLEU 分数和样例。

## 14. build_dataloader

位于第 275 行。把 `ParallelDataset` 包装成 DataLoader。

参数：

- `batch_size`：每个 batch 的句子数。
- `shuffle`：是否打乱训练数据。

## 15. train：训练函数

位于第 294 行。

### 初始化

```python
random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

固定随机种子，保证结果可复现，并自动选择 CPU 或 GPU。

### 读取数据和建立词表

分别读取训练、验证、测试 TSV，并用训练集构建源语言和目标语言词表。

### 构建模型

模型词表大小来自 `len(src_vocab)` 和 `len(tgt_vocab)`。

### 损失函数

```python
criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_id)
```

交叉熵损失衡量预测和正确答案的差距，并忽略 `<pad>` 位置。

### 优化器

```python
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
```

Adam 根据梯度更新模型参数，`lr` 是学习率。

### 学习率调度

```python
lr_lambda=lambda step: min((step + 1) ** -0.5, (step + 1) * args.warmup_steps**-1.5)
```

前 `warmup_steps` 步学习率线性上升，之后按平方根下降。

### 训练循环

每个 epoch 遍历所有训练 batch：

```python
target_input = target[:, :-1]
labels = target[:, 1:]
```

这就是 Teacher Forcing。目标句向右错一位：

```text
输入: <bos> 我 喜 欢 茶
标签:      我 喜 欢 茶 <eos>
```

然后：

1. 模型前向计算。
2. 计算 loss。
3. `backward()` 反向传播。
4. 梯度裁剪，防止梯度过大。
5. `optimizer.step()` 更新参数。
6. `scheduler.step()` 更新学习率。

### 验证与保存

到达打印条件时，在验证集上计算 BLEU。训练结束后，把模型状态、两个词表和参数保存为 `.pt` 文件。

最后在测试集上计算 BLEU，并打印 5 个样例。

## 16. parse_args：命令行参数

位于第 406 行。每个参数的含义和默认值如下：

| 参数 | 默认值 | 作用 |
|---|---|---|
| `--train-file` | `data/demo_train.tsv` | 训练集路径 |
| `--valid-file` | `data/demo_valid.tsv` | 验证集路径 |
| `--test-file` | `data/demo_test.tsv` | 测试集路径 |
| `--output-model` | `checkpoints/demo_transformer.pt` | 模型保存路径 |
| `--epochs` | 120 | 训练轮数 |
| `--batch-size` | 16 | 每个 batch 的句子数 |
| `--learning-rate` | 1e-3 | 学习率 |
| `--warmup-steps` | 100 | 学习率预热步数 |
| `--max-len` | 48 | 单个句子最大字符数 |
| `--report-every` | 20 | 每多少轮打印一次验证 BLEU |
| `--bleu-tokenizer` | zh | BLEU 分词方式 |
| `--max-eval-samples` | None | 评价时最多使用多少句 |
| `--seed` | 42 | 随机种子 |
