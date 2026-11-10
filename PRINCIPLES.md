# 从一句话理解 Transformer 机器翻译

这份笔记对应 `minimal_transformer_mt.py`，用尽可能少的术语解释机器翻译的过程。

## 1. 机器翻译到底在做什么

机器翻译要学习一个映射：

```text
英文句子 x = "I love tea."
中文句子 y = "我喜欢茶。"
```

这是一个“输入长度不一定等于输出长度”的序列转换问题。英文有 3 个单词，中文有 4 个汉字，所以不能简单地把第 1 个英文单词替换成第 1 个汉字。

Transformer 在这里由两部分组成：

- Encoder：读完整句英文，把内容变成一组上下文向量；
- Decoder：根据 Encoder 的结果和已经生成的中文，预测下一个字。

```text
英文句子
  -> Encoder
  -> 英文上下文表示
  -> Decoder
  -> 我 -> 喜 -> 欢 -> 茶 -> EOS
```

## 2. 计算机怎样表示一句话

### Token

Token 是模型处理的最小单位，可以是一个单词、一个汉字或一个片段（subword）。

玩具样例使用字符级 token：

```text
"I love tea" -> I, l, o, v, e, t, e, a
```

正式实验通常使用 BPE 或 SentencePiece，让常见词保持完整，把生僻词拆成较小的片段。

### 词表

词表把每个 token 对应到一个整数编号：

```text
<pad> -> 0
<bos> -> 1
<eos> -> 2
<unk> -> 3
我 -> 4
喜 -> 5
欢 -> 6
茶 -> 7
```

四个特殊符号的含义：

- `<pad>`：把不同长度的句子补齐成同样长度；
- `<bos>`：告诉 Decoder 开始生成；
- `<eos>`：告诉模型句子结束；
- `<unk>`：词表中没有这个 token。

### Word Embedding

整数编号本身没有语义。Embedding 是一张可学习的查找表，把每个编号变成一个向量，例如：

```text
我 -> [0.2, -0.4, 0.7, ...]
喜 -> [0.1,  0.3, -0.2, ...]
```

向量中的数字一开始是随机的，训练后逐渐包含对任务有用的信息。向量相近的 token 不一定永远是同义词，但模型会用这些数字计算上下文。

## 3. 神经网络怎样“学习”

模型内部有很多可以调整的数字，叫作参数或权重。

一次训练包含以下过程：

1. 模型读取英文句子；
2. 模型预测中文下一个 token 的概率；
3. Loss 衡量预测和正确答案之间的差距；
4. 反向传播计算每个参数对错误的影响；
5. Optimizer 把参数稍微向减少 Loss 的方向调整。

常见术语：

- Batch：一次同时处理多少条句子；
- Epoch：模型把所有训练句完整看一遍；
- Learning rate：每次参数调整的幅度；
- Loss：训练时的错误程度，不是最终的 BLEU；
- Gradient：Loss 对参数的变化方向；
- Backpropagation：从输出向输入逐层计算梯度的过程；
- Parameter：模型训练后保留下来的数字。

## 4. Seq2Seq 和 RNN

Seq2Seq 表示序列到序列：Encoder 读输入序列，Decoder 生成输出序列。

RNN 是按顺序读取 token 的神经网络。每读一个字，就把“到目前为止的信息”保存到一个隐藏状态里。

RNN 的问题：

- 长句子中较早的信息容易丢失；
- 后一个位置必须等待前一个位置计算完，难以并行。

Attention 让 Decoder 在生成每个词时，直接查看 Encoder 的所有输出，而不只依赖一个固定摘要。

## 5. Attention 的 Q、K、V

Attention 可以理解为检索：

- Q（Query）：我现在要找什么信息；
- K（Key）：每个位置提供的检索标签；
- V（Value）：每个位置真正保存的内容。

模型计算 Q 和 K 的相似度，相似度越高，对应位置越重要。相似度经过 Softmax 后变成权重，再用权重把所有 V 加权求和。

公式为：

```text
Attention(Q, K, V)
= Softmax(QK^T / sqrt(d_k)) V
```

在机器翻译中：

- Encoder 内部使用 Self-Attention，英文词彼此查看；
- Decoder 内部使用带因果掩码的 Self-Attention，只看已经生成的中文；
- Decoder 和 Encoder 之间使用 Cross-Attention，让中文生成过程查看英文内容。

Multi-Head Attention 同时使用多组 Q、K、V，从不同角度关注句子，最后合并结果。

## 6. Transformer 的关键组成

### Positional Encoding

Attention 本身不知道 token 的先后顺序。位置编码把“这是第 1 个 token、第 2 个 token”的信息加到 Embedding 上。

### Feed-Forward Network

每个位置经过 Attention 后，还会经过一个小型两层全连接网络。它对每个位置分别进行非线性变换。

### Residual Connection

把输入直接加到子层输出上：

```text
output = input + sublayer(input)
```

这样信息和梯度更容易通过深层网络。

### Layer Normalization

把一层中的数值尺度调整到较稳定的范围，使训练更容易收敛。

### Mask

- Padding Mask：Padding 只是补齐长度的占位符，不参与有效注意力；
- Causal Mask：训练时禁止模型偷看未来 token。

## 7. 训练时模型怎样预测下一个字

以“我喜欢茶”为例，目标序列加入特殊符号后是：

```text
<bos> 我 喜 欢 茶 <eos>
```

Decoder 输入与训练标签错开一位：

```text
Decoder 输入: <bos> 我 喜 欢 茶
训练标签:          我 喜 欢 茶 <eos>
```

根据 `<bos>` 预测“我”，根据 `<bos> 我` 预测“喜”，依此类推。这种做法叫作 Teacher Forcing。

Cross-Entropy Loss 衡量模型给正确 token 分配的概率。如果正确 token 的概率很低，Loss 就较大。

## 8. 生成译文时怎样选词

### Greedy Search

每一步选择当前概率最高的 token。速度快，但前面一步选错后难以纠正。

### Beam Search

同时保留概率最高的若干个候选译文，最后选择整体得分最高的一个。通常质量更好，但速度更慢。

## 9. BLEU 在评价什么

BLEU 比较系统译文和参考译文中的 1-gram、2-gram、3-gram、4-gram 重合程度。

例如：

```text
参考：我 喜欢 茶
输出：我 喜欢 茶
```

重合度很高，BLEU 较高。如果输出太短，即使几个词完全正确，BLEU 也可能虚高，所以还要使用长度惩罚 BP。

BLEU 的局限：

- 同义表达可能被判定为不匹配；
- 不理解句子真正含义；
- 不同数据、分词和评测工具得到的分数不能随便比较。

因此报告中除了 BLEU，还应看训练曲线和具体翻译错误。

## 10. 样例代码对应关系

- `read_tsv`：读取句子对；
- `CharVocab`：创建词表并把文字转成编号；
- `ParallelDataset`：把每条训练样本组织成 Tensor；
- `PositionalEncoding`：加入位置信息；
- `ToyTransformer`：Embedding、Transformer 主体和输出层；
- 训练循环：前向计算、Loss、反向传播、参数更新；
- `translate`：Greedy Search 逐字生成；
- `evaluate`：sacreBLEU 计算 BLEU。

## 11. 建议的学习检查点

能够不用看答案回答下面四个问题，就基本理解了实验主线：

1. 为什么模型不能直接输入字符串？
2. Encoder 和 Decoder 分别负责什么？
3. Attention 中的 Q、K、V 分别在寻找什么？
4. 为什么生成第 3 个汉字时不能看到第 4 个汉字？

完成这四个问题后，再学习矩阵尺寸、Beam Search 和消融实验，不会一开始就被术语淹没。
