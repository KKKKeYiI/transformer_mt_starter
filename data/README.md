# 短句演示数据

## 文件

- `demo_train.tsv`：训练集，320 句。
- `demo_valid.tsv`：验证集，40 句。
- `demo_test.tsv`：测试集，40 句。

数据来自 Tatoeba 项目的中英短句对，并从 OPUS 提供的 `cmn-en` 版本中筛选。英文句子长度为 1 到 6 个单词，中文已统一转换为简体中文，并去除了数字、网址和乱码。

为了让模型在少量数据上能学到规律，整理时把意思相同的英语表达按同一句中文归组：训练集保留多种英语说法，验证集和测试集各保留一种同义说法。测试集没有直接复制训练集句子。

来源：

- Tatoeba Project: https://tatoeba.org
- OPUS Tatoeba corpus: https://opus.nlpl.eu

## 格式

每行两列，使用 Tab 分隔：

```text
英文句子<Tab>中文句子
```

## 训练命令

```powershell
python .\minimal_transformer_mt.py
```

默认使用 `demo_train.tsv`、`demo_valid.tsv` 和 `demo_test.tsv`。
