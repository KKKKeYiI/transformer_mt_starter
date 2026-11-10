from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
from sacrebleu.metrics import BLEU
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"


def read_tsv(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                raise ValueError(f"{path}:{line_number} 不是两列非空 TSV")
            pairs.append((parts[0].strip(), parts[1].strip()))
    if not pairs:
        raise ValueError(f"{path} 中没有数据")
    return pairs


def normalize(text: str) -> str:
    # 玩具样例使用字符级词表，去掉英文空格后仍然可以逐字符建模。
    return "".join(text.lower().split())


class CharVocab:
    def __init__(self, texts: list[str]) -> None:
        chars = sorted({char for text in texts for char in normalize(text)})
        tokens = [PAD, BOS, EOS, UNK, *chars]
        self.itos = tokens
        self.stoi = {token: index for index, token in enumerate(tokens)}
        self.pad_id = self.stoi[PAD]
        self.bos_id = self.stoi[BOS]
        self.eos_id = self.stoi[EOS]
        self.unk_id = self.stoi[UNK]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str, *, add_eos: bool = False) -> list[int]:
        ids = [self.stoi.get(char, self.unk_id) for char in normalize(text)]
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        ignored = {self.pad_id, self.bos_id, self.eos_id}
        return "".join(self.itos[index] for index in ids if index not in ignored)


class ParallelDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[str, str]],
        src_vocab: CharVocab,
        tgt_vocab: CharVocab,
        max_len: int,
    ) -> None:
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        source, target = self.pairs[index]
        src_ids = self.src_vocab.encode(source)[: self.max_len]
        tgt_ids = [
            self.tgt_vocab.bos_id,
            *self.tgt_vocab.encode(target)[: self.max_len],
            self.tgt_vocab.eos_id,
        ]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(
            tgt_ids, dtype=torch.long
        )


def make_collate_fn(pad_id: int):
    def collate(
        batch: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sources, targets = zip(*batch, strict=True)
        return (
            pad_sequence(sources, batch_first=True, padding_value=pad_id),
            pad_sequence(targets, batch_first=True, padding_value=pad_id),
        )

    return collate


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float) -> None:
        super().__init__()
        positions = torch.arange(max_len).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model)
        )
        encoding = torch.zeros(1, max_len, d_model)
        encoding[0, :, 0::2] = torch.sin(positions * frequencies)
        encoding[0, :, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("encoding", encoding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.encoding[:, : x.size(1)])


class ToyTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_id: int,
        tgt_pad_id: int,
        max_len: int,
    ) -> None:
        super().__init__()
        d_model = 128
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=src_pad_id)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_pad_id)
        self.position = PositionalEncoding(d_model, max_len + 2, dropout=0.1)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
        )
        self.output = nn.Linear(d_model, tgt_vocab_size)

    def forward(
        self,
        source: torch.Tensor,
        target_input: torch.Tensor,
    ) -> torch.Tensor:
        src_padding_mask = source.eq(self.src_pad_id)
        tgt_padding_mask = target_input.eq(self.tgt_pad_id)
        target_length = target_input.size(1)
        causal_mask = torch.triu(
            torch.ones(
                target_length,
                target_length,
                dtype=torch.bool,
                device=target_input.device,
            ),
            diagonal=1,
        )
        src_hidden = self.position(self.src_embedding(source))
        tgt_hidden = self.position(self.tgt_embedding(target_input))
        hidden = self.transformer(
            src_hidden,
            tgt_hidden,
            tgt_mask=causal_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        return self.output(hidden)


@torch.no_grad()
def translate(
    model: ToyTransformer,
    source: str,
    src_vocab: CharVocab,
    tgt_vocab: CharVocab,
    device: torch.device,
    max_src_len: int,
    max_new_tokens: int,
) -> str:
    model.eval()
    src_ids = src_vocab.encode(source)[:max_src_len]
    source_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    generated = torch.tensor(
        [[tgt_vocab.bos_id]], dtype=torch.long, device=device
    )
    for _ in range(max_new_tokens):
        logits = model(source_tensor, generated)
        next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_id), dim=1)
        if next_id.item() == tgt_vocab.eos_id:
            break
    return tgt_vocab.decode(generated[0].tolist())


def greedy_generate(
    model: ToyTransformer,
    pairs: list[tuple[str, str]],
    sources: list[str],
    src_vocab: CharVocab,
    tgt_vocab: CharVocab,
    device: torch.device,
    max_src_len: int,
    max_new_tokens: int,
) -> list[str]:
    del pairs
    return [
        translate(
            model,
            source,
            src_vocab,
            tgt_vocab,
            device,
            max_src_len,
            max_new_tokens,
        )
        for source in sources
    ]


def evaluate(
    model: ToyTransformer,
    pairs: list[tuple[str, str]],
    src_vocab: CharVocab,
    tgt_vocab: CharVocab,
    device: torch.device,
    max_src_len: int,
    max_new_tokens: int,
    bleu_tokenizer: str,
    max_eval_samples: int | None = None,
) -> tuple[float, list[tuple[str, str, str]]]:
    eval_pairs = pairs if max_eval_samples is None else pairs[:max_eval_samples]
    sources = [source for source, _ in eval_pairs]
    references = [target for _, target in eval_pairs]
    hypotheses = greedy_generate(
        model,
        eval_pairs,
        sources,
        src_vocab,
        tgt_vocab,
        device,
        max_src_len,
        max_new_tokens,
    )
    bleu = BLEU(tokenize=bleu_tokenizer).corpus_score(
        hypotheses, [references]
    ).score
    examples = [
        (source, target, hypothesis)
        for (source, target), hypothesis in zip(
            eval_pairs, hypotheses, strict=True
        )
    ]
    return bleu, examples


def build_dataloader(
    pairs: list[tuple[str, str]],
    src_vocab: CharVocab,
    tgt_vocab: CharVocab,
    max_len: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = ParallelDataset(
        pairs, src_vocab, tgt_vocab, max_len=max_len
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=make_collate_fn(src_vocab.pad_id),
    )


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_pairs = read_tsv(Path(args.train_file))
    valid_pairs = read_tsv(Path(args.valid_file))
    test_pairs = read_tsv(Path(args.test_file))
    src_vocab = CharVocab([source for source, _ in train_pairs])
    tgt_vocab = CharVocab([target for _, target in train_pairs])

    train_loader = build_dataloader(
        train_pairs,
        src_vocab,
        tgt_vocab,
        args.max_len,
        args.batch_size,
        shuffle=True,
    )
    model = ToyTransformer(
        len(src_vocab),
        len(tgt_vocab),
        src_vocab.pad_id,
        tgt_vocab.pad_id,
        args.max_len,
    ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(
            (step + 1) ** -0.5, (step + 1) * args.warmup_steps**-1.5
        ),
    )

    print(f"设备: {device}")
    print(
        f"训练/验证/测试样本数: "
        f"{len(train_pairs)}/{len(valid_pairs)}/{len(test_pairs)}"
    )
    print(f"词表大小: 源语言 {len(src_vocab)}, 目标语言 {len(tgt_vocab)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for source, target in train_loader:
            source = source.to(device)
            target = target.to(device)
            target_input = target[:, :-1]
            labels = target[:, 1:]
            logits = model(source, target_input)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        if epoch == 1 or epoch % args.report_every == 0 or epoch == args.epochs:
            valid_bleu, _ = evaluate(
                model,
                valid_pairs,
                src_vocab,
                tgt_vocab,
                device,
                args.max_len,
                args.max_len + 2,
                args.bleu_tokenizer,
                args.max_eval_samples,
            )
            average_loss = total_loss / len(train_loader)
            print(
                f"Epoch {epoch:03d} | train loss {average_loss:.4f} "
                f"| valid BLEU {valid_bleu:.2f}"
            )

    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "src_itos": src_vocab.itos,
            "tgt_itos": tgt_vocab.itos,
            "args": vars(args),
        },
        output_path,
    )

    test_bleu, examples = evaluate(
        model,
        test_pairs,
        src_vocab,
        tgt_vocab,
        device,
        args.max_len,
        args.max_len + 2,
        args.bleu_tokenizer,
        args.max_eval_samples,
    )
    print(f"测试集 BLEU: {test_bleu:.2f}")
    print("测试集样例:")
    for source, reference, hypothesis in examples[:5]:
        print(f"原文: {source}")
        print(f"参考: {reference}")
        print(f"输出: {hypothesis}")
    print(f"模型已保存到: {output_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="最小可运行的 Transformer 翻译样例")
    parser.add_argument("--train-file", default="data/demo_train.tsv")
    parser.add_argument("--valid-file", default="data/demo_valid.tsv")
    parser.add_argument("--test-file", default="data/demo_test.tsv")
    parser.add_argument("--output-model", default="checkpoints/demo_transformer.pt")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-len", type=int, default=48)
    parser.add_argument("--report-every", type=int, default=20)
    parser.add_argument("--bleu-tokenizer", default="zh")
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
