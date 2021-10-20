from typing import List

import pandas as pd
import torch
from transformers import PreTrainedTokenizer


class GDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, messages: List[str], labels: List[List[int]],
                 max_len: int = 512):
        self.tokenizer = tokenizer
        self.messages = messages
        self.labels = labels
        self.max_len = max_len
        self.num_heads = len(labels)
        assert len(self.labels) > 0, "At least contain one label"

    def __len__(self):
        return len(self.labels[0])

    def __getitem__(self, idx):
        message = str(self.messages[idx]).strip()
        labels = [self.labels[i][idx] for i in range(len(self.labels))]
        encoding = self.tokenizer.encode_plus(
            message,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors='pt',
        )

        return {
            "message": message,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            **{f"label{idx + 1}": labels[idx] for idx in range(len(self.labels))},
        }


def create_data_loader(df: pd.DataFrame, tokenizer: PreTrainedTokenizer, max_len: int, batch_size: int):
    num_heads = sum(df.columns.str.contains("label*"))
    ds = GDataset(
        tokenizer,
        df["text"].to_numpy(),
        [df[f"label{idx + 1}"].to_numpy() for idx in range(num_heads)],
        max_len=max_len
    )
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=4)
