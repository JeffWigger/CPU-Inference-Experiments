#    Copyright 2025 Jeffrey Wigger
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os

import datasets as ds
import numpy as np
import torch

from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
)


# To avoid warning due to forking
os.environ["TOKENIZERS_PARALLELISM"] = "false"

to_label = {0: "NEGATIVE", 1: "POSITIVE"}
to_id = {"NEGATIVE": 0, "POSITIVE": 1}


def load_imbd_test(path_ds, path_model, batch_size=4, device="cpu", ds_size=25000) -> DataLoader:
    imdb = ds.load_dataset("imdb", cache_dir=path_ds, split=ds.Split.TEST)

    tokenizer = AutoTokenizer.from_pretrained(str(path_model))

    def tokenize(feature):
        return tokenizer(feature["text"], truncation=True)

    tokenized = imdb.map(tokenize, batched=True)
    tokenized = tokenized.remove_columns("text")
    imbd_ds = tokenized.with_format("torch", device=device)
    print(f"IMDB Dataset has {len(imbd_ds)} data points.")
    len_ds = len(imbd_ds)
    indices = np.arange(len_ds)
    np.random.shuffle(indices)
    indices = list(indices[:ds_size])
    test_loader = DataLoader(
        imbd_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        sampler=torch.utils.data.SubsetRandomSampler(indices),
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding=True),
        drop_last=True,  # Needed for exported models
        prefetch_factor=2,
    )
    return test_loader


def load_imbd_train(path_ds, path_model, batch_size=4, device="cpu") -> DataLoader:
    imdb = ds.load_dataset("imdb", cache_dir=path_ds, split=ds.Split.TRAIN)

    tokenizer = AutoTokenizer.from_pretrained(str(path_model))

    def tokenize(feature):
        return tokenizer(feature["text"], truncation=True)

    tokenized = imdb.map(tokenize, batched=True)
    tokenized = tokenized.remove_columns("text")
    imbd_ds = tokenized.with_format("torch", device=device)

    train_loader = DataLoader(
        imbd_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
        drop_last=True,  # Needed for exported models
        prefetch_factor=2,
    )
    return train_loader
