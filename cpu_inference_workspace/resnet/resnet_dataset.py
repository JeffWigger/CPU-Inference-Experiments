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

from pathlib import Path

import numpy as np
import torch

from torch.utils.data import DataLoader
from torchvision.datasets import Imagenette


# Mapping Imagenette Labels to Imagenet
LABEL_MAPPING = {0: 0, 1: 217, 2: 482, 3: 491, 4: 497, 5: 566, 6: 569, 7: 571, 8: 574, 9: 701}


def label_mapping(x):
    return LABEL_MAPPING[x]


def load_imagenette_val(tform, path_ds, batch_size=4, ds_size=3925) -> DataLoader:
    download = False
    if not Path(path_ds).exists():
        download = True
    ds = Imagenette(path_ds, "val", size="full", download=download, transform=tform, target_transform=label_mapping)
    len_ds = len(ds)
    indices = np.arange(len_ds)
    np.random.shuffle(indices)
    indices = list(indices[:ds_size])
    test_loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=torch.utils.data.SubsetRandomSampler(indices),
        num_workers=2,
        drop_last=True,  # Needed for exported models
        prefetch_factor=2,
    )
    return test_loader


def load_imagenette_train(tform, path_ds, batch_size=4) -> DataLoader:
    download = False
    if not Path(path_ds).exists():
        download = True
    ds = Imagenette(path_ds, "train", size="full", download=download, transform=tform)

    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True, prefetch_factor=2)
    return train_loader
