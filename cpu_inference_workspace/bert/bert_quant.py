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

import torch

from transformers import (
    AutoModelForSequenceClassification,
)

from cpu_inference_workspace.bert.bert import BertEmbeddings


class BertQuant(torch.nn.Module):
    def __init__(self, path_model):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.bert = AutoModelForSequenceClassification.from_pretrained(str(path_model))
        self.bert.bert.embeddings = BertEmbeddings(self.bert.config)
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.config = self.bert.config

    def forward(self, *p, **x):
        x = self.bert(*p, **x)
        x = self.dequant(x)
        return x
