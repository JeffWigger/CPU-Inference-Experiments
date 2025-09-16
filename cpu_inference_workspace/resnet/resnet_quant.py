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

from torchvision.models import ResNet101_Weights
from torchvision.models.resnet import _resnet

from cpu_inference_workspace.resnet.bottleneck_quant import BottleneckQ


class RQuant(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        weights = ResNet101_Weights.verify(ResNet101_Weights.IMAGENET1K_V2)
        self.resnet = _resnet(BottleneckQ, [3, 4, 23, 3], weights, True)
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.resnet(x)
        x = self.dequant(x)
        return x
