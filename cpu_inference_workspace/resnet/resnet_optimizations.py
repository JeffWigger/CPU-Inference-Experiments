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

import intel_extension_for_pytorch as ipex
import torch

from torch.utils.data import DataLoader
from torchao.quantization.quant_api import int8_dynamic_activation_int8_weight, int8_weight_only, quantize_
from torchvision import models
from torchvision.models import ResNet101_Weights

from cpu_inference_workspace.resnet.resnet_quant import RQuant


path_ds = os.environ["HOME"] + "/weights"


def resnet_torch_ao_quant_8wo(loader: DataLoader, device: str = "cpu") -> torch.nn.Module:
    resnet = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
    resnet.eval()
    resnet.to(device)
    quantize_(resnet, int8_weight_only())
    return resnet


def resnet_torch_ao_quant_8da(loader: DataLoader, device: str = "cpu") -> torch.nn.Module:
    resnet = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
    resnet.eval()
    resnet.to(device)
    quantize_(resnet, int8_dynamic_activation_int8_weight())
    return resnet


def eager_mode_static_quant(loader: DataLoader, device: str = "cpu") -> torch.nn.Module:
    rquant = RQuant()
    rquant.to(device)
    rquant.eval()
    rquant.qconfig = torch.ao.quantization.get_default_qconfig("x86")  # or fbgemm

    resnet_quant_no_fuse_prepared = torch.ao.quantization.prepare(rquant, inplace=False)

    with torch.no_grad():
        i = 0
        for data, _ in loader:
            resnet_quant_no_fuse_prepared(data)
            i += len(data)
            if i > 100:
                break

    resnet_quant_no_fuse_fin = torch.ao.quantization.convert(resnet_quant_no_fuse_prepared, inplace=False)
    resnet_quant_no_fuse_fin.to(device)
    return resnet_quant_no_fuse_fin


def eager_mode_static_quant_fuse(loader: DataLoader, device: str = "cpu") -> torch.nn.Module:
    rquant = RQuant()
    rquant.to(device)
    rquant.eval()
    config = torch.ao.quantization.get_default_qconfig("x86")  # or fbgemm, onednn

    rquant.qconfig = config

    layer0 = [["resnet.conv1", "resnet.bn1", "resnet.relu"]]  #
    layer1 = [
        [
            [f"resnet.layer1.{i}.conv1", f"resnet.layer1.{i}.bn1"],
            [f"resnet.layer1.{i}.conv2", f"resnet.layer1.{i}.bn2"],
            [f"resnet.layer1.{i}.conv3", f"resnet.layer1.{i}.bn3"],
        ]
        for i in range(3)
    ]  # , f'resnet.layer1.{i}.relu'
    layer2 = [
        [
            [f"resnet.layer2.{i}.conv1", f"resnet.layer2.{i}.bn1"],
            [f"resnet.layer2.{i}.conv2", f"resnet.layer2.{i}.bn2"],
            [f"resnet.layer2.{i}.conv3", f"resnet.layer2.{i}.bn3"],
        ]
        for i in range(4)
    ]  # , f'resnet.layer2.{i}.relu'
    layer3 = [
        [
            [f"resnet.layer3.{i}.conv1", f"resnet.layer3.{i}.bn1"],
            [f"resnet.layer3.{i}.conv2", f"resnet.layer3.{i}.bn2"],
            [f"resnet.layer3.{i}.conv3", f"resnet.layer3.{i}.bn3"],
        ]
        for i in range(23)
    ]  # , f'resnet.layer3.{i}.relu'
    layer4 = [
        [
            [f"resnet.layer4.{i}.conv1", f"resnet.layer4.{i}.bn1"],
            [f"resnet.layer4.{i}.conv2", f"resnet.layer4.{i}.bn2"],
            [f"resnet.layer4.{i}.conv3", f"resnet.layer4.{i}.bn3"],
        ]
        for i in range(3)
    ]  # , f'resnet.layer4.{i}.relu'
    fuse_list = (
        layer0
        + [ll for l in layer1 for ll in l]
        + [ll for l in layer2 for ll in l]
        + [ll for l in layer3 for ll in l]
        + [ll for l in layer4 for ll in l]
    )
    resnet_quant = torch.ao.quantization.fuse_modules(rquant, fuse_list)
    resnet_quant_prepared = torch.ao.quantization.prepare(resnet_quant, inplace=False)

    with torch.no_grad():
        i = 0
        for data, _ in loader:
            resnet_quant_prepared(data)
            i += len(data)
            if i > 100:
                break

    model_quant_fin = torch.ao.quantization.convert(resnet_quant_prepared, inplace=False)
    model_quant_fin.to(device)
    return model_quant_fin


def resnet_no_opt(loader: DataLoader, device: str = "cpu") -> torch.nn.Module:
    resnet = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
    resnet.eval()
    resnet.to(device)
    return resnet


def resnet_compiled(loader: DataLoader, device: str = "cpu") -> torch.nn.Module:
    resnet = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
    resnet.eval()
    resnet.to(device)
    comp_resnet = torch.compile(resnet)
    with torch.no_grad():
        i = 0
        for data, _ in loader:
            comp_resnet(data)
            i += len(data)
            if i > 100:
                break
    return comp_resnet


def resnet_export_compiled(loader: DataLoader, device: str = "cpu") -> torch.nn.Module:
    resnet = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
    resnet.eval()
    resnet.to(device)
    for data, _ in loader:
        data.to(device)
        break
    exp_resnet = torch.export.export(resnet, (data,))
    model = torch.compile(exp_resnet.module(), backend="inductor")
    with torch.no_grad():
        i = 0
        for data, _ in loader:
            model(data)
            i += len(data)
            if i > 100:
                break
    return model


def resnet_onnx(loader: DataLoader, device: str = "cpu"):
    resnet = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
    resnet.eval()
    resnet.to(device)
    for data, _ in loader:
        data.to(device)
        break
    onnx_program = torch.onnx.export(
        resnet,
        data,
        "resnet.onnx",
        input_names=["input"],
        dynamo=True,
    )
    onnx_program.save("resnet.onnx")
    return onnx_program


def resnet_ovn(loader: DataLoader, device: str = "cpu"):
    import openvino as ov

    resnet = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
    # resnet = eager_mode_static_quant(loader), does not work
    resnet.eval()
    resnet.to(device)
    for data, _ in loader:
        data.to(device)
        break
    exported_model = torch.export.export(resnet, (data,))
    # We could also use a onnx model here
    ov_model = ov.convert_model(exported_model)
    return ov.compile_model(ov_model)


def resnet_compile_ovn(loader: DataLoader, device: str = "cpu"):
    resnet = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
    resnet.eval()
    resnet.to(device)
    model = torch.compile(resnet, backend="openvino")
    with torch.no_grad():
        i = 0
        for data, _ in loader:
            model(data)
            i += len(data)
            if i > 100:
                break
    return model


def resnet_compile_ovn_quant(loader: DataLoader, device: str = "cpu"):
    resnet = eager_mode_static_quant(loader)
    resnet.eval()
    resnet.to(device)
    for data, _ in loader:
        data.to(device)
        break
    model = torch.compile(resnet, backend="openvino")
    with torch.no_grad():
        i = 0
        for data, _ in loader:
            model(data)
            i += len(data)
            if i > 100:
                break
    return model


def resnet_compile_ipex(loader: DataLoader, device: str = "cpu"):
    resnet = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
    resnet.eval()
    resnet.to(device)

    model = ipex.optimize(resnet, weights_prepack=False)
    model = torch.compile(model, backend="ipex")
    with torch.no_grad():
        i = 0
        for data, _ in loader:
            model(data)
            i += len(data)
            if i > 100:
                break
    return model


def resnet_compile_ipex_quant(loader: DataLoader, device: str = "cpu"):
    resnet = eager_mode_static_quant_fuse(loader)
    resnet.eval()
    resnet.to(device)

    model = ipex.optimize(resnet, weights_prepack=False)
    model = torch.compile(model, backend="ipex")
    with torch.no_grad():
        i = 0
        for data, _ in loader:
            model(data)
            i += len(data)
            if i > 100:
                break
    return model
