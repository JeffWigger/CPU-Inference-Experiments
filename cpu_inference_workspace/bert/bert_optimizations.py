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

from pathlib import Path

import intel_extension_for_pytorch as ipex
import torch
import torch.ao.quantization
import transformers.onnx

from torch.utils.data import DataLoader
from torchao.quantization.quant_api import int8_dynamic_activation_int8_weight, int8_weight_only, quantize_
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.convert_graph_to_onnx import convert

from cpu_inference_workspace.bert.bert_quant import BertQuant


# To avoid warning due to forking
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def bert_torch_ao_quant_8wo(loader: DataLoader, path_model: str, device: str = "cpu") -> torch.nn.Module:
    bert = AutoModelForSequenceClassification.from_pretrained(str(path_model))
    bert.eval()
    bert.to(device)
    quantize_(bert, int8_weight_only())
    return bert


def bert_torch_ao_quant_8da(loader: DataLoader, path_model: str, device: str = "cpu") -> torch.nn.Module:
    bert = AutoModelForSequenceClassification.from_pretrained(str(path_model))
    bert.eval()
    bert.to(device)
    quantize_(bert, int8_dynamic_activation_int8_weight())
    return bert


# def bert_torch_ao_quant_auto(loader: DataLoader) -> torch.nn.Module:
#     # Needs cuda
#     bert = AutoModelForSequenceClassification.from_pretrained(str(path_model))
#     bert.eval()
#     bert.to(device)
#     bert = autoquant(bert)
#     return bert


def bert_eager_mode_static_quant(loader: DataLoader, path_model: str, device: str = "cpu") -> torch.nn.Module:
    # bert = AutoModelForSequenceClassification.from_pretrained(str(path_model))
    # bert.qconfig = torch.ao.quantization.get_default_qconfig("x86")
    # bquant = torch.ao.quantization.QuantWrapper(bert)
    bquant = BertQuant(path_model=path_model)
    bquant.to(device)
    bquant.eval()
    bquant.bert.bert.embeddings.word_embeddings.qconfig = torch.ao.quantization.float_qparams_weight_only_qconfig
    bquant.bert.bert.embeddings.position_embeddings.qconfig = torch.ao.quantization.float_qparams_weight_only_qconfig
    bquant.bert.bert.embeddings.token_type_embeddings.qconfig = torch.ao.quantization.float_qparams_weight_only_qconfig
    bquant.bert.bert.embeddings.LayerNorm.qconfig = torch.ao.quantization.get_default_qconfig("x86")
    bquant.bert.bert.embeddings.dropout.qconfig = torch.ao.quantization.get_default_qconfig("x86")
    bquant.bert.bert.embeddings.ff_add.qconfig = torch.ao.quantization.get_default_qconfig("x86")
    bquant.bert.bert.embeddings.ff_add_2.qconfig = torch.ao.quantization.get_default_qconfig("x86")
    bquant.bert.bert.encoder.qconfig = torch.ao.quantization.get_default_qconfig("x86")
    bquant.bert.bert.pooler.qconfig = torch.ao.quantization.get_default_qconfig("x86")
    bquant.bert.dropout.qconfig = torch.ao.quantization.get_default_qconfig("x86")
    bquant.bert.classifier.qconfig = torch.ao.quantization.get_default_qconfig("x86")
    # PlaceholderObserver cannot be used torch.ao.quantization.convert
    # embeddings are not supported with this setting

    bert_quant_no_fuse_prepared = torch.ao.quantization.prepare(bquant, inplace=False)

    with torch.no_grad():
        i = 0
        for items in loader:
            items.to(device)
            bert_quant_no_fuse_prepared(**items)
            i += len(items["labels"])
            if i > 100:
                break

    bert_quant_no_fuse_fin = torch.ao.quantization.convert(bert_quant_no_fuse_prepared, inplace=False)
    # TODO: The layernorm after the embeddings receives non quantized inputs, need to quantize them.
    bert_quant_no_fuse_fin.to(device)
    return bert_quant_no_fuse_fin


def bert_eager_mode_dynamic_quant(loader: DataLoader, path_model: str, device: str = "cpu") -> torch.nn.Module:
    bquant = BertQuant(path_model=path_model)
    bquant.to(device)
    bquant.eval()
    bquant.qconfig = torch.ao.quantization.get_default_qconfig("x86")

    bert_dyn_quant = torch.ao.quantization.quantize_dynamic(
        bquant, {torch.nn.Embedding: torch.ao.quantization.float_qparams_weight_only_qconfig}, dtype=torch.qint8
    )

    bert_dyn_quant.to(device)
    return bert_dyn_quant


def bert_no_opt(loader: DataLoader, path_model: str, device: str = "cpu") -> torch.nn.Module:
    bert = AutoModelForSequenceClassification.from_pretrained(str(path_model))
    bert.eval()
    bert.to(device)
    return bert


def bert_compiled(loader: DataLoader, path_model: str, device: str = "cpu") -> torch.nn.Module:
    bert = AutoModelForSequenceClassification.from_pretrained(str(path_model))
    bert.eval()
    bert.to(device)
    comp_bert = torch.compile(bert)
    # warm up
    with torch.no_grad():
        i = 0
        for items in loader:
            items.to(device)
            comp_bert(**items)
            i += len(items["labels"])
            if i > 100:
                break
    return comp_bert


def bert_export_compiled(loader: DataLoader, path_model: str, device: str = "cpu") -> torch.nn.Module:
    bert = AutoModelForSequenceClassification.from_pretrained(str(path_model))
    bert.eval()
    bert.to(device)
    with torch.no_grad():
        i = 0
        for items in loader:
            items.to(device)
            break
    batch = torch.export.Dim("batch1")
    seq = torch.export.Dim("sequence1", max=512)
    dyna = {"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq}, "token_type_ids": {0: batch, 1: seq}}
    exp_bert = torch.export.export(
        bert,
        args=(items["input_ids"], items["attention_mask"], items["token_type_ids"]),
        dynamic_shapes=dyna,
        strict=False,
    )
    model = torch.compile(exp_bert.module(), backend="inductor")
    with torch.no_grad():
        i = 0
        for items in loader:
            items.to(device)
            model(**items)
            i += len(items["labels"])
            if i > 100:
                break
    return model


def bert_onnx_graph_export(loader: DataLoader, path_model: str, device: str = "cpu"):
    bert = AutoModelForSequenceClassification.from_pretrained(str(path_model))
    bert.eval()
    bert.to(device)
    with torch.no_grad():
        for items in loader:
            items.to(device)
            break
    items = {k: v.to(device) for k, v in items.items()}
    export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
    onnx_program = torch.onnx.dynamo_export(
        bert, items["input_ids"], items["attention_mask"], items["token_type_ids"], export_options=export_options
    )
    onnx_program.save("bert.onnx")
    return onnx_program


def bert_onnx_graph(loader: DataLoader, path_model: str, device: str = "cpu"):
    convert(framework="pt", model=str(path_model), output="bert.onnx", opset=11)
    return


def bert_onnx(loader: DataLoader, path_model: str, device: str = "cpu"):
    bert = AutoModelForSequenceClassification.from_pretrained(str(path_model))
    tokenizer = AutoTokenizer.from_pretrained(str(path_model))
    bert.eval()
    bert.to(device)
    # load config
    _, model_onnx_config = transformers.onnx.FeaturesManager.check_supported_model_or_raise(
        bert, feature="sequence-classification"
    )
    onnx_config = model_onnx_config(bert.config)

    # export
    onnx_inputs, onnx_outputs = transformers.onnx.export(
        preprocessor=tokenizer, model=bert, config=onnx_config, opset=14, output=Path("bert.onnx")
    )
    return (onnx_inputs, onnx_outputs)


# def bert_onnx_quant(loader: DataLoader):
#     bert = bert_eager_mode_dynamic_quant(loader=loader)
#     bert.eval()
#     bert.to(device)
#     # load config
#     model_kind, model_onnx_config = transformers.onnx.FeaturesManager.check_supported_model_or_raise(bert, feature="sequence-classification")
#     onnx_config = model_onnx_config(bert.config)

#     # export
#     onnx_inputs = transformers.onnx.export(
#             preprocessor=tokenizer,
#             model=bert,
#             config=onnx_config,
#             opset=14,
#             output=Path("bert.onnx")
#     )
#     print(onnx_inputs)
#     return onnx_inputs


def bert_ovn(loader: DataLoader, path_model: str, device: str = "cpu"):
    import openvino as ov

    bert = AutoModelForSequenceClassification.from_pretrained(str(path_model))
    bert.eval()
    bert.to(device)
    for items in loader:
        items.to(device)
        break
    dyna = [
        ("input_ids", [-1, -1]),
        ("attention_mask", [-1, -1]),
        ("token_type_ids", [-1, -1]),
    ]
    exported_model = torch.export.export(bert, (items["input_ids"], items["attention_mask"], items["token_type_ids"]))
    # We could also use a onnx model here
    ov_model = ov.convert_model(
        exported_model, dyna, example_input=(items["input_ids"], items["attention_mask"], items["token_type_ids"])
    )
    return ov.compile_model(ov_model)


def bert_ovn_onnx(loader: DataLoader, path_model: str, device: str = "cpu"):
    import openvino as ov

    bert_onnx(loader, path_model, device)
    for items in loader:
        items.to(device)
        break
    dyna = [
        ("input_ids", [-1, -1]),
        ("attention_mask", [-1, -1]),
        ("token_type_ids", [-1, -1]),
    ]
    ov_model = ov.convert_model(
        "bert.onnx", dyna, example_input=(items["input_ids"], items["attention_mask"], items["token_type_ids"])
    )
    return ov.compile_model(ov_model)


def bert_compile_ovn(loader: DataLoader, path_model: str, device: str = "cpu"):
    bert = AutoModelForSequenceClassification.from_pretrained(str(path_model))
    bert.eval()
    bert.to(device)

    model = torch.compile(bert, backend="openvino")
    with torch.no_grad():
        i = 0
        for items in loader:
            items.to(device)
            model(**items)
            i += len(items["labels"])
            if i > 100:
                break
    return model


def bert_compile_ovn_quant(loader: DataLoader, path_model: str, device: str = "cpu"):
    bert = bert_eager_mode_dynamic_quant(loader=loader, path_model=path_model, device=device)
    bert.eval()
    bert.to(device)
    model = torch.compile(bert, backend="openvino")
    with torch.no_grad():
        i = 0
        for items in loader:
            items.to(device)
            model(**items)
            i += len(items["labels"])
            if i > 100:
                break
    return model


def bert_compile_ipex(loader: DataLoader, path_model: str, device: str = "cpu"):
    bert = AutoModelForSequenceClassification.from_pretrained(str(path_model))
    bert.eval()
    bert.to(device)

    model = ipex.optimize(bert, weights_prepack=False)
    model = torch.compile(model, backend="ipex")
    with torch.no_grad():
        i = 0
        for items in loader:
            items.to(device)
            model(**items)
            i += len(items["labels"])
            if i > 100:
                break
    return model


def bert_compile_ipex_quant(loader: DataLoader, path_model: str, device: str = "cpu"):
    bert = bert_eager_mode_dynamic_quant(loader=loader, path_model=path_model, device=device)
    bert.eval()
    bert.to(device)

    model = ipex.optimize(bert, weights_prepack=False)
    model = torch.compile(model, backend="ipex")
    with torch.no_grad():
        i = 0
        for items in loader:
            items.to(device)
            model(**items)
            i += len(items["labels"])
            if i > 100:
                break
    return model
