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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


df = pd.read_csv("resnet-cpu-experiments-1757793025.csv", names=["experiment", "b_size", "t_iter", "t_total", "acc"])
df_b = pd.read_csv("bert-cpu-experiments-1757804244.csv", names=["experiment", "b_size", "t_iter", "t_total", "acc"])


b_sizes = ("Batch Size 1", "Batch Size 4", "Batch Size 8")

x_pos = np.arange(len(b_sizes))


def crate_plot(datasets, labels, name, title, ncols=None):
    width = 1 / (len(datasets) + 1)
    print(width)
    fig, ax = plt.subplots(layout="constrained", figsize=(8, 5))
    colors = plt.cm.Pastel1(np.linspace(0, 0.1 * len(datasets), len(datasets)))
    for i, (ds, color) in enumerate(zip(datasets, colors, strict=False)):
        offset = width * (i + 1)
        print(offset)
        print(x_pos)
        ax.bar(x_pos + offset, ds.t_total, width, label=labels[i], color=color)

    ax.set_ylabel("Runtime (Seconds)")
    ax.set_title(title)
    ax.set_xticks(x_pos + 0.5, b_sizes)
    ax.legend(loc="upper left", ncols=len(datasets) if ncols is None else ncols)
    ax.set_ylim(0, max([ds.t_total.max() for ds in datasets]) + 40)

    plt.savefig(name)


# Compiled
r_no_opt = df[df.experiment == "resnet_no_opt"]
r_compiled = df[df.experiment == "resnet_compiled"]
r_compiled_exp = df[df.experiment == "resnet_export_compiled"]
datasets = [r_no_opt, r_compiled, r_compiled_exp]
labels = ["No Optimization", "Compiled", "Compiled and Exported"]
crate_plot(datasets, labels, "resnet_compiled.svg", "Resnet Compiled")

b_no_opt = df_b[df_b.experiment == "bert_no_opt"]
b_compiled = df_b[df_b.experiment == "bert_compiled"]
datasets = [b_no_opt, b_compiled]
labels = ["No Optimization", "Compiled"]
crate_plot(datasets, labels, "bert_compiled.svg", "BERT Compiled")


# OpenVINO
r_ovn = df[df.experiment == "resnet_ovn"]
r_compiled_ovn = df[df.experiment == "resnet_compile_ovn"]
datasets = [r_no_opt, r_compiled_ovn, r_ovn]
labels = ["No Optimization", "Compiled OpenVINO", "OpenVINO"]
crate_plot(datasets, labels, "resnet_ovn.svg", "Resnet OpenVINO")

b_compiled_ovn = df_b[df_b.experiment == "bert_compile_ovn"]
datasets = [b_no_opt, b_compiled_ovn]
labels = ["No Optimization", "Compiled OpenVINO"]
crate_plot(datasets, labels, "bert_ovn.svg", "BERT OpenVINO")


# IPEX
r_compile_ipex = df[df.experiment == "resnet_compile_ipex"]
datasets = [r_no_opt, r_compile_ipex]
labels = ["No Optimization", "Compiled IPEX"]
crate_plot(datasets, labels, "resnet_ipex.svg", "Resnet IPEX")


b_compiled_ipex = df_b[df_b.experiment == "bert_compile_ipex"]
datasets = [b_no_opt, b_compiled_ipex]
labels = ["No Optimization", "Compiled IPEX"]
crate_plot(datasets, labels, "bert_ipex.svg", "BERT IPEX")


# ONNX
r_onnx = df[df.experiment == "resnet_onnx"]
datasets = [r_no_opt, r_onnx]
labels = ["No Optimization", "Compiled ONNX"]
crate_plot(datasets, labels, "resnet_onnx.svg", "Resnet ONNX")

b_onnx = df_b[df_b.experiment == "bert_onnx"]
b_ovn_onnx = df_b[df_b.experiment == "bert_ovn_onnx"]
datasets = [b_no_opt, b_onnx, b_ovn_onnx]
labels = ["No Optimization", "ONNX", "OpenVINO ONNX"]
crate_plot(datasets, labels, "bert_onnx.svg", "BERT ONNX")


# Quantization
r_static_quant = df[df.experiment == "eager_mode_static_quant"]
r_static_quant_fuse = df[df.experiment == "eager_mode_static_quant_fuse"]
datasets = [r_no_opt, r_static_quant, r_static_quant_fuse]
labels = ["No Optimization", "Static Quantization", "Static Quantization with Fusion"]
crate_plot(datasets, labels, "resnet_quant.svg", "Resnet Quantization")

b_dynamic_quantization = df_b[df_b.experiment == "bert_eager_mode_dynamic_quant"]
datasets = [b_no_opt, b_dynamic_quantization]
labels = ["No Optimization", "Dynamic Quantization"]
crate_plot(datasets, labels, "bert_quant.svg", "BERT Quantization")


# Quantization OVN
print("problem")
r_ovn = df[df.experiment == "resnet_ovn"]
r_compiled_ovn = df[df.experiment == "resnet_compile_ovn"]
r_static_quant_ovn = df[df.experiment == "resnet_compile_ovn_quant"]
datasets = [r_no_opt, r_compiled_ovn, r_ovn, r_static_quant_ovn]
labels = ["No Optimization", "Compiled OpenVINO", "OpenVINO", "Compiled OpenVINO + Quant"]
crate_plot(datasets, labels, "resnet_ovn_quant.svg", "Resnet OpenVINO with Quantization", ncols=2)

b_compiled_ovn = df_b[df_b.experiment == "bert_compile_ovn"]
b_dynamic_quantization_ovn = df_b[df_b.experiment == "bert_compile_ovn_quant"]
datasets = [b_no_opt, b_compiled_ovn, b_dynamic_quantization_ovn]
labels = ["No Optimization", "Compiled OpenVINO", "Compiled OpenVINO + Quant"]
crate_plot(datasets, labels, "bert_ovn_quant.svg", "BERT OpenVINO with Quantization")


# Quantization IPEX
r_compile_ipex = df[df.experiment == "resnet_compile_ipex"]
r_static_quant_ipex = df[df.experiment == "resnet_compile_ipex_quant"]
datasets = [r_no_opt, r_compile_ipex, r_static_quant_ipex]
labels = ["No Optimization", "Compiled IPEX", "Compiled IPEX + Quant"]
crate_plot(datasets, labels, "resnet_ipex_quant.svg", "Resnet IPEX with Quantization")


b_compiled_ipex = df_b[df_b.experiment == "bert_compile_ipex"]
b_dynamic_quantization_quant_ipex = df_b[df_b.experiment == "bert_compile_ipex_quant"]
datasets = [b_no_opt, b_compiled_ipex, b_dynamic_quantization_quant_ipex]
labels = ["No Optimization", "Compiled IPEX", "Compiled IPEX + Quant"]
crate_plot(datasets, labels, "bert_ipex_quant.svg", "BERT IPEX with Quantization")
