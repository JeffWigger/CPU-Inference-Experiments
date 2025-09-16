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

import argparse
import os
import time

from functools import partial
from pathlib import Path

import onnxruntime
import torch

from torch.utils.data import DataLoader
from torchvision.models import ResNet101_Weights
from tqdm import tqdm

import cpu_inference_workspace.resnet.resnet_optimizations as opt

from cpu_inference_workspace.resnet.resnet_dataset import load_imagenette_train, load_imagenette_val


tform = ResNet101_Weights.IMAGENET1K_V2.transforms()
smax = torch.nn.Softmax(dim=1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


NS_TO_SECONDS = 1e9


def parse_args():
    base_path = Path(os.environ["HOME"]) / "weights"
    parser = argparse.ArgumentParser(description="Resnet experiments with various optimizations")
    parser.add_argument("--ds-size", type=int, default=160, help="Dataset size to use for experiments")
    parser.add_argument("--path-ds", type=str, default=str(base_path), help="Path to the datasets")
    parser.add_argument(
        "--exp-name",
        type=str,
        default="experiments/resnet-cpu-experiments",
        help="Base name for the experiment file, relative path to an existing folder. Omit the .csv ending.",
    )
    return parser.parse_args()


def run_and_measure(model: torch.nn.Module, loader: DataLoader, ds_size: int):
    total_time = 0
    correct = 0
    with tqdm(total=ds_size) as pbar:
        with torch.no_grad():
            for data, target in loader:
                data.to(device)
                target.to(device)
                start_time = time.monotonic_ns()
                pred = model(data)
                end_time = time.monotonic_ns()
                total_time += end_time - start_time
                _, predicted = torch.max(smax(pred.data), -1)
                correct += (predicted == target).sum().item()
                pbar.update(target.size()[0])
    time_per_input = (total_time / ds_size) / NS_TO_SECONDS
    total_time = total_time / NS_TO_SECONDS
    accuracy = correct / ds_size
    print(
        f"Execution for batchsize {loader.batch_size} took {time_per_input} seconds per image, total {total_time}, accuracy {accuracy}"
    )
    return (time_per_input, total_time, accuracy)


def run_and_measure_onnx(onnx_program, loader, ds_size: int, ort_session):
    total_time = 0
    correct = 0
    with tqdm(total=ds_size) as pbar:
        with torch.no_grad():
            for data, target in loader:
                data.to(device)
                target.to(device)
                # onnx_input = onnx_program.adapt_torch_inputs_to_onnx(data)
                start_time = time.monotonic_ns()
                onnxruntime_input = {ort_session.get_inputs()[0].name: data.cpu().numpy()}
                onnxruntime_outputs = ort_session.run(None, onnxruntime_input)
                end_time = time.monotonic_ns()
                total_time += end_time - start_time
                # TODO: slow conversion, see warning
                pred = torch.squeeze(torch.tensor(onnxruntime_outputs))
                if len(pred.shape) == 1:
                    pred = pred.reshape((1, 1000))
                _, predicted = torch.max(smax(pred.data), -1)
                correct += (predicted == target).sum().item()
                pbar.update(target.size()[0])
    time_per_input = (total_time / ds_size) / NS_TO_SECONDS
    total_time = total_time / NS_TO_SECONDS
    accuracy = correct / ds_size
    print(
        f"Execution for batchsize {loader.batch_size} took {time_per_input} seconds per image, total {total_time}, accuracy {accuracy}"
    )
    return (time_per_input, total_time, accuracy)


def run_and_measure_ov(ov_model, loader, ds_size: int):
    total_time = 0
    correct = 0
    with tqdm(total=ds_size) as pbar:
        for data, target in loader:
            data.to(device)
            target.to(device)
            start_time = time.monotonic_ns()
            ov_outputs = ov_model(data)
            end_time = time.monotonic_ns()
            pred = torch.squeeze(torch.tensor(ov_outputs[0]))
            if len(pred.shape) == 1:
                pred = pred.reshape((1, 1000))
            total_time += end_time - start_time
            _, predicted = torch.max(smax(pred.data), -1)
            correct += (predicted == target).sum().item()
            pbar.update(target.size()[0])
    time_per_input = (total_time / ds_size) / NS_TO_SECONDS
    total_time = total_time / NS_TO_SECONDS
    accuracy = correct / ds_size
    print(
        f"Execution for batchsize {loader.batch_size} took {time_per_input} seconds per image, total {total_time}, accuracy {accuracy}"
    )
    return (time_per_input, total_time, accuracy)


if __name__ == "__main__":
    args = parse_args()
    exp_name = f"{args.exp_name}-{int(time.time())}.csv"
    header = "name,batch_size,time_per_input,total_time,accuracy"

    # importing openvino breaks onnx

    exp_funcs = [
        (opt.resnet_no_opt, run_and_measure),
        # (opt.resnet_torch_ao_quant_auto, run_and_measure),
        (opt.resnet_torch_ao_quant_8wo, run_and_measure),
        (opt.resnet_torch_ao_quant_8da, run_and_measure),
        (opt.resnet_ovn, run_and_measure_ov),
        (opt.resnet_compile_ovn, run_and_measure),
        (opt.resnet_compile_ovn_quant, run_and_measure),
        (opt.resnet_compile_ipex, run_and_measure),
        (opt.resnet_onnx, run_and_measure_onnx),
        (opt.resnet_compile_ipex_quant, run_and_measure),
        (opt.resnet_no_opt, run_and_measure),
        (opt.resnet_compiled, run_and_measure),
        (opt.resnet_export_compiled, run_and_measure),
        (opt.eager_mode_static_quant, run_and_measure),
        (opt.eager_mode_static_quant_fuse, run_and_measure),
    ]

    warm_up_ds = 8
    with open(exp_name, "w") as fh:
        for bs in [1, 4, 8]:
            print(f"Batchsize: {bs}")
            loader_val = load_imagenette_val(tform, args.path_ds, bs, ds_size=args.ds_size)
            loader_val_warm = load_imagenette_val(tform, args.path_ds, bs, ds_size=warm_up_ds)
            loader_train = load_imagenette_train(tform, args.path_ds, bs)
            for f, runner in exp_funcs:
                line = f"{f.__name__},{bs},"

                model = f(loader_train)
                if runner is run_and_measure_onnx:
                    ort_session = onnxruntime.InferenceSession("./resnet.onnx", providers=["CPUExecutionProvider"])

                    exp_runner = partial(runner, ort_session=ort_session)
                else:
                    exp_runner = runner

                print(f"Warm-up: {f.__name__}")
                _ = exp_runner(model, loader_val_warm, warm_up_ds)
                print(f"Experiment: {f.__name__}")
                result = exp_runner(model, loader_val, args.ds_size)
                fh.write(line + ",".join(map(str, result)) + "\n")
                fh.flush()
