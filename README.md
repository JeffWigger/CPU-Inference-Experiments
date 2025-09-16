# CPU Inference Experiments

This project applies and compares several optimizations that improve the inference performance of machine learning models on CPUs.

The results of the experiments can be found in the `results` folder.

The terraform code to create the infrastructure to reproduce the experiments can be found in the `infrastructure` folder.

Finally, run the experiments with:

```
python mlops_inference_workspace/bert_experiments.py --ds-size 1000
```

and

```
python mlops_inference_workspace/resnet_experiments.py --ds-size 1000
```
