# A Kernel-Based View of Language Model Fine-Tuning

This is the implementation for the paper [A Kernel-Based View of Language Model Fine-tuning]()
and can be used to compute kernel approximations for the fine-tuning of pre-trained language models.

We extend the [LM-BFF](https://github.com/princeton-nlp/LM-BFF) repository and
add a new "kernel trainer" powered by [functorch](https://github.com/pytorch/functorch) to compute empirical NTK kernel matrices using the SGD, SignGD or Asymmetric-SignGD kernel formulas.
If you are interested in analyzing our kernels, we also provide our pre-computed kernels for download.

## Installation
Please install all the dependency packages by using the following command:
```
pip install -r requirements.txt
```

We updated the LM-BFF code to work with a newer version of HuggingFace transformers and additionally require functorch.
If you would like to run LoRA fine-tuning, install the LoRA version of the transformers library ([see here](https://github.com/microsoft/LoRA/tree/main/examples/NLU)) and add the flags `--apply_lora --lora_alpha .... --lora_r ...` .

<!-- TODO: update requirements -->

## Prepare the data
Please run the following commands to download and prepare the data:

```bash
( cd data; bash download_dataset.sh )

for K in 16 64 512; do
    # Generate k-shot splits for seeds 13,21,42,87,100 with a maximum of 1k test examples in data/k-shot-1k-test
    python tools/generate_k_shot_data.py --mode k-shot-1k-test --k $K
done
```

**NOTE**: During training, the model will generate/load cache files in the data folder. If your data have changed, make sure to clean all the cache files (starting with "cache").

## Run the code
To easily run our experiments, you can use `run_fewshot.sh`:

```bash
TAG=kernel-prompting TRAINER=kernel TASK=SST-2 SEED=42 MODEL=roberta-base bash run_experiment.sh
```

The templates and label word mappings are already defined, so you only need to set hyper-parameters and `TAG` (you can use whatever tag you want and it just makes finding results easier). See `run_fewshot.sh` for more options. Besides, you can easily add extra arguments:

```bash
TAG=kernel-prompting TRAINER=kernel TASK=SST-2 SEED=42 MODEL=roberta-base bash run_experiment.sh \
    --kernel_formula signgd --kernel_solver logistic  --per_device_train_batch_size 2 --per_device_eval_batch_size 4
```
This uses the SignGD kernel formula and a logistic kernel solver (the default is least-squares regression) and uses batch sizes 2 and 4 along the two axes of the kernel matrices.

For more advanced use cases, such as [how to aggregate results over multiple runs](https://github.com/princeton-nlp/LM-BFF#experiments-with-multiple-runs), [zero-shot experiments](https://github.com/princeton-nlp/LM-BFF#zero-shot-experiments) or [writing your own prompt formats](https://github.com/princeton-nlp/LM-BFF#how-to-design-your-own-templates), we refer to the README in the LM-BFF repo.
Note that we deleted some tools to do automatic prompt and label search that are unrelated to our paper.

 ## Download our pre-computed kernels
Coming soon...

## Citation

Please cite our work if you make use of our code or our pre-computed kernels in your work:

```bibtex
Coming soon...
```
