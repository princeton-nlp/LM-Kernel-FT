# A Kernel-Based View of Language Model Fine-Tuning (ICML'23)

This is the implementation for the paper [A Kernel-Based View of Language Model Fine-tuning](https://arxiv.org/abs/2210.05643)
and can be used to compute kernel approximations for the fine-tuning of pre-trained language models.

We extend the [LM-BFF](https://github.com/princeton-nlp/LM-BFF) repository and
add a new "kernel trainer" powered by [functorch](https://github.com/pytorch/functorch) to compute empirical-NTK kernel matrices using the SGD, SignGD or Asymmetric-SignGD kernel formulas.
We also provide our pre-computed kernels for download to facilitate further analysis.

## Installation
Please install all the dependency packages by using the following command:
```
pip install -r requirements.txt
```

We updated the LM-BFF code to work with a newer version of HuggingFace transformers and additionally require functorch.
If you would like to run LoRA fine-tuning, install the LoRA version of the transformers library ([see here](https://github.com/microsoft/LoRA/tree/main/examples/NLU)) and add the flags `--apply_lora --lora_alpha .... --lora_r ...` .

**NOTE**: Different versions of some packages (`pytorch`, `numpy`, `transformers`) may cause minor variations in kernels and results.

## Prepare the data
Please run the following commands to download and prepare the data:

```bash
( cd data; bash download_dataset.sh )

for K in 16 64 512; do
    # Generate k-shot splits for seeds 13,21,42,87,100 with a maximum of 1k test examples in data/k-shot-1k-test,
    # where k is the number of training/validation examples per label
    python tools/generate_k_shot_data.py --mode k-shot-1k-test --k $K
done
```

This follows LM-BFF, but `download_dataset.sh` additionally rebalances the `cr` dataset and uses the GLUE version of the SST-2 dataset. Additionally `k-shot-1k-test` limits test datasets to 1k examples for faster evaluation.

**NOTE**: During training, the model will generate/load cache files in the data folder. If your data have changed, make sure to clean all the cache files (starting with "cache").

## Run the code
To easily run our experiments, you can use `run_fewshot.sh`:

```bash
TAG=kernel-prompting TRAINER=kernel TASK=SST-2 SEED=42 MODEL=roberta-base bash run_fewshot.sh
```

The templates and label word mappings are already defined, so you only need to set hyper-parameters and `TAG` (you can use whatever tag you want and it just makes finding results easier). See `run_fewshot.sh` for more options. Besides, you can easily add extra arguments:

```bash
NUM_GPU=4 TAG=kernel-prompting TRAINER=kernel TASK=SST-2 SEED=42 MODEL=roberta-base bash run_fewshot.sh \
    --kernel_formula signgd --kernel_solver logistic  --per_device_train_batch_size 2 --per_device_eval_batch_size 4
```
This splits the kernel computation across 4 GPUs and uses the SignGD kernel formula and a logistic kernel solver (the default is least-squares regression) and uses batch sizes 2 and 4 along the two axes of the kernel matrices respectively.

For more advanced use cases, such as [how to aggregate results over multiple runs](https://github.com/princeton-nlp/LM-BFF#experiments-with-multiple-runs), [zero-shot experiments](https://github.com/princeton-nlp/LM-BFF#zero-shot-experiments) or [writing your own prompt formats](https://github.com/princeton-nlp/LM-BFF#how-to-design-your-own-templates), we refer to the README in the LM-BFF repo.
Note that we deleted some tools to do automatic prompt and label search that are unrelated to our paper.

 ## Download our pre-computed kernels
Here are the links for downloading our pre-computed kernels:

|        | SGD | SignGD | Asymmetric-SignGD | 
|--------|-----|--------|-------------------|
| 16-shot| [prompt](https://nlp.cs.princeton.edu/projects/LM-Kernel-FT/roberta-base/prompt-sgd-16-shot.zip) / [no-prompt](https://nlp.cs.princeton.edu/projects/LM-Kernel-FT/roberta-base/no_prompt-sgd-16-shot.zip)    |  [prompt](https://nlp.cs.princeton.edu/projects/LM-Kernel-FT/roberta-base/prompt-signgd-16-shot.zip) / [no-prompt](https://nlp.cs.princeton.edu/projects/LM-Kernel-FT/roberta-base/no_prompt-signgd-16-shot.zip)       |  [prompt](https://nlp.cs.princeton.edu/projects/LM-Kernel-FT/roberta-base/prompt-asymmetric_signgd-16-shot.zip) / [no-prompt](https://nlp.cs.princeton.edu/projects/LM-Kernel-FT/roberta-base/no_prompt-asymmetric_signgd-16-shot.zip) |
| 64-shot| [prompt](https://nlp.cs.princeton.edu/projects/LM-Kernel-FT/roberta-base/prompt-sgd-64-shot.zip) / [no-prompt](https://nlp.cs.princeton.edu/projects/LM-Kernel-FT/roberta-base/no_prompt-sgd-64-shot.zip)    |  [prompt](https://nlp.cs.princeton.edu/projects/LM-Kernel-FT/roberta-base/prompt-signgd-64-shot.zip) / [no-prompt](https://nlp.cs.princeton.edu/projects/LM-Kernel-FT/roberta-base/no_prompt-signgd-64-shot.zip)       |  [prompt](https://nlp.cs.princeton.edu/projects/LM-Kernel-FT/roberta-base/prompt-asymmetric_signgd-64-shot.zip) / [no-prompt](https://nlp.cs.princeton.edu/projects/LM-Kernel-FT/roberta-base/no_prompt-asymmetric_signgd-64-shot.zip) |
| 512-shot| [prompt](https://nlp.cs.princeton.edu/projects/LM-Kernel-FT/roberta-base/prompt-sgd-512-shot.zip) / [no-prompt](https://nlp.cs.princeton.edu/projects/LM-Kernel-FT/roberta-base/no_prompt-sgd-512-shot.zip)  | | |||

The provided kernels were computed for RoBERTa-base for 14 datasets (SST-2, SST-5, MR, CR, MPQA, Subj, TREC, AG News, MNLI, SNLI, QNLI, RTE, MRPC, QQP). The no prompt kernels were obtained by initializing the [CLS] head with the logistic regression solution.

For each task and data split, we include separate files for training, development, test kernel matrices and pre-trained logits. Each file can be read using `torch.load` and contains a tuple of (kernel matrix, labels),
and the kernel matrix has the shape of [training examples, training logits, *X* examples, *X* logits], where *X* dataset is given by the file name (train, dev or test).

## Bugs and questions?
If you have any questions related to the code or the paper, feel free to email Alexander and Sadhika (`{awettig,smalladi}@cs.princeton.edu`). If you encounter a problem or bug when using the code, you can also open an issue.

## Citation

Please cite our work if you make use of our code or our pre-computed kernels in your work:

```bibtex

@InProceedings{malladi2023kernel,
  title = 	 {A Kernel-Based View of Language Model Fine-Tuning},
  author =       {Malladi, Sadhika and Wettig, Alexander and Yu, Dingli and Chen, Danqi and Arora, Sanjeev},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {23610--23641},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v202/malladi23a/malladi23a.pdf},
  url = 	 {https://proceedings.mlr.press/v202/malladi23a.html}
}
```
