# SMM4HNER

SMM4HNER is a command-line toolkit for named-entity recognition and prompt classification workflows. It can train and run multiple NER and classification models, evaluate predictions with the shared-task metrics, augment datasets, and run the GEPA-based prompt-evolution pipeline used in this repository.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Train a model with automatic device detection:

```bash
hner train data/new_train_data.csv data/new_dev_data.csv output/bert_adapter --type bert_adapter
```

Override the runtime device when needed:

```bash
hner train data/new_train_data.csv data/new_dev_data.csv output/cpu_run --type bert_adapter --device cpu
hner train data/new_train_data.csv data/new_dev_data.csv output/apple_silicon --type bert_adapter --device mps
hner train data/new_train_data.csv data/new_dev_data.csv output/amd_gpu --type bert_adapter --device rocm
```

Supported device values include `cpu`, `cuda`, `rocm`, `mps`, and `xpu`. If `--device` is omitted, the code scans the available accelerators automatically and picks the best supported option.

## Data format

NER training and inference files are CSV files with at least:

- `ID`
- `tokens`
- `ner_tags`

The `tokens` and `ner_tags` columns are stored as Python-style lists in the example data under `data/`.

## Examples

Train and save a fine-tuned model:

```bash
hner train data/new_train_data.csv data/new_dev_data.csv output/bert_finetuned --type bert_finetuned --device auto
```

Run inference with a saved model:

```bash
hner test data/new_dev_data.csv output/predictions.csv --type bert_finetuned --model output/bert_finetuned --device cpu
```

Evaluate predictions with strict and relaxed SMM4H metrics:

```bash
hner evaluate data/new_dev_data-output.csv --gold-col ner_tags --pred-col prediction
```

Run GEPA prompt evolution:

```bash
hner evolve data/new_train_data.csv data/new_dev_data.csv --ollama-model your-ollama-model --device mps
```

Augment a dataset with the Ollama-based augmenter:

```bash
hner augment data/new_train_data.csv output/augmented.csv --strategy negate
```
