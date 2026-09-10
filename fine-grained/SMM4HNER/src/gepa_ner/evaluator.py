"""Lightweight BERT adapter training and evaluation for GEPA fitness.

Reuses the adapter-based approach from ``models.bert_adapter`` but with
much smaller hyper-parameters so that each GEPA iteration stays fast:
  - bert-base-cased (instead of deberta-v3-large)
  - 30 training epochs (with best-model checkpoint selection on relaxed F1)
  - batch size 16
  - trains in a temporary directory that is cleaned up afterwards

The public API is a single function :func:`train_and_evaluate` that returns
per-example predictions and an aggregate F1 score.
"""

from __future__ import annotations

import ast
import os
import shutil
import tempfile
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from data.evaluation import (
    calculate_f1_per_entity_covering_all,
    evaluate_test_strict_ner,
)
from utils.device import get_device, get_training_arguments_device_kwargs, log_device
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
)

LABEL_LIST = [
    "O",
    "B-ClinicalImpacts",
    "I-ClinicalImpacts",
    "B-SocialImpacts",
    "I-SocialImpacts",
]
LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}

MODEL_NAME = "bert-base-cased"
ADAPTER_NAME = "ner_adapter"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def autocorrect_labels(ner_tags: list[str]) -> list[str]:
    """Fix invalid BIO sequences (lone I- without preceding B-)."""
    corrected: list[str] = []
    for tag in ner_tags:
        if tag.startswith("I-"):
            entity = tag[2:]
            if not corrected or corrected[-1] not in (f"B-{entity}", f"I-{entity}"):
                corrected.append(f"B-{entity}")
            else:
                corrected.append(tag)
        else:
            corrected.append(tag)
    return corrected


def _compute_metrics(eval_pred) -> dict:
    from seqeval.metrics import f1_score, precision_score, recall_score

    logits, label_ids = eval_pred
    predictions = np.argmax(logits, axis=2)
    true_labels, true_preds = [], []
    for pred_seq, label_seq in zip(predictions, label_ids):
        tl, tp = [], []
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                tl.append(ID2LABEL[l])
                tp.append(ID2LABEL[p])
        true_labels.append(tl)
        true_preds.append(tp)

    per_entity = calculate_f1_per_entity_covering_all(true_labels, true_preds)
    overall_relaxed = per_entity.get("Overall", {})

    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds),
        "relaxed_f1": overall_relaxed.get("F1-Score", 0.0),
    }


def _records_to_hf_dataset(records: list[dict]) -> "Dataset":
    from datasets import Dataset
    return Dataset.from_list(records)


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    records = []
    for _, row in df.iterrows():
        tokens = ast.literal_eval(row["tokens"]) if isinstance(row["tokens"], str) else row["tokens"]
        ner_tags = ast.literal_eval(row["ner_tags"]) if isinstance(row["ner_tags"], str) else row["ner_tags"]
        records.append({"tokens": tokens, "ner_tags": ner_tags})
    return records


def _make_tokenize_fn(tokenizer):
    """Return a tokenize-and-align closure for HF dataset ``.map()``."""
    def tokenize_and_align(examples):
        tok_inputs = tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True,
        )
        all_labels = []
        for i, tags in enumerate(examples["ner_tags"]):
            word_ids = tok_inputs.word_ids(batch_index=i)
            label_ids = []
            prev = None
            for wid in word_ids:
                if wid is None:
                    label_ids.append(-100)
                elif wid != prev:
                    label_ids.append(LABEL2ID.get(tags[wid], 0))
                else:
                    label_ids.append(-100)
                prev = wid
            all_labels.append(label_ids)
        tok_inputs["labels"] = all_labels
        return tok_inputs
    return tokenize_and_align


def _predict_on_records(model, tokenizer, records: list[dict]) -> "EvalResult":
    """Run inference on *records* and return an :class:`EvalResult`."""
    device = get_device()
    model.to(device)
    model.eval()
    model.set_active_adapters(ADAPTER_NAME)
    model.active_head = ADAPTER_NAME

    all_gold: list[list[str]] = []
    all_pred: list[list[str]] = []
    per_example: list[dict] = []

    for rec in records:
        tokens = rec["tokens"]
        gold_tags = rec["ner_tags"]
        encoding = tokenizer(
            tokens, is_split_into_words=True,
            return_tensors="pt", truncation=True,
        ).to(device)
        with torch.no_grad():
            logits = model(**encoding).logits
        preds = torch.argmax(logits, dim=-1).squeeze().tolist()
        if isinstance(preds, int):
            preds = [preds]
        word_ids = encoding.word_ids()

        word_preds: dict[int, str] = {}
        for idx, wid in enumerate(word_ids):
            if wid is not None and wid not in word_preds:
                word_preds[wid] = ID2LABEL[preds[idx]]

        pred_tags = autocorrect_labels(
            [word_preds.get(i, "O") for i in range(len(tokens))]
        )
        correct = pred_tags == gold_tags
        per_example.append({
            "tokens": tokens,
            "gold_tags": gold_tags,
            "pred_tags": pred_tags,
            "correct": correct,
        })
        all_gold.append(gold_tags)
        all_pred.append(pred_tags)

    strict_df = pd.DataFrame({"gold": all_gold, "pred": all_pred})
    strict = evaluate_test_strict_ner(
        strict_df, gold_col="gold", pred_col="pred", print_report=False,
    )
    per_entity = calculate_f1_per_entity_covering_all(all_gold, all_pred)
    overall_relaxed = per_entity.get("Overall", {})

    return EvalResult(
        f1=strict["f1_strict"],
        precision=strict["precision_strict"],
        recall=strict["recall_strict"],
        relaxed_f1=overall_relaxed.get("F1-Score", 0.0),
        relaxed_precision=overall_relaxed.get("Precision", 0.0),
        relaxed_recall=overall_relaxed.get("Recall", 0.0),
        per_entity_results=per_entity,
        per_example=per_example,
    )


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

@dataclass
class EvalResult:
    """Result of a train-and-evaluate cycle."""
    f1: float
    precision: float
    recall: float
    relaxed_f1: float
    relaxed_precision: float
    relaxed_recall: float
    per_entity_results: dict
    per_example: list[dict]
    """List of dicts with keys: tokens, gold_tags, pred_tags, correct."""


def train_and_evaluate(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    *,
    eval_df: pd.DataFrame | None = None,
    num_epochs: int = 30,
    batch_size: int = 16,
) -> EvalResult:
    """Train a BERT adapter on *train_df*, predict on *dev_df*, return results.

    Parameters
    ----------
    eval_df : DataFrame, optional
        Full dev set used for training-time evaluation metrics (the Trainer's
        ``eval_dataset``).  Falls back to *dev_df* when not provided.
    dev_df : DataFrame
        Examples to produce final per-example predictions on (the GEPA batch).
    """
    from adapters import AdapterConfig, AdapterTrainer, AutoAdapterModel

    log_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoAdapterModel.from_pretrained(MODEL_NAME)

    adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=6)
    model.add_adapter(ADAPTER_NAME, config=adapter_config)
    model.add_tagging_head(
        ADAPTER_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
    )
    model.train_adapter(ADAPTER_NAME)

    tokenize_and_align = _make_tokenize_fn(tokenizer)

    train_records = _df_to_records(train_df)
    dev_records = _df_to_records(dev_df)
    eval_records = _df_to_records(eval_df) if eval_df is not None else dev_records

    train_ds = _records_to_hf_dataset(train_records).map(
        tokenize_and_align, batched=True, remove_columns=["tokens", "ner_tags"],
    )
    eval_ds = _records_to_hf_dataset(eval_records).map(
        tokenize_and_align, batched=True, remove_columns=["tokens", "ner_tags"],
    )

    tmpdir = tempfile.mkdtemp(prefix="gepa_bert_")
    try:
        training_args = TrainingArguments(
            output_dir=tmpdir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=5e-4,
            warmup_ratio=0.1,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_relaxed_f1",
            greater_is_better=True,
            save_total_limit=1,
            logging_strategy="steps",
            logging_steps=10,
            disable_tqdm=False,
            report_to="none",
            **get_training_arguments_device_kwargs(),
        )

        trainer = AdapterTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
            data_collator=DataCollatorForTokenClassification(tokenizer),
            compute_metrics=_compute_metrics,
        )
        trainer.train()

        return _predict_on_records(model, tokenizer, dev_records)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def train_baseline(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    save_dir: str,
    *,
    num_epochs: int = 30,
    batch_size: int = 16,
) -> tuple[float, str]:
    """Train a BERT adapter on *train_df*, save checkpoint, return baseline relaxed F1.

    The adapter weights, tagging head, and tokenizer are persisted to
    *save_dir* so they can be reloaded by :func:`finetune_from_checkpoint`.

    Returns ``(baseline_relaxed_f1, save_dir)``.
    """
    from adapters import AdapterConfig, AdapterTrainer, AutoAdapterModel

    log_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoAdapterModel.from_pretrained(MODEL_NAME)

    adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=6)
    model.add_adapter(ADAPTER_NAME, config=adapter_config)
    model.add_tagging_head(
        ADAPTER_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
    )
    model.train_adapter(ADAPTER_NAME)

    tokenize_and_align = _make_tokenize_fn(tokenizer)

    train_records = _df_to_records(train_df)
    dev_records = _df_to_records(dev_df)

    train_ds = _records_to_hf_dataset(train_records).map(
        tokenize_and_align, batched=True, remove_columns=["tokens", "ner_tags"],
    )
    eval_ds = _records_to_hf_dataset(dev_records).map(
        tokenize_and_align, batched=True, remove_columns=["tokens", "ner_tags"],
    )

    tmpdir = tempfile.mkdtemp(prefix="gepa_baseline_")
    try:
        training_args = TrainingArguments(
            output_dir=tmpdir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=5e-4,
            warmup_ratio=0.1,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_relaxed_f1",
            greater_is_better=True,
            save_total_limit=1,
            logging_strategy="steps",
            logging_steps=10,
            disable_tqdm=False,
            report_to="none",
            **get_training_arguments_device_kwargs(),
        )

        trainer = AdapterTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
            data_collator=DataCollatorForTokenClassification(tokenizer),
            compute_metrics=_compute_metrics,
        )
        trainer.train()

        os.makedirs(save_dir, exist_ok=True)
        model.save_adapter(save_dir, ADAPTER_NAME)
        model.save_head(save_dir, ADAPTER_NAME)
        tokenizer.save_pretrained(save_dir)

        result = _predict_on_records(model, tokenizer, dev_records)
        return result.relaxed_f1, save_dir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def finetune_from_checkpoint(
    checkpoint_dir: str,
    synth_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    *,
    num_epochs: int = 30,
    batch_size: int = 16,
) -> EvalResult:
    """Load a saved baseline adapter, fine-tune on *synth_df*, evaluate on *dev_df*."""
    from adapters import AdapterTrainer, AutoAdapterModel

    log_device()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoAdapterModel.from_pretrained(MODEL_NAME)

    model.load_adapter(checkpoint_dir)
    model.load_head(checkpoint_dir)
    model.train_adapter(ADAPTER_NAME)

    tokenize_and_align = _make_tokenize_fn(tokenizer)

    synth_records = _df_to_records(synth_df)
    dev_records = _df_to_records(dev_df)

    synth_ds = _records_to_hf_dataset(synth_records).map(
        tokenize_and_align, batched=True, remove_columns=["tokens", "ner_tags"],
    )
    eval_ds = _records_to_hf_dataset(dev_records).map(
        tokenize_and_align, batched=True, remove_columns=["tokens", "ner_tags"],
    )

    tmpdir = tempfile.mkdtemp(prefix="gepa_finetune_")
    try:
        training_args = TrainingArguments(
            output_dir=tmpdir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=5e-4,
            warmup_ratio=0.1,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_relaxed_f1",
            greater_is_better=True,
            save_total_limit=1,
            logging_strategy="steps",
            logging_steps=10,
            disable_tqdm=False,
            report_to="none",
            **get_training_arguments_device_kwargs(),
        )

        trainer = AdapterTrainer(
            model=model,
            args=training_args,
            train_dataset=synth_ds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
            data_collator=DataCollatorForTokenClassification(tokenizer),
            compute_metrics=_compute_metrics,
        )
        trainer.train()

        return _predict_on_records(model, tokenizer, dev_records)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
