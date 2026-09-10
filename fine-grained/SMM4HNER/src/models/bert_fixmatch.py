"""BERT model with FixMatch based semi-supervised token classification training."""

import ast
import json
import os
from copy import deepcopy

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

from models.base import BaseNERModel
from models.losses import MulticlassDiceLoss
from utils.device import get_device, log_device

class UnlabeledDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: AutoTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load_and_chunk(data_path)

    def _load_and_chunk(self, data_path: str):
        samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get("text", "")
                except json.JSONDecodeError:
                    continue
                if not text:
                    continue
                # tokenize and chunk. is_split_into_words=False because it's raw text
                tokenized = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_length,
                    return_overflowing_tokens=True,
                    stride=0,
                    padding="max_length"
                )
                for i in range(len(tokenized["input_ids"])):
                    samples.append({
                        "input_ids": torch.tensor(tokenized["input_ids"][i]),
                        "attention_mask": torch.tensor(tokenized["attention_mask"][i])
                    })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class LabeledDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, label2id: dict, max_length: int = 512):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.samples = self._tokenize_and_align(df)

    def _tokenize_and_align(self, df: pd.DataFrame):
        samples = []
        for _, row in df.iterrows():
            tokens = ast.literal_eval(row["tokens"]) if isinstance(row["tokens"], str) else row["tokens"]
            ner_tags = ast.literal_eval(row["ner_tags"]) if isinstance(row["ner_tags"], str) else row["ner_tags"]
            labels = [self.label2id.get(tag, 0) for tag in ner_tags]

            tokenized = self.tokenizer(
                tokens,
                truncation=True,
                max_length=self.max_length,
                is_split_into_words=True,
                padding="max_length"
            )

            word_ids = tokenized.word_ids()
            new_labels = []
            current_word = None
            for word_id in word_ids:
                if word_id != current_word:
                    current_word = word_id
                    label = -100 if word_id is None else labels[word_id]
                    new_labels.append(label)
                elif word_id is None:
                    new_labels.append(-100)
                else:
                    label = labels[word_id]
                    if label % 2 == 1:
                        label += 1
                    new_labels.append(label)

            samples.append({
                "input_ids": torch.tensor(tokenized["input_ids"]),
                "attention_mask": torch.tensor(tokenized["attention_mask"]),
                "labels": torch.tensor(new_labels)
            })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class BertFixMatchModel(BaseNERModel):
    """Semi-supervised BERT for NER using FixMatch principles in embedding space."""

    DEFAULT_MODEL_NAME = "microsoft/deberta-v3-small"
    DEFAULT_MAX_LENGTH: int = 512
    DEFAULT_BATCH_SIZE: int = 4

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.max_length: int = self.DEFAULT_MAX_LENGTH
        self.batch_size: int = self.DEFAULT_BATCH_SIZE

    def _build_tokenizer_and_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.LABEL_LIST),
            id2label=self.ID2LABEL,
            label2id=self.LABEL2ID,
        )

    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        output_path: str,
        metric: str = "f1",
        **kwargs
    ) -> None:
        log_device()
        device = get_device()
        self._build_tokenizer_and_model()
        self.model.to(device)

        unlabeled_file = kwargs.get("unlabeled_file", None)
        if not unlabeled_file:
            raise ValueError("unlabeled_file must be provided for FixMatch training.")

        mu = kwargs.get("mu", 2)  # Ratio of unlabeled to labeled batches
        threshold = kwargs.get("threshold", 0.8) # Entity threshold
        threshold_o = kwargs.get("threshold_o", 0.95) # High threshold specifically for "O" class
        lambda_u = kwargs.get("lambda_u", 1.0)
        batch_size = kwargs.get("batch_size", self.DEFAULT_BATCH_SIZE)
        epochs = kwargs.get("max_training_epochs", kwargs.get("epochs", 10))
        warmup_epochs = kwargs.get("warmup_epochs", 1)  # Epochs with no unlabeled loss
        lr = kwargs.get("learning_rate", 5e-5)
        # Support both max_sequence_len (new unified kwarg) and the legacy max_length kwarg.
        max_length = kwargs.get("max_sequence_len", kwargs.get("max_length", self.DEFAULT_MAX_LENGTH))
        self.max_length = max_length
        self.batch_size = batch_size
        alpha_ce = kwargs.get("alpha_ce", 0.5)
        alpha_dice = kwargs.get("alpha_dice", 0.5)

        o_label_id = self.LABEL2ID.get("O", 0)

        print(f"[hner][fixmatch] Building datasets (mu={mu}, thresh_entity={threshold}, thresh_O={threshold_o})...")
        labeled_ds = LabeledDataset(train_data, self.tokenizer, self.LABEL2ID, max_length=max_length)
        unlabeled_ds = UnlabeledDataset(unlabeled_file, self.tokenizer, max_length=max_length)
        val_ds = LabeledDataset(val_data, self.tokenizer, self.LABEL2ID, max_length=max_length)

        labeled_loader = DataLoader(labeled_ds, batch_size=batch_size, shuffle=True)
        unlabeled_loader = DataLoader(unlabeled_ds, batch_size=batch_size * mu, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        total_steps = len(labeled_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)

        dice_loss_fn = MulticlassDiceLoss(ignore_index=-100)
        best_metric = -1.0
        os.makedirs(output_path, exist_ok=True)

        for epoch in range(epochs):
            self.model.train()
            labeled_iter = iter(labeled_loader)
            unlabeled_iter = iter(unlabeled_loader)
            
            pbar = tqdm(range(len(labeled_loader)), desc=f"Epoch {epoch+1}/{epochs}")
            total_labeled_loss = 0.0
            total_unlabeled_loss = 0.0

            for step in pbar:
                try:
                    batch_l = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(labeled_loader)
                    batch_l = next(labeled_iter)
                    
                try:
                    batch_u = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    batch_u = next(unlabeled_iter)

                # Move labeled data to device
                input_ids_l = batch_l["input_ids"].to(device)
                att_mask_l = batch_l["attention_mask"].to(device)
                labels_l = batch_l["labels"].to(device)

                # Move unlabeled data to device
                input_ids_u = batch_u["input_ids"].to(device)
                att_mask_u = batch_u["attention_mask"].to(device)

                # --- Labeled Forward Phase ---
                outputs_l = self.model(input_ids=input_ids_l, attention_mask=att_mask_l, labels=labels_l)
                labeled_loss = alpha_ce * outputs_l.loss + alpha_dice * dice_loss_fn(outputs_l.logits, labels_l)

                if epoch < warmup_epochs:
                    unlabeled_loss = torch.tensor(0.0, device=device)
                    pass_rate = 0.0
                else:
                    # --- FixMatch Unlabeled Phase ---
                    embeds_u = self.model.get_input_embeddings()(input_ids_u)
                    weak_embeds = embeds_u.detach() + torch.randn_like(embeds_u) * 0.01
                    
                    # Strong augmentation: more noise + dropout/cutoff
                    noise_scale = 0.1 * embeds_u.std(dim=-1, keepdim=True)
                    strong_embeds = embeds_u + torch.randn_like(embeds_u) * noise_scale
                    
                    # Cutoff (Token Dropout over the embedding dimension)
                    cutoff_prob = 0.1
                    cutoff_mask = (torch.rand(strong_embeds.shape[:-1], device=device).unsqueeze(-1) > cutoff_prob).float()
                    # PROTECT THE [CLS] TOKEN
                    cutoff_mask[:, 0, :] = 1.0
                    strong_embeds = strong_embeds * cutoff_mask / (1.0 - cutoff_prob + 1e-6)

                    # Pseudo-labels from weak forward pass
                    with torch.no_grad():
                        weak_logits = self.model(inputs_embeds=weak_embeds, attention_mask=att_mask_u).logits
                        probs = torch.softmax(weak_logits, dim=-1)
                        max_probs, pseudo_labels = torch.max(probs, dim=-1) # Hard pseudo labels
                        
                        # Class-specific threshold logic
                        is_o_class = (pseudo_labels == o_label_id)
                        confidence_mask = torch.where(
                            is_o_class,
                            max_probs >= threshold_o,
                            max_probs >= threshold
                        )

                    # Strong predictions
                    strong_logits = self.model(inputs_embeds=strong_embeds, attention_mask=att_mask_u).logits
                    
                    # Compute unlabeled cross-entropy loss against the pseudo-labels
                    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                    ul_unreduced = loss_fct(strong_logits.view(-1, self.model.config.num_labels), pseudo_labels.view(-1))
                    ul_unreduced = ul_unreduced.view(strong_logits.shape[0], strong_logits.shape[1])
                    
                    # Apply mask: tokens must pass confidence threshold and be real (attention_mask == 1)
                    valid_mask = confidence_mask & att_mask_u.bool()
                    
                    total_valid_u = att_mask_u.sum().float()
                    passed_u = valid_mask.sum().float()
                    pass_rate = (passed_u / total_valid_u).item() * 100.0 if total_valid_u > 0 else 0.0

                    if valid_mask.sum() > 0:
                        unlabeled_loss = (ul_unreduced * valid_mask.float()).sum() / valid_mask.sum()
                    else:
                        unlabeled_loss = torch.tensor(0.0, device=device)

                # Overall Loss
                loss = labeled_loss + lambda_u * unlabeled_loss

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total_labeled_loss += labeled_loss.item()
                total_unlabeled_loss += unlabeled_loss.item()

                pbar.set_postfix({
                    "L_lab": f"{labeled_loss.item():.3f}", 
                    "L_unl": f"{unlabeled_loss.item():.3f}",
                    "Pass%": f"{pass_rate:.1f}"
                })

            # --- Validation Phase ---
            metrics = self._evaluate(val_loader, device)
            met_val = metrics.get(metric, 0)
            print(f"Epoch {epoch+1} Metrics: {metrics}")
            
            if met_val > best_metric:
                best_metric = met_val
                self.save(output_path)
                print(f"*** New best model saved with {metric}: {best_metric:.4f} ***")

    def _evaluate(self, val_loader, device):
        self.model.eval()
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                all_logits.append(outputs.logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
        logits = np.concatenate(all_logits, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        eval_pred = (logits, labels)
        metrics = self.compute_metrics(eval_pred)
        return metrics

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        assert self.model is not None and self.tokenizer is not None, "Model not loaded. Call load() first."
        self.model.eval()
        device = get_device()
        self.model.to(device)

        results = []
        for _, row in data.iterrows():
            tokens = ast.literal_eval(row["tokens"]) if isinstance(row["tokens"], str) else row["tokens"]

            # Tokenize with sliding-window segmentation so that sequences longer
            # than max_length sub-tokens are split into overlapping chunks of at most
            # max_length tokens with a 128-token overlap between adjacent chunks.
            encoding = self.tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=self.max_length,
                return_overflowing_tokens=True,
                stride=128,
                padding=True,
                return_tensors="pt",
            )

            word_preds: dict = {}
            num_chunks = encoding["input_ids"].shape[0]

            for chunk_idx in range(num_chunks):
                # Run each chunk separately to avoid out-of-memory issues.
                chunk_input_ids = encoding["input_ids"][chunk_idx : chunk_idx + 1].to(device)
                chunk_attention_mask = encoding["attention_mask"][chunk_idx : chunk_idx + 1].to(device)

                with torch.no_grad():
                    logits = self.model(
                        input_ids=chunk_input_ids,
                        attention_mask=chunk_attention_mask,
                    ).logits

                predictions = torch.argmax(logits, dim=-1).squeeze().tolist()
                if not isinstance(predictions, list):
                    predictions = [predictions]

                word_ids = encoding.word_ids(batch_index=chunk_idx)
                for idx, word_id in enumerate(word_ids):
                    if word_id is None or idx >= len(predictions):
                        continue
                    pred_label = self.ID2LABEL[predictions[idx]]
                    if word_id not in word_preds:
                        word_preds[word_id] = pred_label
                    else:
                        word_preds[word_id] = self._merge_word_prediction(
                            word_preds[word_id], pred_label
                        )

            ner_tags = self.autocorrect_labels([word_preds.get(i, "O") for i in range(len(tokens))])
            labels = [self.bio_to_label(tag) for tag in ner_tags]
            results.append({"ID": row["ID"], "tokens": row["tokens"], "labels": str(labels), "ner_tags": str(ner_tags)})

        return pd.DataFrame(results)

    def save(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str) -> "BertFixMatchModel":
        instance = cls(model_name=path)
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        instance.model = AutoModelForTokenClassification.from_pretrained(path)
        return instance
