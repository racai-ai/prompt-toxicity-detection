"""BERT fine-tuned NER model."""

import ast
import os

import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer as HFTrainer,
    TrainingArguments,
)

from models.base import BaseNERModel, TrainEvalCallback
from models.losses import MulticlassDiceLoss
from models.pretrained import is_local_model_path, resolve_model_name_or_path
from utils.device import get_device, get_training_arguments_device_kwargs, log_device


class DiceCETrainer(HFTrainer):
    """HuggingFace Trainer that combines CrossEntropy and Dice loss."""

    def __init__(self, *args, alpha_ce: float = 0.5, alpha_dice: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self._dice_loss = MulticlassDiceLoss(ignore_index=-100)
        self.alpha_ce = alpha_ce
        self.alpha_dice = alpha_dice

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        ce_loss = outputs.loss
        if labels is not None:
            dice_loss = self._dice_loss(outputs.logits, labels)
            loss = self.alpha_ce * ce_loss + self.alpha_dice * dice_loss
        else:
            loss = ce_loss
        return (loss, outputs) if return_outputs else loss


class BertFinetunedModel(BaseNERModel):
    """Standard BERT model fine-tuned for token classification (NER)."""

    DEFAULT_MODEL_NAME = "microsoft/deberta-v3-large"
    DEFAULT_MAX_LENGTH: int = 512
    DEFAULT_BATCH_SIZE: int = 16
    TOKEN_CLASSIFICATION_HEAD_ATTRS = ("classifier", "score")

    # Subdirectory names and config filename used by the dual-head seq-cls path.
    SEQ_CLS_CONFIG = "seq_cls_config.json"
    LABEL_HEAD_DIR = "label_head"
    GRANULAR_HEAD_DIR = "granular_head"

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = resolve_model_name_or_path(model_name)
        self.tokenizer = None
        self.model = None
        self.max_length: int = self.DEFAULT_MAX_LENGTH
        self.batch_size: int = self.DEFAULT_BATCH_SIZE
        # Populated by the sequence-classification path (train_seq_cls / load_seq_cls).
        self.granular_model = None
        self._label_classes: list | None = None
        self._granular_classes: list | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_token_classification_head(self, model) -> None:
        """Replace a loaded token-classification head with a fresh NER head.

        Parameters
        ----------
        model:
            Hugging Face token-classification model whose prediction head should
            be reinitialized for the repository's NER label set.

        Raises
        ------
        AttributeError
            If the model does not expose a supported token-classification head
            attribute.
        """
        num_labels = len(self.LABEL_LIST)
        # Hugging Face token-classification architectures commonly expose the
        # prediction head as either ``classifier`` (BERT/DeBERTa family) or
        # ``score`` (some encoder-decoder wrappers). Reinitialize whichever one
        # exists so local checkpoints always get a fresh task-specific head.
        for attr_name in self.TOKEN_CLASSIFICATION_HEAD_ATTRS:
            head = getattr(model, attr_name, None)
            if isinstance(head, torch.nn.Linear):
                new_head = torch.nn.Linear(
                    head.in_features,
                    num_labels,
                    bias=head.bias is not None,
                )
                if hasattr(model, "_init_weights"):
                    model._init_weights(new_head)
                setattr(model, attr_name, new_head)
                model.num_labels = num_labels
                model.config.num_labels = num_labels
                model.config.id2label = self.ID2LABEL
                model.config.label2id = self.LABEL2ID
                return
        raise AttributeError(
            "Could not find a token classification head to replace "
            f"(checked attributes: {', '.join(self.TOKEN_CLASSIFICATION_HEAD_ATTRS)})."
        )

    def _build_tokenizer_and_model(self):
        is_local_pretrained = is_local_model_path(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model_kwargs = {
            "num_labels": len(self.LABEL_LIST),
            "id2label": self.ID2LABEL,
            "label2id": self.LABEL2ID,
        }
        if is_local_pretrained:
            model_kwargs["ignore_mismatched_sizes"] = True
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name, **model_kwargs)
        if is_local_pretrained:
            self._reset_token_classification_head(self.model)

    def _tokenize_and_align(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            max_length=self.max_length,
            is_split_into_words=True,
        )
        all_labels = []
        for i, ner_tags in enumerate(examples["ner_tags"]):
            # Convert the string tags to integer IDs first
            labels = [self.LABEL2ID.get(tag, 0) for tag in ner_tags]
            word_ids = tokenized_inputs.word_ids(batch_index=i)

            new_labels = []
            current_word = None
            for word_id in word_ids:
                if word_id != current_word:
                    # Start of a new word!
                    current_word = word_id
                    label = -100 if word_id is None else labels[word_id]
                    new_labels.append(label)
                elif word_id is None:
                    # Special token
                    new_labels.append(-100)
                else:
                    # Same word as previous token
                    label = labels[word_id]
                    # If the label is B-XXX (odd), we change it to I-XXX (even)
                    if label % 2 == 1:
                        label += 1
                    new_labels.append(label)

            all_labels.append(new_labels)

        tokenized_inputs["labels"] = all_labels
        return tokenized_inputs

    def _df_to_hf_dataset(self, df: pd.DataFrame):
        from datasets import Dataset

        records = []
        for _, row in df.iterrows():
            tokens = ast.literal_eval(row["tokens"]) if isinstance(row["tokens"], str) else row["tokens"]
            ner_tags = ast.literal_eval(row["ner_tags"]) if isinstance(row["ner_tags"], str) else row["ner_tags"]
            records.append({"tokens": tokens, "ner_tags": ner_tags})
        return Dataset.from_list(records)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        output_path: str,
        metric: str = "f1",
        max_training_epochs: int = 50,
        alpha_ce: float = 0.5,
        alpha_dice: float = 0.5,
        **kwargs,
    ) -> None:
        self.max_length = kwargs.get("max_sequence_len", self.DEFAULT_MAX_LENGTH)
        batch_size = kwargs.get("batch_size", self.DEFAULT_BATCH_SIZE)
        log_device()
        self._build_tokenizer_and_model()

        train_dataset = self._df_to_hf_dataset(train_data).map(
            self._tokenize_and_align, batched=True, remove_columns=["tokens", "ner_tags"]
        )
        val_dataset = self._df_to_hf_dataset(val_data).map(
            self._tokenize_and_align, batched=True, remove_columns=["tokens", "ner_tags"]
        )

        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        training_args = TrainingArguments(
            output_dir=output_path,
            num_train_epochs=max_training_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model=metric,
            greater_is_better=True,
            save_total_limit=2,
            logging_dir=os.path.join(output_path, "logs"),
            **get_training_arguments_device_kwargs(),
        )

        train_eval_cb = TrainEvalCallback(train_dataset, val_dataset)
        trainer = DiceCETrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[train_eval_cb],
            alpha_ce=alpha_ce,
            alpha_dice=alpha_dice,
        )
        train_eval_cb.trainer = trainer
        trainer.train()
        self.save(output_path)

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
    def load(cls, path: str) -> "BertFinetunedModel":
        path = resolve_model_name_or_path(path)
        instance = cls(model_name=path)
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        instance.model = AutoModelForTokenClassification.from_pretrained(path)
        return instance

    # ------------------------------------------------------------------
    # Sequence-level classification (label + granular_label)
    # ------------------------------------------------------------------

    def _train_single_seq_cls_head(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        output_path: str,
        label_col: str,
        classes: list,
        max_training_epochs: int,
        batch_size: int | None = None,
        max_sequence_len: int | None = None,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        warm_start_path: str | None = None,
    ) -> None:
        """Train one sequence-classification head and save it to *output_path*.

        Parameters
        ----------
        train_data / val_data:
            DataFrames that must contain a ``prompt`` column and *label_col*.
        output_path:
            Directory where the trained model and tokenizer are saved.
        label_col:
            Name of the target column (``"label"`` or ``"granular_label"``).
        classes:
            Sorted list of unique class strings.  Used to build the label→id
            mapping.
        max_training_epochs:
            Number of training epochs.
        batch_size:
            Mini-batch size for training.  Defaults to :attr:`DEFAULT_BATCH_SIZE`.
        max_sequence_len:
            Maximum input sequence length.  Defaults to :attr:`DEFAULT_MAX_LENGTH`.
        learning_rate:
            Peak learning rate (default: 2e-5).
        warmup_ratio:
            Fraction of training steps used for linear warmup (default: 0.1).
        weight_decay:
            L2 regularisation coefficient (default: 0.01).
        warm_start_path:
            Optional path to a trained sequence classification head to initialize weights from.
        """
        from datasets import Dataset
        from transformers import DataCollatorWithPadding

        effective_batch = batch_size if batch_size is not None else self.DEFAULT_BATCH_SIZE
        effective_max_len = max_sequence_len if max_sequence_len is not None else self.DEFAULT_MAX_LENGTH

        num_labels = len(classes)
        label2id = {c: i for i, c in enumerate(classes)}

        if warm_start_path is not None:
            print(f"[hner] Warm starting '{label_col}' head from {warm_start_path}")
            tokenizer = AutoTokenizer.from_pretrained(warm_start_path)
            model = AutoModelForSequenceClassification.from_pretrained(
                warm_start_path, num_labels=num_labels, ignore_mismatched_sizes=True
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=num_labels
            )

        def _build_records(df):
            return [
                {"prompt": str(row["prompt"]), "label": label2id[str(row[label_col])]}
                for _, row in df.iterrows()
            ]

        def _tokenize(examples):
            return tokenizer(
                examples["prompt"],
                truncation=True,
                max_length=effective_max_len,
                padding=False,
            )

        train_dataset = Dataset.from_list(_build_records(train_data)).map(
            _tokenize, batched=True, remove_columns=["prompt"]
        )
        val_dataset = Dataset.from_list(_build_records(val_data)).map(
            _tokenize, batched=True, remove_columns=["prompt"]
        )

        def _compute_metrics(eval_pred):
            import numpy as np
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

            logits, labels = eval_pred
            preds = np.argmax(logits, axis=1)
            return {
                "accuracy": accuracy_score(labels, preds),
                "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
                "precision": precision_score(labels, preds, average="macro", zero_division=0),
                "recall": recall_score(labels, preds, average="macro", zero_division=0),
            }

        data_collator = DataCollatorWithPadding(tokenizer)
        training_args = TrainingArguments(
            output_dir=output_path,
            num_train_epochs=max_training_epochs,
            per_device_train_batch_size=effective_batch,
            per_device_eval_batch_size=effective_batch,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            save_total_limit=2,
            logging_dir=os.path.join(output_path, "logs"),
            **get_training_arguments_device_kwargs(),
        )

        trainer = HFTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=_compute_metrics,
        )
        trainer.train()
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

    def train_seq_cls(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        output_path: str,
        max_training_epochs: int = 50,
        **kwargs,
    ) -> None:
        """Train sequence classifiers for ``label`` and, if present, ``granular_label``.

        Both target columns are handled as string labels (e.g. ``"safe"`` /
        ``"unsafe"`` for ``label``).  The sorted unique values are used as the
        class list; the mapping is persisted to ``seq_cls_config.json`` so that
        inference can decode integer predictions back to their original strings.

        Each target gets its own subdirectory (``label_head/`` /
        ``granular_head/``) inside *output_path*.
        """
        import json

        batch_size = kwargs.get("batch_size", None)
        max_sequence_len = kwargs.get("max_sequence_len", None)
        learning_rate = kwargs.get("learning_rate", 2e-5)
        warmup_ratio = kwargs.get("warmup_ratio", 0.1)
        weight_decay = kwargs.get("weight_decay", 0.01)
        warm_pipeline = kwargs.get("warm_pipeline", False)

        log_device()
        os.makedirs(output_path, exist_ok=True)

        # Train binary label head
        label_classes = sorted(str(v) for v in train_data["label"].unique())
        label_head_path = os.path.join(output_path, self.LABEL_HEAD_DIR)
        print(f"[hner] Training 'label' head ({len(label_classes)} classes: {label_classes})")
        self._train_single_seq_cls_head(
            train_data, val_data, label_head_path, "label", label_classes, max_training_epochs,
            batch_size=batch_size, max_sequence_len=max_sequence_len,
            learning_rate=learning_rate, warmup_ratio=warmup_ratio, weight_decay=weight_decay,
        )

        config: dict = {"label_classes": label_classes, "has_granular": False}

        # Train granular_label head if the column is present
        if "granular_label" in train_data.columns:
            granular_classes = sorted(str(v) for v in train_data["granular_label"].unique())
            granular_head_path = os.path.join(output_path, self.GRANULAR_HEAD_DIR)
            print(
                f"[hner] Training 'granular_label' head "
                f"({len(granular_classes)} classes: {granular_classes})"
            )
            warm_start_path = label_head_path if warm_pipeline else None
            self._train_single_seq_cls_head(
                train_data, val_data, granular_head_path,
                "granular_label", granular_classes, max_training_epochs,
                batch_size=batch_size, max_sequence_len=max_sequence_len,
                learning_rate=learning_rate, warmup_ratio=warmup_ratio, weight_decay=weight_decay,
                warm_start_path=warm_start_path,
            )
            config["granular_classes"] = granular_classes
            config["has_granular"] = True

        # Persist label-encoding config for inference
        with open(
            os.path.join(output_path, self.SEQ_CLS_CONFIG), "w", encoding="utf-8"
        ) as fh:
            json.dump(config, fh, ensure_ascii=False, indent=2)

    def _run_seq_cls_inference(self, model, prompt: str, device) -> int:
        """Run a single forward pass and return the argmax class index."""
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        return torch.argmax(logits, dim=-1).item()

    def predict_seq_cls(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict ``label`` and, if a granular head is loaded, ``granular_label``.

        Output columns ``prediction_label`` (and ``prediction_granular_label``
        when the granular head is available) are added alongside all original
        input columns so that ground-truth columns are preserved for evaluation.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_seq_cls() first.")
        device = get_device()
        self.model.eval()
        self.model.to(device)
        if self.granular_model is not None:
            self.granular_model.eval()
            self.granular_model.to(device)

        results = []
        for _, row in data.iterrows():
            prompt = str(row["prompt"])
            label_idx = self._run_seq_cls_inference(self.model, prompt, device)
            label_pred = (
                self._label_classes[label_idx]
                if self._label_classes is not None
                else label_idx
            )

            # Pass through all original columns first, then add predictions.
            result: dict = {col: row[col] for col in data.columns}
            result["prediction_label"] = label_pred

            if self.granular_model is not None:
                gran_idx = self._run_seq_cls_inference(self.granular_model, prompt, device)
                gran_pred = (
                    self._granular_classes[gran_idx]
                    if self._granular_classes is not None
                    else gran_idx
                )
                result["prediction_granular_label"] = gran_pred

            results.append(result)

        return pd.DataFrame(results)

    @classmethod
    def load_seq_cls(cls, path: str) -> "BertFinetunedModel":
        """Load a previously saved dual-head (or legacy single-head) seq-cls model.

        Reads ``seq_cls_config.json`` to discover the label classes and whether
        a granular head was trained.  If the config is absent (models saved
        before this feature) the label head is loaded from *path* directly and
        predictions will be raw integer indices.
        """
        import json

        path = resolve_model_name_or_path(path)
        config_path = os.path.join(path, cls.SEQ_CLS_CONFIG)
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as fh:
                config = json.load(fh)
        else:
            # Legacy model: single head saved directly in path.
            config = {"has_granular": False}

        has_granular = config.get("has_granular", False)

        # Label head may live in a subdirectory (new style) or at root (legacy).
        label_head_path = os.path.join(path, cls.LABEL_HEAD_DIR)
        if not os.path.isdir(label_head_path):
            label_head_path = path

        instance = cls(model_name=path)
        instance._label_classes = config.get("label_classes")
        instance._granular_classes = config.get("granular_classes") if has_granular else None
        instance.granular_model = None

        instance.tokenizer = AutoTokenizer.from_pretrained(label_head_path)
        instance.model = AutoModelForSequenceClassification.from_pretrained(label_head_path)

        if has_granular:
            granular_head_path = os.path.join(path, cls.GRANULAR_HEAD_DIR)
            instance.granular_model = AutoModelForSequenceClassification.from_pretrained(
                granular_head_path
            )

        return instance
