"""BERT with Adapter NER model using the `adapters` library."""

import ast
import os

import pandas as pd
import torch
from transformers import AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments

from models.base import BaseNERModel, TrainEvalCallback
from models.losses import MulticlassDiceLoss
from models.pretrained import (
    is_local_model_path,
    resolve_adapter_base_model_name_or_path,
    resolve_model_name_or_path,
)
from utils.device import get_device, get_training_arguments_device_kwargs, log_device


def _fix_adapter_config_bug(path: str) -> None:
    """Fix adapter hub bug where base model is stored as model_name instead of base_model_name_or_path."""
    import json
    config_file = os.path.join(path, "adapter_config.json")
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                adapter_config = json.load(f)
            if "model_name" in adapter_config:
                adapter_config["base_model_name_or_path"] = adapter_config.pop("model_name")
                with open(config_file, "w", encoding="utf-8") as f:
                    json.dump(adapter_config, f, indent=2, ensure_ascii=False)
                print(f"[hner] Fixed base model name key in {config_file}")
        except Exception as e:
            print(f"[hner] Warning: could not fix base model name key in {config_file}: {e}")


class DiceCEAdapterTrainer:
    """Mixin that adds Dice loss to the AdapterTrainer's compute_loss."""

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


_dice_adapter_trainer_cls = None


def _get_dice_adapter_trainer():
    """Return a cached concrete trainer combining DiceCEAdapterTrainer and AdapterTrainer."""
    global _dice_adapter_trainer_cls
    if _dice_adapter_trainer_cls is None:
        from adapters import AdapterTrainer

        class _DiceCEAdapterTrainerImpl(DiceCEAdapterTrainer, AdapterTrainer):
            pass

        _dice_adapter_trainer_cls = _DiceCEAdapterTrainerImpl
    return _dice_adapter_trainer_cls


class BertAdapterModel(BaseNERModel):
    """BERT model with a task-specific adapter for NER.

    Only the adapter weights (plus a classification head) are updated during
    training, while the pre-trained BERT backbone is kept frozen.
    """

    DEFAULT_MODEL_NAME = "microsoft/deberta-v3-large"
    ADAPTER_NAME = "ner_adapter"
    DEFAULT_MAX_LENGTH: int = 512
    DEFAULT_BATCH_SIZE: int = 32

    # Subdirectory names and config filename used by the dual-head seq-cls path.
    SEQ_CLS_CONFIG = "seq_cls_config.json"
    LABEL_HEAD_DIR = "label_head"
    GRANULAR_HEAD_DIR = "granular_head"

    # Adapter name used *within* each saved seq-cls head directory.
    _SEQ_CLS_ADAPTER_NAME = "seq_cls_adapter"

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

    def _build_tokenizer_and_model(self):
        import adapters
        from adapters import AdapterConfig, AutoAdapterModel

        backbone_model_name = resolve_adapter_base_model_name_or_path(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_model_name)
        device = get_device()
        self.model = AutoAdapterModel.from_pretrained(backbone_model_name).to(device)

        if is_local_model_path(self.model_name):
            local_adapter_path = resolve_model_name_or_path(self.model_name)
            self.model.load_adapter(local_adapter_path, load_as=self.ADAPTER_NAME)
            if self.ADAPTER_NAME in self.model.heads:
                # Reuse the saved adapter weights but rebuild the tagging head
                # for this task so label dimensions and mappings are always fresh.
                self.model.delete_head(self.ADAPTER_NAME)
        else:
            adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=6)
            self.model.add_adapter(self.ADAPTER_NAME, config=adapter_config)
        self.model.add_tagging_head(
            self.ADAPTER_NAME,
            num_labels=len(self.LABEL_LIST),
            id2label=self.ID2LABEL,
        )
        self.model.train_adapter(self.ADAPTER_NAME)

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
            label_ids = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(self.LABEL2ID.get(ner_tags[word_idx], 0))
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            all_labels.append(label_ids)
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
        TrainerCls = _get_dice_adapter_trainer()

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
        trainer = TrainerCls(
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
        self.model.set_active_adapters(self.ADAPTER_NAME)
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
        self.model.save_adapter(path, self.ADAPTER_NAME)
        self.tokenizer.save_pretrained(path)
        # Also save the base model config so we can reload
        self.model.save_pretrained(path)
        _fix_adapter_config_bug(path)

    @classmethod
    def load(cls, path: str) -> "BertAdapterModel":
        import adapters
        from adapters import AutoAdapterModel

        path = resolve_model_name_or_path(path)
        instance = cls(model_name=path)
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        instance.model = AutoAdapterModel.from_pretrained(path)
        instance.model.load_adapter(path, load_as=cls.ADAPTER_NAME)
        instance.model.set_active_adapters(cls.ADAPTER_NAME)
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
        warm_start_path: str | None = None,
    ) -> None:
        """Train one adapter-based sequence-classification head and save it.

        Parameters
        ----------
        train_data / val_data:
            DataFrames that must contain a ``prompt`` column and *label_col*.
        output_path:
            Directory where the adapter weights, backbone config, and tokenizer
            are saved.
        label_col:
            Name of the target column (``"label"`` or ``"granular_label"``).
        classes:
            Sorted list of unique class strings.
        max_training_epochs:
            Number of training epochs.
        batch_size:
            Mini-batch size for training.  Defaults to :attr:`DEFAULT_BATCH_SIZE`.
        max_sequence_len:
            Maximum input sequence length.  Defaults to :attr:`DEFAULT_MAX_LENGTH`.
        warm_start_path:
            Optional path to a trained sequence classification head to initialize weights from.
        """
        import adapters
        from adapters import AdapterConfig, AutoAdapterModel
        from datasets import Dataset
        from transformers import DataCollatorWithPadding

        effective_batch = batch_size if batch_size is not None else self.DEFAULT_BATCH_SIZE
        effective_max_len = max_sequence_len if max_sequence_len is not None else self.DEFAULT_MAX_LENGTH

        num_labels = len(classes)
        label2id = {c: i for i, c in enumerate(classes)}
        is_binary = num_labels == 2

        device = get_device()
        backbone_model_name = resolve_adapter_base_model_name_or_path(self.model_name)
        print(backbone_model_name)
        if warm_start_path is not None:
            print(f"[hner] Warm starting '{label_col}' head from {warm_start_path}")
            tokenizer = AutoTokenizer.from_pretrained(warm_start_path)
            model = AutoAdapterModel.from_pretrained(backbone_model_name).to(device)
            model.load_adapter(warm_start_path, load_as=self._SEQ_CLS_ADAPTER_NAME)
            if self._SEQ_CLS_ADAPTER_NAME in model.heads:
                model.delete_head(self._SEQ_CLS_ADAPTER_NAME)
            model.add_classification_head(self._SEQ_CLS_ADAPTER_NAME, num_labels=num_labels)
            model.train_adapter(self._SEQ_CLS_ADAPTER_NAME)
        else:
            tokenizer = AutoTokenizer.from_pretrained(backbone_model_name)
            model = AutoAdapterModel.from_pretrained(backbone_model_name).to(device)
            if is_local_model_path(self.model_name):
                local_adapter_path = resolve_model_name_or_path(self.model_name)
                model.load_adapter(local_adapter_path, load_as=self._SEQ_CLS_ADAPTER_NAME)
                if self._SEQ_CLS_ADAPTER_NAME in model.heads:
                    model.delete_head(self._SEQ_CLS_ADAPTER_NAME)
            else:
                adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=6)
                model.add_adapter(self._SEQ_CLS_ADAPTER_NAME, config=adapter_config)
            model.add_classification_head(self._SEQ_CLS_ADAPTER_NAME, num_labels=num_labels)
            model.train_adapter(self._SEQ_CLS_ADAPTER_NAME)

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
            avg = "binary" if is_binary else "macro"
            return {
                "accuracy": accuracy_score(labels, preds),
                "f1": f1_score(labels, preds, average=avg, zero_division=0),
                "precision": precision_score(labels, preds, average=avg, zero_division=0),
                "recall": recall_score(labels, preds, average=avg, zero_division=0),
            }

        TrainerCls = _get_dice_adapter_trainer()
        data_collator = DataCollatorWithPadding(tokenizer)
        training_args = TrainingArguments(
            output_dir=output_path,
            num_train_epochs=max_training_epochs,
            # Adapter models freeze the backbone weights and only train the adapter
            # parameters, so the same batch size is used for both train and eval.
            per_device_train_batch_size=effective_batch,
            per_device_eval_batch_size=effective_batch,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            greater_is_better=True,
            save_total_limit=2,
            metric_for_best_model="f1",
            logging_dir=os.path.join(output_path, "logs"),
            **get_training_arguments_device_kwargs(),
        )

        trainer = TrainerCls(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=_compute_metrics,
            # Dice loss is designed for token-level imbalanced label distributions;
            # use standard cross-entropy only for sequence classification.
            alpha_ce=1.0,
            alpha_dice=0.0,
        )
        trainer.train()
        model.save_adapter(output_path, self._SEQ_CLS_ADAPTER_NAME)
        tokenizer.save_pretrained(output_path)
        model.save_pretrained(output_path)
        _fix_adapter_config_bug(output_path)

    def train_seq_cls(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        output_path: str,
        max_training_epochs: int = 50,
        **kwargs,
    ) -> None:
        """Train adapter seq classifiers for ``label`` and, if present, ``granular_label``.

        Each target gets its own subdirectory (``label_head/`` /
        ``granular_head/``) inside *output_path*.  A ``seq_cls_config.json``
        file records the sorted class lists so that inference can decode integer
        predictions back to their original string labels.
        """
        import json

        batch_size = kwargs.get("batch_size", None)
        max_sequence_len = kwargs.get("max_sequence_len", None)
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
        self.model.set_active_adapters(self._SEQ_CLS_ADAPTER_NAME)
        self.model.eval()
        self.model.to(device)
        if self.granular_model is not None:
            self.granular_model.set_active_adapters(self._SEQ_CLS_ADAPTER_NAME)
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
    def load_seq_cls(cls, path: str) -> "BertAdapterModel":
        """Load a previously saved dual-head (or legacy single-head) seq-cls model.

        Reads ``seq_cls_config.json`` to discover the label classes and whether
        a granular head was trained.  If the config is absent (models saved
        before this feature) the label head is loaded from *path* directly and
        predictions will be raw integer indices.
        """
        import json
        import adapters
        from adapters import AutoAdapterModel

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
        instance.model = AutoAdapterModel.from_pretrained(label_head_path)
        instance.model.load_adapter(label_head_path, load_as=cls._SEQ_CLS_ADAPTER_NAME)
        instance.model.set_active_adapters(cls._SEQ_CLS_ADAPTER_NAME)

        if has_granular:
            granular_head_path = os.path.join(path, cls.GRANULAR_HEAD_DIR)
            instance.granular_model = AutoAdapterModel.from_pretrained(granular_head_path)
            instance.granular_model.load_adapter(
                granular_head_path, load_as=cls._SEQ_CLS_ADAPTER_NAME
            )
            instance.granular_model.set_active_adapters(cls._SEQ_CLS_ADAPTER_NAME)

        return instance
