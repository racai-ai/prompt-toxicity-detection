import ast
import os

import pandas as pd
import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer as HFTrainer,
    TrainingArguments,
    TrainerCallback,
)
from transformers.modeling_outputs import TokenClassifierOutput
#from torchcrf import CRF

# --- NEW PEFT IMPORTS ---
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig

from models.base import BaseNERModel
from models.losses import MulticlassDiceLoss
from utils.device import get_device, get_training_arguments_device_kwargs, log_device


# ------------------------------------------------------------------
# 1. The Custom Hugging Face Callback for the "Jigsaw" Schedule
# ------------------------------------------------------------------
class CyclicalKLAnnealingCallback(TrainerCallback):
    """
    Gradually ramps up the KL Divergence weight (beta) in cycles.
    Looks like a jigsaw tooth over the course of training.
    """

    def __init__(self, num_cycles=4, max_beta=1e-1, warmup_ratio=0.5):
        self.num_cycles = num_cycles
        self.max_beta = max_beta
        self.warmup_ratio = warmup_ratio  # Proportion of the cycle spent ramping up

    def on_step_begin(self, args, state, control, model, **kwargs):
        if state.max_steps <= 0:
            return

        cycle_length = state.max_steps / self.num_cycles
        current_step_in_cycle = state.global_step % cycle_length
        tau = current_step_in_cycle / cycle_length

        if tau <= self.warmup_ratio:
            current_beta = self.max_beta * (tau / self.warmup_ratio)
        else:
            current_beta = self.max_beta

        if hasattr(model, "module"):
            model.module.beta = current_beta
        else:
            model.beta = current_beta

    def on_log(self, args, state, control, model, logs=None, **kwargs):
        if logs is not None:
            actual_model = model.module if hasattr(model, "module") else model
            logs["kl_beta"] = actual_model.beta


# ------------------------------------------------------------------
# 2. The VIB Model
# ------------------------------------------------------------------
class VIBTokenClassificationModel(nn.Module):
    """Custom model adding a Variational Information Bottleneck + optional CRF to a Hugging Face encoder."""

    def __init__(self, model_name, num_labels, vib_dim=256, beta=1e-2, use_crf=False, apply_lora=True,
                 alpha_ce: float = 0.5, alpha_dice: float = 0.5):
        super().__init__()
        self.num_labels = num_labels
        self.beta = beta
        self.use_crf = use_crf
        self.alpha_ce = alpha_ce
        self.alpha_dice = alpha_dice

        # 1. Base Encoder
        self.encoder = AutoModel.from_pretrained(model_name)

        # --- LORA INTEGRATION ---
        if apply_lora:
            lora_config = LoraConfig(
                r=1,  # Rank of the adapter matrices
                lora_alpha=2,  # Scaling factor
                target_modules="all-linear",  # Targets Q, K, V, and MLP dense layers automatically
                lora_dropout=0.2,
                bias="none",
            )
            self.encoder = get_peft_model(self.encoder, lora_config)
            print("\n--- LoRA Trainable Parameters ---")
            self.encoder.print_trainable_parameters()

        hidden_dim = self.encoder.config.hidden_size

        # 2. VIB Projection Layers (Fully Trainable)
        self.hidden2mean = nn.Linear(hidden_dim, vib_dim)
        self.hidden2logv = nn.Linear(hidden_dim, vib_dim)
        self.dropout = nn.Dropout(0.3)

        # 3. Classification Head & Optional CRF (Fully Trainable)
        self.classifier = nn.Linear(vib_dim, num_labels)
        if self.use_crf:
            self.crf = CRF(num_tags=num_labels, batch_first=True)
        else:
            self.crf = None
        self.dice_loss = MulticlassDiceLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        kwargs.pop("return_loss", None)

        outputs = self.encoder(input_ids, attention_mask=attention_mask, **kwargs)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        # --- VIB Reparameterization ---
        mu = self.hidden2mean(sequence_output)
        logv = self.hidden2logv(sequence_output)

        if self.training:
            std = torch.exp(0.5 * logv)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu

        logits = self.classifier(z)

        # --- Loss Calculation ---
        loss = None
        if labels is not None:
            kl_loss_raw = -0.5 * torch.sum(1 + logv - mu.pow(2) - logv.exp(), dim=-1)
            active_tokens = labels.view(-1) != -100
            active_kl = kl_loss_raw.view(-1)[active_tokens]
            kl_loss = active_kl.mean() if active_kl.numel() > 0 else torch.tensor(0.0).to(logits.device)

            if self.use_crf:
                crf_mask = attention_mask.bool()
                safe_labels = labels.clone()
                safe_labels[safe_labels == -100] = 0
                main_loss = -self.crf(logits, safe_labels, mask=crf_mask, reduction='mean')
            else:
                loss_fct = nn.CrossEntropyLoss()
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                main_loss = self.alpha_ce * loss_fct(active_logits, active_labels) + self.alpha_dice * self.dice_loss(logits, labels)

            loss = main_loss + (self.beta * kl_loss)

        if not self.training and self.use_crf:
            decoded_tags = self.crf.decode(logits, mask=attention_mask.bool())
            fake_logits = torch.zeros_like(logits)
            for i, tags in enumerate(decoded_tags):
                for j, tag in enumerate(tags):
                    fake_logits[i, j, tag] = 10000.0
            logits = fake_logits

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# ------------------------------------------------------------------
# 3. The NER Wrapper
# ------------------------------------------------------------------
class BertVariational(BaseNERModel):
    """VIB-enhanced BERT model fine-tuned for token classification (NER)."""

    DEFAULT_MODEL_NAME = "RashidNLP/NER-Deberta"
    DEFAULT_MAX_LENGTH: int = 512
    DEFAULT_BATCH_SIZE: int = 16

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, beta: float = 5e-2, vib_dim: int = 256,
                 use_crf: bool = False, alpha_ce: float = 0.5, alpha_dice: float = 0.5):
        self.model_name = model_name
        self.beta = beta
        self.vib_dim = vib_dim
        self.use_crf = use_crf
        self.alpha_ce = alpha_ce
        self.alpha_dice = alpha_dice
        self.tokenizer = None
        self.model = None
        self.max_length: int = self.DEFAULT_MAX_LENGTH
        self.batch_size: int = self.DEFAULT_BATCH_SIZE

    def _build_tokenizer_and_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = VIBTokenClassificationModel(
            model_name=self.model_name,
            num_labels=len(self.LABEL_LIST),
            vib_dim=self.vib_dim,
            beta=self.beta,
            use_crf=self.use_crf,
            apply_lora=True,
            alpha_ce=self.alpha_ce,
            alpha_dice=self.alpha_dice,
        )

    def _tokenize_and_align(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            max_length=self.max_length,
            is_split_into_words=True,
        )
        all_labels = []
        for i, ner_tags in enumerate(examples["ner_tags"]):
            labels = [self.LABEL2ID.get(tag, 0) for tag in ner_tags]
            word_ids = tokenized_inputs.word_ids(batch_index=i)

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
        """Train the BertVariational model and save the best checkpoint.

        Parameters
        ----------
        train_data:
            Training ``DataFrame``.
        val_data:
            Validation ``DataFrame`` used for model selection.
        output_path:
            Directory where the model will be saved.
        metric:
            One of ``"f1"``, ``"relaxed_f1"``, ``"avg_f1"``.
        max_training_epochs:
            Number of training epochs (default: 50).
        alpha_ce:
            Weight for the CrossEntropy loss component (default: 0.5).
        alpha_dice:
            Weight for the Dice loss component (default: 0.5).
        """
        self.alpha_ce = alpha_ce
        self.alpha_dice = alpha_dice
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
            logging_strategy="steps",
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model=metric,
            greater_is_better=True,
            save_total_limit=2,
            logging_dir=os.path.join(output_path, "logs"),
            **get_training_arguments_device_kwargs(),
            max_grad_norm=1.0,
            weight_decay=0.05,
            lr_scheduler_type="linear",
            learning_rate=2e-4

        )

        trainer = HFTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[CyclicalKLAnnealingCallback(num_cycles=1, max_beta=self.beta, warmup_ratio=0.2)],
        )
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
                    outputs = self.model.encoder(
                        chunk_input_ids,
                        attention_mask=chunk_attention_mask,
                    )
                    sequence_output = outputs[0]
                    mu = self.model.hidden2mean(sequence_output)
                    emissions = self.model.classifier(mu)

                    if self.model.use_crf:
                        crf_predictions = self.model.crf.decode(
                            emissions, mask=chunk_attention_mask.bool()
                        )
                        chunk_preds = crf_predictions[0]
                    else:
                        chunk_preds = torch.argmax(emissions, dim=-1).squeeze(0).tolist()
                        if not isinstance(chunk_preds, list):
                            chunk_preds = [chunk_preds]

                word_ids = encoding.word_ids(batch_index=chunk_idx)
                for idx, word_id in enumerate(word_ids):
                    if word_id is None or idx >= len(chunk_preds):
                        continue
                    pred_label = self.ID2LABEL[chunk_preds[idx]]
                    if word_id not in word_preds:
                        word_preds[word_id] = pred_label
                    else:
                        word_preds[word_id] = self._merge_word_prediction(
                            word_preds[word_id], pred_label
                        )

            ner_tags = self.autocorrect_labels([word_preds.get(i, "O") for i in range(len(tokens))])
            labels = [self.bio_to_label(tag) for tag in ner_tags]

            results.append({
                "ID": row["ID"],
                "tokens": row["tokens"],
                "labels": str(labels),
                "ner_tags": str(ner_tags)
            })

        return pd.DataFrame(results)

    def save(self, path: str) -> None:
        """Saves LoRA adapters and custom head weights separately."""
        os.makedirs(path, exist_ok=True)
        self.tokenizer.save_pretrained(path)

        # 1. Save LoRA adapters (saves adapter_model.safetensors/bin)
        self.model.encoder.save_pretrained(path)

        # 2. Save only the custom heads (VIB, classifier, CRF)
        # Filter out anything starting with "encoder." to prevent saving the base model
        custom_state_dict = {k: v for k, v in self.model.state_dict().items() if not k.startswith("encoder.")}
        torch.save(custom_state_dict, os.path.join(path, "vib_model.pt"))

    @classmethod
    def load(cls, path: str) -> "BertVariational":
        """Loads base model, attaches trained LoRA adapters, and loads custom heads."""
        # Detect base model from the saved LoRA configuration
        peft_config = PeftConfig.from_pretrained(path)
        base_model_name = peft_config.base_model_name_or_path

        state_dict_path = os.path.join(path, "vib_model.pt")
        state_dict = torch.load(state_dict_path, map_location=get_device())
        used_crf = any(key.startswith('crf.') for key in state_dict.keys())

        instance = cls(model_name=base_model_name, use_crf=used_crf)
        instance.tokenizer = AutoTokenizer.from_pretrained(path)

        # 1. Initialize custom architecture using the BASE model, but DO NOT apply fresh LoRA
        instance.model = VIBTokenClassificationModel(
            model_name=base_model_name,
            num_labels=len(instance.LABEL_LIST),
            use_crf=used_crf,
            apply_lora=False
        )

        # 2. Attach the trained LoRA adapters to the encoder
        instance.model.encoder = PeftModel.from_pretrained(instance.model.encoder, path)

        # 3. Load weights into the custom heads (VIB, classifier, CRF)
        # strict=False because the loaded dictionary doesn't contain encoder weights anymore
        instance.model.load_state_dict(state_dict, strict=False)
        return instance


if __name__ == "__main__":
    print("--- Testing _tokenize_and_align Subword Propagation ---")

    model = BertVariational(model_name="microsoft/deberta-v3-small", use_crf=True)

    model.LABEL_LIST = ["O", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    model.LABEL2ID = {tag: i for i, tag in enumerate(model.LABEL_LIST)}
    model.ID2LABEL = {i: tag for i, tag in enumerate(model.LABEL_LIST)}

    model._build_tokenizer_and_model()

    test_examples = {
        "tokens": [["HuggingFace", "Inc.", "is", "in", "Massachusetts", "."]],
        "ner_tags": [["B-ORG", "I-ORG", "O", "O", "B-LOC", "O"]]
    }

    print("\n[Original Input]")
    for word, tag in zip(test_examples["tokens"][0], test_examples["ner_tags"][0]):
        print(f"{word:<15} -> {tag}")

    aligned_outputs = model._tokenize_and_align(test_examples)

    input_ids = aligned_outputs["input_ids"][0]
    label_ids = aligned_outputs["labels"][0]
    tokens = model.tokenizer.convert_ids_to_tokens(input_ids)

    print("\n[Tokenized & Aligned Output]")
    print(f"{'SUBWORD TOKEN':<18} | {'WORD ID':<8} | {'RAW ID':<8} | {'ASSIGNED TAG'}")
    print("-" * 60)

    word_ids = aligned_outputs.word_ids(batch_index=0)

    for token, word_id, label_id in zip(tokens, word_ids, label_ids):
        if label_id == -100:
            label_str = "-100 (IGNORE)"
        else:
            label_str = model.ID2LABEL.get(label_id, "ERROR")

        print(f"{token:<18} | {str(word_id):<8} | {str(label_id):<8} | {label_str}")
