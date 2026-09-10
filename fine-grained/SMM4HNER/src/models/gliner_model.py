"""GLiNER-based NER model (zero-shot or fine-tuned)."""

import ast
import os
import re

import pandas as pd

from models.base import BaseNERModel


class GlinerModel(BaseNERModel):
    """NER model backed by GLiNER.

    In zero-shot mode (no ``model_path`` provided) the default pre-trained
    GLiNER checkpoint is used directly.  When a fine-tuned checkpoint is
    supplied, that checkpoint is loaded instead and can be further fine-tuned
    with :py:meth:`train`.

    The model uses an extended set of entity types (``GLINER_ENTITY_TYPES``)
    that GLiNER can recognise directly, and a ``ENTITY_TO_IMPACT`` translation
    layer that maps each granular type back to one of the two shared-task
    classes (``ClinicalImpacts`` / ``SocialImpacts``).  Only spans from
    first-person sentences are kept when ``require_first_person=True``.
    """

    DEFAULT_MODEL_NAME = "urchade/gliner_medium-v2.1"
    DEFAULT_MAX_LENGTH: int = 512
    DEFAULT_BATCH_SIZE: int = 8

    # ------------------------------------------------------------------
    # Extended entity types used for zero-shot GLiNER prediction
    # ------------------------------------------------------------------

    # Granular entity labels understood by GLiNER
    GLINER_ENTITY_TYPES: list[str] = [
        # ClinicalImpacts sub-types
        "symptom",
        "withdrawal symptom",
        "addiction",
        "substance",
        "drug use",
        "abuse",
        "overdose",
        "treatment",
        "rehabilitation",
        "mental health condition",
        # SocialImpacts sub-types
        "homelessness",
        "legal issue",
        "arrest",
        "imprisonment",
        "relationship breakdown",
        "family estrangement",
        "social isolation",
        "job loss",
        "financial hardship",
    ]

    # Translation layer: granular GLiNER label → shared-task class
    ENTITY_TO_IMPACT: dict[str, str] = {
        # ClinicalImpacts
        "symptom": "ClinicalImpacts",
        "withdrawal symptom": "ClinicalImpacts",
        "addiction": "ClinicalImpacts",
        "substance": "ClinicalImpacts",
        "drug use": "ClinicalImpacts",
        "abuse": "ClinicalImpacts",
        "overdose": "ClinicalImpacts",
        "treatment": "ClinicalImpacts",
        "rehabilitation": "ClinicalImpacts",
        "mental health condition": "ClinicalImpacts",
        # SocialImpacts
        "homelessness": "SocialImpacts",
        "legal issue": "SocialImpacts",
        "arrest": "SocialImpacts",
        "imprisonment": "SocialImpacts",
        "relationship breakdown": "SocialImpacts",
        "family estrangement": "SocialImpacts",
        "social isolation": "SocialImpacts",
        "job loss": "SocialImpacts",
        "financial hardship": "SocialImpacts",
    }

    # First-person keywords used for first-person sentence detection
    _FIRST_PERSON_PATTERN = re.compile(
        r"\b(i|i'm|i've|i'll|i'd|i'ma|my|me|myself|mine|we|we're|we've|we'll|we'd|our|us|ourselves)\b",
        re.IGNORECASE,
    )

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, require_first_person: bool = False):
        self.model_name = model_name
        self.model = None
        self.require_first_person = require_first_person
        self.max_length: int = 512
        self.batch_size: int = 8

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_gliner(self):
        from gliner import GLiNER

        self.model = GLiNER.from_pretrained(self.model_name)

    def _df_to_records(self, df: pd.DataFrame):
        """Convert a DataFrame row into (id, tokens) pairs."""
        records = []
        for _, row in df.iterrows():
            tokens = ast.literal_eval(row["tokens"]) if isinstance(row["tokens"], str) else row["tokens"]
            records.append((row["ID"], tokens))
        return records

    def _spans_to_bio(self, tokens: list, spans: list) -> tuple:
        """Convert GLiNER span predictions to parallel BIO and flat label lists."""
        ner_tags = ["O"] * len(tokens)
        labels = [""] * len(tokens)

        # Sort spans by start position so B- vs I- assignment is correct
        for span in sorted(spans, key=lambda s: s["start"]):
            entity_type = span["label"]
            start = span["start"]
            end = span["end"]  # exclusive token index
            for i in range(start, min(end, len(tokens))):
                if i == start:
                    ner_tags[i] = f"B-{entity_type}"
                else:
                    ner_tags[i] = f"I-{entity_type}"
                labels[i] = entity_type

        return labels, ner_tags

    @classmethod
    def _is_first_person(cls, tokens: list[str]) -> bool:
        """Return *True* if the sentence appears to be written in the first person.

        The check looks for first-person singular/plural pronouns anywhere in
        the token list using a pre-compiled regular expression.
        """
        text = " ".join(tokens)
        return bool(cls._FIRST_PERSON_PATTERN.search(text))

    def _translate_spans(self, token_spans: list[dict]) -> list[dict]:
        """Translate granular GLiNER entity labels to shared-task classes.

        Each span's ``"label"`` key is replaced by the corresponding entry in
        :attr:`ENTITY_TO_IMPACT`.  Spans whose label is not in the mapping
        (e.g. the bare ``"ClinicalImpacts"`` / ``"SocialImpacts"`` labels used
        during fine-tuning) are passed through unchanged.
        """
        translated = []
        for span in token_spans:
            label = span["label"]
            mapped = self.ENTITY_TO_IMPACT.get(label, label)
            translated.append({**span, "label": mapped})
        return translated

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        output_path: str,
        metric: str = "f1",
        **kwargs,
    ) -> None:
        """Fine-tune the GLiNER model on the supplied data.

        The *metric* parameter is accepted for interface consistency but is not
        used because the GLiNER trainer does not expose a ``metric_for_best_model``
        option.
        """
        from gliner import GLiNER
        from gliner.training import Trainer as GlinerTrainer, TrainingConfig

        if self.model is None:
            self._load_gliner()

        # Build training examples in GLiNER's expected format
        def _df_to_gliner_examples(df: pd.DataFrame):
            examples = []
            for _, row in df.iterrows():
                tokens = ast.literal_eval(row["tokens"]) if isinstance(row["tokens"], str) else row["tokens"]
                ner_tags = ast.literal_eval(row["ner_tags"]) if isinstance(row["ner_tags"], str) else row["ner_tags"]
                # Extract entity spans from BIO tags
                entities = []
                start = None
                entity_type = None
                for idx, tag in enumerate(ner_tags):
                    if tag.startswith("B-"):
                        if start is not None:
                            entities.append([start, idx, entity_type])
                        start = idx
                        entity_type = tag[2:]
                    elif tag.startswith("I-") and start is not None:
                        pass  # continuation
                    else:
                        if start is not None:
                            entities.append([start, idx, entity_type])
                            start = None
                            entity_type = None
                if start is not None:
                    entities.append([start, len(ner_tags), entity_type])
                examples.append({"tokenized_text": tokens, "ner": entities})
            return examples

        train_examples = _df_to_gliner_examples(train_data)
        val_examples = _df_to_gliner_examples(val_data)

        batch_size = kwargs.get("batch_size", self.batch_size)
        config = TrainingConfig(
            num_steps=500,
            train_batch_size=batch_size,
            val_every_n_steps=100,
            save_directory=output_path,
        )
        trainer = GlinerTrainer(model=self.model, config=config)
        trainer.train(train_data=train_examples, val_data=val_examples)
        self.save(output_path)

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            self._load_gliner()

        results = []
        for sentence_id, tokens in self._df_to_records(data):
            # Decide whether to restrict predictions to first-person sentences
            if self.require_first_person and not self._is_first_person(tokens):
                ner_tags = ["O"] * len(tokens)
                labels = [""] * len(tokens)
            else:
                spans = self.model.predict_entities(
                    " ".join(tokens),
                    self.GLINER_ENTITY_TYPES,
                    flat_ner=True,
                )
                # GLiNER returns character-level spans; convert to token-level
                text = " ".join(tokens)
                token_spans = self._char_spans_to_token_spans(text, tokens, spans)
                # Translate granular labels → ClinicalImpacts / SocialImpacts
                token_spans = self._translate_spans(token_spans)
                labels, ner_tags = self._spans_to_bio(tokens, token_spans)
                ner_tags = self.autocorrect_labels(ner_tags)
                labels = [self.bio_to_label(tag) for tag in ner_tags]
            results.append(
                {
                    "ID": sentence_id,
                    "tokens": str(tokens),
                    "labels": str(labels),
                    "ner_tags": str(ner_tags),
                }
            )
        return pd.DataFrame(results)

    def _char_spans_to_token_spans(self, text: str, tokens: list, char_spans: list) -> list:
        """Map character-level GLiNER spans to token indices."""
        # Build a char→token index map
        char_to_token = {}
        pos = 0
        for token_idx, token in enumerate(tokens):
            for _ in token:
                char_to_token[pos] = token_idx
                pos += 1
            if pos < len(text):
                char_to_token[pos] = None  # space
                pos += 1

        token_spans = []
        for span in char_spans:
            start_char = span["start"]
            end_char = span["end"]
            start_token = char_to_token.get(start_char)
            # Walk back from end_char to find last token
            end_token = None
            for c in range(end_char - 1, start_char - 1, -1):
                t = char_to_token.get(c)
                if t is not None:
                    end_token = t + 1
                    break
            if start_token is not None and end_token is not None:
                token_spans.append({"start": start_token, "end": end_token, "label": span["label"]})
        return token_spans

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)

    @classmethod
    def load(cls, path: str) -> "GlinerModel":
        from gliner import GLiNER

        instance = cls(model_name=path)
        instance.model = GLiNER.from_pretrained(path)
        return instance
