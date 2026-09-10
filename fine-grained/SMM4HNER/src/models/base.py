"""Abstract base class for all NER models."""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from transformers import TrainerCallback


def _print_metrics_block(title: str, metrics: dict) -> None:
    """Print NER metrics in a grouped, readable format.

    Parameters
    ----------
    title:
        Label shown as the block header (e.g. ``"Trainset"`` or
        ``"Validation set"``).
    metrics:
        Dictionary returned by :meth:`BaseNERModel.compute_metrics` or by
        ``Trainer.evaluate()``.  Both ``"f1"`` and ``"eval_f1"`` key styles
        are handled.
    """
    def _get(key):
        return metrics.get(key, metrics.get(f"eval_{key}", 0.0))

    print(f"\n{title}")
    print(f"    F1  = {_get('f1'):.4f}")
    print(f"    P   = {_get('precision'):.4f}")
    print(f"    R   = {_get('recall'):.4f}")
    print(f"    RF1 = {_get('relaxed_f1'):.4f}")
    print(f"    RP  = {_get('relaxed_precision'):.4f}")
    print(f"    RR  = {_get('relaxed_recall'):.4f}")
    print()


class TrainEvalCallback(TrainerCallback):
    """Evaluates on training and validation splits at every epoch end.

    Printing both sets side-by-side makes it easy to spot overfitting.

    Usage::

        cb = TrainEvalCallback(train_dataset, val_dataset)
        trainer = Trainer(..., callbacks=[cb])
        cb.trainer = trainer   # back-reference so the callback can call evaluate()
        trainer.train()
    """

    def __init__(self, train_dataset, val_dataset):
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset
        self.trainer = None  # populated after Trainer is constructed

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.trainer is None:
            return control
        train_metrics = self.trainer.evaluate(eval_dataset=self._train_dataset)
        val_metrics = self.trainer.evaluate(eval_dataset=self._val_dataset)
        print(f"[hner] ── epoch {int(state.epoch)} ──────────────────────")
        _print_metrics_block("Trainset", train_metrics)
        _print_metrics_block("Validation set", val_metrics)
        return control


class BaseNERModel(ABC):
    """Base interface for NER models used in the SMM4H shared task."""

    LABEL_LIST = ["O", "B-ClinicalImpacts", "I-ClinicalImpacts", "B-SocialImpacts", "I-SocialImpacts"]
    LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
    ID2LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}

    # Entity types used by span-based models (e.g. GLiNER)
    ENTITY_TYPES = ["ClinicalImpacts", "SocialImpacts"]

    @abstractmethod
    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        output_path: str,
        metric: str = "f1",
        **kwargs,
    ) -> None:
        """Train the model and save it to *output_path*.

        Parameters
        ----------
        train_data:
            Training set as a ``pandas.DataFrame``.
        val_data:
            Validation set used for model selection during training.
        output_path:
            Directory where the best checkpoint will be persisted.
        metric:
            Metric used for best-model selection.  One of ``"f1"`` (strict,
            seqeval-based), ``"relaxed_f1"`` (overlap-based relaxed F1), or
            ``"avg_f1"`` (average of strict and relaxed F1).
        """

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return *data* with new 'labels' and 'ner_tags' columns filled in."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the model to *path*."""

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseNERModel":
        """Load a previously saved model from *path*."""

    # ------------------------------------------------------------------
    # Shared helper utilities
    # ------------------------------------------------------------------

    @staticmethod
    def compute_metrics(eval_pred) -> dict:
        """Compute strict and relaxed NER metrics for HF Trainer.

        Parameters
        ----------
        eval_pred:
            An ``EvalPrediction`` namedtuple with fields ``predictions``
            (logits, shape ``[batch, seq, num_labels]``) and ``label_ids``
            (shape ``[batch, seq]``, with ``-100`` for padding tokens).

        Returns
        -------
        dict
            A dictionary with strict (``"precision"``, ``"recall"``, ``"f1"``)
            and relaxed (``"relaxed_f1"``) metrics.
        """
        from seqeval.metrics import f1_score, precision_score, recall_score
        from data.evaluation import calculate_f1_per_entity_covering_all

        logits, label_ids = eval_pred
        predictions = np.argmax(logits, axis=2)

        id2label = BaseNERModel.ID2LABEL
        true_labels = []
        true_preds = []
        for pred_seq, label_seq in zip(predictions, label_ids):
            true_label_seq = []
            true_pred_seq = []
            for pred, label in zip(pred_seq, label_seq):
                if label != -100:
                    true_label_seq.append(id2label[label])
                    true_pred_seq.append(id2label[pred])
            true_labels.append(true_label_seq)
            true_preds.append(true_pred_seq)

        per_entity = calculate_f1_per_entity_covering_all(true_labels, true_preds)
        overall_relaxed = per_entity.get("Overall", {})

        strict_f1 = f1_score(true_labels, true_preds)
        relaxed_f1 = overall_relaxed.get("F1-Score", 0.0)
        results = {
            "precision": precision_score(true_labels, true_preds),
            "recall": recall_score(true_labels, true_preds),
            "f1": strict_f1,
            "relaxed_f1": relaxed_f1,
            "avg_f1": (strict_f1 + relaxed_f1) / 2,
            "relaxed_precision": overall_relaxed.get("Precision", 0.0),
            "relaxed_recall": overall_relaxed.get("Recall", 0.0),
        }
        return results

    @staticmethod
    def autocorrect_labels(ner_tags: list) -> list:
        """Fix invalid BIO tag sequences in *ner_tags*.

        An ``I-X`` tag is only valid when it immediately follows a ``B-X`` or
        ``I-X`` tag.  Any ``I-X`` that appears at the start of a sequence or
        after an ``O`` tag (or after a tag for a different entity type) is
        converted to ``B-X``.

        Parameters
        ----------
        ner_tags:
            A list of BIO tags, e.g. ``['O', 'I-ClinicalImpacts', 'O']``.

        Returns
        -------
        list
            The corrected tag sequence.
        """
        corrected = []
        for tag in ner_tags:
            if tag.startswith("I-"):
                entity_type = tag[2:]
                if not corrected or corrected[-1] not in (f"B-{entity_type}", f"I-{entity_type}"):
                    corrected.append(f"B-{entity_type}")
                else:
                    corrected.append(tag)
            else:
                corrected.append(tag)
        return corrected

    @staticmethod
    def _merge_word_prediction(existing: str, new: str) -> str:
        """Resolve a labeling conflict when two overlapping chunks predict the same word.

        Rules (from the issue):

        * If one chunk labels the word ``"O"`` and the other assigns any non-O
          label, use the non-O label.
        * If both chunks assign different non-O labels, keep the existing one
          (i.e. the prediction from the earlier chunk).

        Parameters
        ----------
        existing:
            The label already stored for this word (from a previous chunk).
        new:
            The label predicted by the current chunk.

        Returns
        -------
        str
            The resolved label.
        """
        if existing == "O" and new != "O":
            return new
        return existing

    @staticmethod
    def bio_to_label(ner_tag: str) -> str:
        """Convert a BIO tag (e.g. 'B-ClinicalImpacts') to a flat label."""
        if ner_tag == "O":
            return ""
        return ner_tag.split("-", 1)[1]

    @staticmethod
    def labels_to_bio(labels: list) -> list:
        """Convert a sequence of flat labels to BIO tags."""
        bio = []
        prev = ""
        for label in labels:
            if not label:
                bio.append("O")
                prev = ""
            elif label == prev:
                bio.append(f"I-{label}")
            else:
                bio.append(f"B-{label}")
                prev = label
        return bio

    # ------------------------------------------------------------------
    # Sequence-level classification interface
    # ------------------------------------------------------------------

    def train_seq_cls(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        output_path: str,
        **kwargs,
    ) -> None:
        """Train the model for sequence-level (binary) classification.

        The dataframe is expected to contain at least ``prompt`` and ``label``
        columns, where ``label`` is a binary integer (0 or 1).

        Parameters
        ----------
        train_data:
            Training set as a ``pandas.DataFrame``.
        val_data:
            Validation set used for model selection during training.
        output_path:
            Directory where the best checkpoint will be persisted.

        Raises
        ------
        NotImplementedError
            By default.  Concrete subclasses that support sequence
            classification must override this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support sequence classification "
            "training.  Use the 'bert_finetuned', 'bert_adapter', or 'lstm' model type."
        )

    def predict_seq_cls(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run sequence-level classification inference.

        Parameters
        ----------
        data:
            Dataframe with at least a ``prompt`` column.

        Returns
        -------
        pd.DataFrame
            Input dataframe with an additional ``prediction`` column containing
            the predicted binary label (0 or 1).

        Raises
        ------
        NotImplementedError
            By default.  Concrete subclasses must override this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support sequence classification "
            "inference.  Use the 'bert_finetuned', 'bert_adapter', or 'lstm' model type."
        )
