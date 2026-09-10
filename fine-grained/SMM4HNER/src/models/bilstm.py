"""Bidirectional LSTM NER model.

This module provides :class:`BiLSTMModel`, a ``BaseNERModel``-compatible
wrapper around a subword-augmented Bidirectional LSTM sequence labeller.

Architecture
------------
For each sentence the model:

1. Looks up a **direct embedding** (E_de) for every token — a trained
   300-dimensional vector for words seen at least twice in the training
   corpus.
2. Looks up a frozen **word2vec embedding** (E_w2v) for every token using
   pre-trained 300-dimensional word2vec vectors loaded from an external
   file.
3. Computes a **subword embedding** (E_sw) by averaging the learned
   300-dimensional embeddings of all character n-grams (trigrams, 4-grams
   and 5-grams with boundary markers ``<`` / ``>``) extracted from the
   token.
4. Sums the three embeddings (E_de + E_w2v + E_sw) and feeds the result
   through a multi-layer **BiLSTM**.
5. Applies a linear **classification head** to produce per-token logits
   over the BIO label set.

During training, each of the three embedding types is independently
dropped with 20 % probability per sample to improve robustness.  When
one embedding is dropped the sum of the remaining two is scaled by 2;
when two are dropped the surviving embedding is scaled by 3.

Vocabulary / encoding
---------------------
The vocabularies (word-to-index and n-gram-to-index) are built from the
training corpus by :class:`~io_utils.embeddings.Encodings` and serialised
to YAML alongside the model weights so that the same encodings can be
reloaded at inference time.

When an external word2vec file is supplied, a dedicated ``w2v_word2int``
index covering **all** words in that file is built and stored in the
:class:`~io_utils.embeddings.Encodings`.  This allows E_w2v to generalise
to unseen words at inference time, not just to words present in the
training corpus.
"""

import ast
import json
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from io_utils.embeddings import Collate, Encodings
from models.base import BaseNERModel, _print_metrics_block
from models.losses import MulticlassDiceLoss
from utils.device import get_device, log_device

# Minimum w2v vocabulary size when no embeddings file is provided (PAD + UNK).
_MIN_W2V_VOCAB_SIZE = 2


# ---------------------------------------------------------------------------
# Word2Vec weight-matrix loader
# ---------------------------------------------------------------------------

def _load_vectors(fname: str):
    """Load vectors from a word2vec text-format file.

    Returns a tuple of (data, vector_size) where data is a dict mapping
    words to their vector values, and vector_size is the embedding dimension.
    """
    with open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
        n, d = map(int, fin.readline().split())
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = list(map(float, tokens[1:]))
    return data, d


def _load_w2v_weights(
    word2vec_file: Optional[str],
    emb_dim: int,
) -> tuple:
    """Build a frozen weight matrix for the word2vec embedding layer.

    Unlike the direct-embedding vocabulary (which only covers training-set
    words), this function loads **all** words from the external embeddings
    file and constructs a dedicated ``w2v_word2int`` index.  This allows
    the model to provide word2vec representations for unseen words at
    inference time.

    Parameters
    ----------
    word2vec_file:
        Path to a word2vec **text**-format file (first line contains the
        vocabulary size and dimension; each subsequent line starts with
        the word followed by the vector values).  If ``None`` an empty
        vocabulary is returned.
    emb_dim:
        Expected embedding dimension.  If the file dimension differs, an
        empty vocabulary is returned.

    Returns
    -------
    tuple
        ``(weights, w2v_word2int)`` where:

        * ``weights`` – ``FloatTensor[w2v_vocab_size, emb_dim]``
        * ``w2v_word2int`` – ``Dict[str, int]`` mapping every word in
          the embeddings file to a unique index (PAD=0, UNK=1).
    """
    # Minimal fallback: PAD=0, UNK=1
    empty_w2v_word2int: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    empty_weights = torch.zeros(len(empty_w2v_word2int), emb_dim)

    if not word2vec_file:
        return empty_weights, empty_w2v_word2int

    try:
        vectors, vector_size = _load_vectors(word2vec_file)
    except Exception as exc:  # pragma: no cover
        print(f"[hner] Warning: could not load word2vec from '{word2vec_file}': {exc}")
        return empty_weights, empty_w2v_word2int

    if vector_size != emb_dim:
        print(
            f"[hner] Warning: word2vec dimension ({vector_size}) "
            f"!= emb_dim ({emb_dim}). Word2Vec embeddings will be zeros."
        )
        return empty_weights, empty_w2v_word2int

    # Build a dedicated vocabulary from the embeddings file
    w2v_word2int: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for word in vectors:
        if word not in w2v_word2int:
            w2v_word2int[word] = len(w2v_word2int)

    vocab_size = len(w2v_word2int)
    weights = torch.zeros(vocab_size, emb_dim)
    for word, idx in w2v_word2int.items():
        if word in vectors:
            weights[idx] = torch.tensor(vectors[word], dtype=torch.float)

    print(f"[hner] Word2Vec: loaded {len(vectors)} vectors from '{word2vec_file}' "
          f"(w2v vocab size = {vocab_size})")
    return weights, w2v_word2int


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class NERDataset(Dataset):
    """Token-classification dataset backed by a :class:`Collate` encoder.

    Parameters
    ----------
    data:
        A ``pandas.DataFrame`` with ``"tokens"`` and ``"ner_tags"``
        columns.  Both may be stored as string-encoded Python lists.
    collate:
        A fitted :class:`Collate` instance used to convert raw words and
        n-grams to integer indices.
    """

    def __init__(self, data: pd.DataFrame, collate: Collate) -> None:
        self.examples: List[tuple] = []
        for _, row in data.iterrows():
            tokens = (
                ast.literal_eval(row["tokens"])
                if isinstance(row["tokens"], str)
                else list(row["tokens"])
            )
            ner_tags = (
                ast.literal_eval(row["ner_tags"])
                if isinstance(row["ner_tags"], str)
                else list(row["ner_tags"])
            )
            word_ids = [collate.encode_word(t) for t in tokens]
            w2v_word_ids = [collate.encode_word_w2v(t) for t in tokens]
            subword_ids = [collate.encode_subwords(t) for t in tokens]
            label_ids = [BaseNERModel.LABEL2ID.get(tag, 0) for tag in ner_tags]
            self.examples.append((word_ids, w2v_word_ids, subword_ids, label_ids))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]


class SequenceClassificationDataset(Dataset):
    """Sequence-classification dataset for BiLSTM prompt classification."""

    def __init__(
        self,
        data: pd.DataFrame,
        collate: Collate,
        label2id: Dict[str, int],
        granular2id: Optional[Dict[str, int]] = None,
        max_length: Optional[int] = None,
    ) -> None:
        self.examples: List[tuple] = []
        has_granular = granular2id is not None and "granular_label" in data.columns
        for _, row in data.iterrows():
            tokens = BiLSTMModel._prompt_to_tokens(str(row["prompt"]))
            if max_length is not None:
                tokens = tokens[:max_length]
                if not tokens:
                    tokens = ["<EMPTY>"]
            word_ids = [collate.encode_word(t) for t in tokens]
            w2v_word_ids = [collate.encode_word_w2v(t) for t in tokens]
            subword_ids = [collate.encode_subwords(t) for t in tokens]
            label_value = str(row["label"])
            if label_value not in label2id:
                raise ValueError(f"Unknown label value '{label_value}' encountered in sequence data.")
            label_id = label2id[label_value]
            granular_id = None
            if has_granular:
                granular_value = str(row["granular_label"])
                if granular_value not in granular2id:
                    raise ValueError(
                        f"Unknown granular_label value '{granular_value}' encountered in sequence data."
                    )
                granular_id = granular2id[granular_value]
            self.examples.append((word_ids, w2v_word_ids, subword_ids, label_id, granular_id))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]


class SequenceClassificationCollate:
    """Collate sequence-classification examples into padded tensors."""

    def __init__(self, collate: Collate, has_granular: bool) -> None:
        self.collate = collate
        self.has_granular = has_granular

    def __call__(self, batch):
        token_batch = [
            (word_ids, w2v_word_ids, subword_ids, [0] * len(word_ids))
            for word_ids, w2v_word_ids, subword_ids, _label, _granular in batch
        ]
        word_t, w2v_word_t, sub_t, _label_t, lengths = self.collate(token_batch)
        label_t = torch.tensor([label for *_prefix, label, _gran in batch], dtype=torch.long)
        granular_t = None
        if self.has_granular:
            granular_t = torch.tensor([gran for *_prefix, _label, gran in batch], dtype=torch.long)
        return word_t, w2v_word_t, sub_t, lengths, label_t, granular_t


# ---------------------------------------------------------------------------
# PyTorch module
# ---------------------------------------------------------------------------

class _BiLSTMNER(nn.Module):
    """Inner PyTorch module for the BiLSTM NER tagger.

    Parameters
    ----------
    vocab_size:
        Number of entries in the word vocabulary (including ``<PAD>`` /
        ``<UNK>``).
    w2v_vocab_size:
        Number of entries in the word2vec vocabulary built from the
        external embeddings file.  This is independent of *vocab_size*
        and covers all words present in the embeddings file.
    token_vocab_size:
        Number of entries in the subword (character n-gram) vocabulary.
    emb_dim:
        Dimensionality shared by all three embedding types (direct,
        word2vec and subword).  Defaults to 300.
    hidden_dim:
        Number of hidden units in each LSTM direction.  The output size
        of the BiLSTM is therefore ``2 * hidden_dim``.
    num_labels:
        Number of BIO label classes.
    num_layers:
        Number of stacked LSTM layers.
    dropout:
        Dropout probability applied after the embedding layer and after
        the BiLSTM output.
    w2v_weights:
        Optional ``FloatTensor[w2v_vocab_size, emb_dim]`` of pre-trained
        word2vec vectors aligned with *w2v_vocab_size*.  When provided
        these weights initialise the frozen word2vec embedding layer;
        missing rows should be zero-filled by the caller.
    """

    def __init__(
        self,
        vocab_size: int,
        w2v_vocab_size: int,
        token_vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        num_labels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        w2v_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        # E_de — direct word embeddings (trained)
        self.word_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        # E_w2v — word2vec embeddings (frozen), sized to the full w2v vocabulary
        self.w2v_emb = nn.Embedding(w2v_vocab_size, emb_dim, padding_idx=0)
        nn.init.zeros_(self.w2v_emb.weight)
        self.w2v_emb.weight.requires_grad_(False)
        if w2v_weights is not None:
            with torch.no_grad():
                self.w2v_emb.weight.copy_(w2v_weights)

        # E_sw — subword embeddings (trained, averaged over n-grams)
        self.subword_emb = nn.Embedding(token_vocab_size, emb_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    # ------------------------------------------------------------------
    # Embedding dropout helper
    # ------------------------------------------------------------------

    def _apply_embedding_dropout(
        self,
        e_de: torch.Tensor,
        e_w2v: torch.Tensor,
        e_sw: torch.Tensor,
    ) -> torch.Tensor:
        """Sum embeddings with per-sample dropout (used during training).

        Each of the three embedding types is independently zeroed out
        with 20 % probability per sample.  The remaining representations
        are then scaled to preserve expected magnitude:

        * 1 dropped  → scale the sum of the other two by **2**
        * 2 dropped  → scale the surviving representation by **3**
        * 0 or 3 dropped → no scaling (sum or zero vector respectively)

        Parameters
        ----------
        e_de, e_w2v, e_sw:
            ``FloatTensor[B, L, emb_dim]``.

        Returns
        -------
        torch.Tensor
            ``FloatTensor[B, L, emb_dim]``
        """
        B = e_de.size(0)
        device = e_de.device

        # Independent 20 % dropout masks per sample — shape [B, 1, 1]
        keep_de  = (torch.rand(B, 1, 1, device=device) > 0.5).float()
        keep_w2v = (torch.rand(B, 1, 1, device=device) > 0.5).float()
        keep_sw  = (torch.rand(B, 1, 1, device=device) > 0.5).float()

        n_kept = keep_de + keep_w2v + keep_sw  # [B, 1, 1]

        # scale = 4 - n_kept:
        #   n_kept=3 (0 dropped) → scale=1
        #   n_kept=2 (1 dropped) → scale=2
        #   n_kept=1 (2 dropped) → scale=3
        #   n_kept=0 (3 dropped) → scale=4, but result is 0 anyway
        scale = 4.0 - n_kept

        return (e_de * keep_de + e_w2v * keep_w2v + e_sw * keep_sw) * scale

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        word_ids: torch.Tensor,
        w2v_word_ids: torch.Tensor,
        subword_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        word_ids:
            ``LongTensor[batch, seq_len]`` — indices into the training
            vocabulary (used for E_de).
        w2v_word_ids:
            ``LongTensor[batch, seq_len]`` — indices into the word2vec
            vocabulary (used for E_w2v).
        subword_ids:
            ``LongTensor[batch, seq_len, max_sub]``

        Returns
        -------
        torch.Tensor
            ``FloatTensor[batch, seq_len, num_labels]`` — raw logits.
        """
        # E_de — direct word embeddings
        e_de = self.word_emb(word_ids)          # [B, L, emb_dim]

        # E_w2v — frozen word2vec embeddings (full vocabulary)
        e_w2v = self.w2v_emb(w2v_word_ids)      # [B, L, emb_dim]

        # E_sw — subword embeddings: average over the n-gram dimension
        sw_all = self.subword_emb(subword_ids)  # [B, L, max_sub, emb_dim]
        mask = (subword_ids != 0).unsqueeze(-1).float()   # [B, L, max_sub, 1]
        e_sw = (sw_all * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1.0)

        # Combine embeddings
        if self.training:
            x = self._apply_embedding_dropout(e_de, e_w2v, e_sw)
        else:
            x = e_de + e_w2v + e_sw          # [B, L, emb_dim]

        lstm_out, _ = self.lstm(x)           # [B, L, 2*hidden_dim]

        logits = self.classifier(lstm_out)   # [B, L, num_labels]
        return logits


class _BiLSTMSeqCls(nn.Module):
    """BiLSTM encoder with one or two sequence-classification heads."""

    def __init__(
        self,
        vocab_size: int,
        w2v_vocab_size: int,
        token_vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        num_label_classes: int,
        num_granular_classes: int = 0,
        num_layers: int = 2,
        dropout: float = 0.5,
        w2v_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.w2v_emb = nn.Embedding(w2v_vocab_size, emb_dim, padding_idx=0)
        nn.init.zeros_(self.w2v_emb.weight)
        self.w2v_emb.weight.requires_grad_(False)
        if w2v_weights is not None:
            with torch.no_grad():
                self.w2v_emb.weight.copy_(w2v_weights)

        self.subword_emb = nn.Embedding(token_vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

        head_input_dim = hidden_dim * 2

        def _make_head(num_classes: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(head_input_dim, head_input_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_input_dim, num_classes),
            )

        self.label_head = _make_head(num_label_classes)
        self.granular_head = _make_head(num_granular_classes) if num_granular_classes > 0 else None

    # ------------------------------------------------------------------
    # Embedding dropout helper
    # ------------------------------------------------------------------

    def _apply_embedding_dropout(
        self,
        e_de: torch.Tensor,
        e_w2v: torch.Tensor,
        e_sw: torch.Tensor,
    ) -> torch.Tensor:
        """Sum embeddings with per-sample dropout (used during training).

        Each of the three embedding types is independently zeroed out
        with 20 % probability per sample.  The remaining representations
        are then scaled to preserve expected magnitude:

        * 1 dropped  → scale the sum of the other two by **2**
        * 2 dropped  → scale the surviving representation by **3**
        * 0 or 3 dropped → no scaling (sum or zero vector respectively)

        Parameters
        ----------
        e_de, e_w2v, e_sw:
            ``FloatTensor[B, L, emb_dim]``.

        Returns
        -------
        torch.Tensor
            ``FloatTensor[B, L, emb_dim]``
        """
        B = e_de.size(0)
        device = e_de.device

        # Independent 20 % dropout masks per sample — shape [B, 1, 1]
        keep_de  = (torch.rand(B, 1, 1, device=device) > 0.5).float()
        keep_w2v = (torch.rand(B, 1, 1, device=device) > 0.5).float()
        keep_sw  = (torch.rand(B, 1, 1, device=device) > 0.5).float()

        n_kept = keep_de + keep_w2v + keep_sw  # [B, 1, 1]

        # scale = 4 - n_kept:
        #   n_kept=3 (0 dropped) → scale=1
        #   n_kept=2 (1 dropped) → scale=2
        #   n_kept=1 (2 dropped) → scale=3
        #   n_kept=0 (3 dropped) → scale=4, but result is 0 anyway
        scale = 4.0 - n_kept

        return (e_de * keep_de + e_w2v * keep_w2v + e_sw * keep_sw) * scale

    # ------------------------------------------------------------------
    # Pooling helper
    # ------------------------------------------------------------------

    def _pool_embeddings(self, embeddings: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = embeddings.size()
        mask = (
            torch.arange(seq_len, device=embeddings.device)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
            < lengths.unsqueeze(1)
        ).unsqueeze(-1).float()
        pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled

    def forward(
        self,
        word_ids: torch.Tensor,
        w2v_word_ids: torch.Tensor,
        subword_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        e_de = self.word_emb(word_ids)
        e_w2v = self.w2v_emb(w2v_word_ids)
        sw_all = self.subword_emb(subword_ids)
        mask = (subword_ids != 0).unsqueeze(-1).float()
        e_sw = (sw_all * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1.0)

        # Combine embeddings
        if self.training:
            x = self._apply_embedding_dropout(e_de, e_w2v, e_sw)
        else:
            x = e_de + e_w2v + e_sw

        lstm_out, _ = self.lstm(x)
        pooled = self._pool_embeddings(lstm_out, lengths)
        pooled = self.dropout(pooled)

        label_logits = self.label_head(pooled)
        granular_logits = self.granular_head(pooled) if self.granular_head is not None else None
        return label_logits, granular_logits


# ---------------------------------------------------------------------------
# BaseNERModel wrapper
# ---------------------------------------------------------------------------

class BiLSTMModel(BaseNERModel):
    """Bidirectional LSTM NER model with three 300-dimensional embedding types.

    The word representation fed into the BiLSTM is the element-wise sum of:

    * **E_de** — direct (trained) word embeddings for words that appear at
      least twice in the training corpus.
    * **E_w2v** — frozen word2vec embeddings loaded from an external text
      file at training time (requires ``--word2vec-file``).
    * **E_sw** — subword embeddings computed as the average of learned
      300-dimensional embeddings for all character n-grams (3–5-grams with
      boundary markers) extracted from each token.

    During training each embedding type is independently dropped with 20 %
    probability per sample; the surviving representations are rescaled so
    that their expected magnitude remains constant.

    Hyper-parameters
    ----------------
    EMB_DIM : int
        Dimensionality shared by all three embedding types (default 300).
    HIDDEN_DIM : int
        Hidden units per LSTM direction (default 256).
    NUM_LAYERS : int
        Number of stacked LSTM layers (default 2).
    EPOCHS : int
        Maximum training epochs (default 50).
    BATCH_SIZE : int
        Mini-batch size (default 32).
    LR : float
        Adam learning rate (default 1e-4).
    """

    EMB_DIM: int = 300
    HIDDEN_DIM: int = 50
    NUM_LAYERS: int = 2
    EPOCHS: int = 200
    BATCH_SIZE: int = 32
    DEFAULT_MAX_LENGTH: int = 4096
    LR: float = 1e-3

    _CONFIG_FILE = "config.yaml"
    _WEIGHTS_FILE = "model.pt"
    _ENCODINGS_FILE = "encodings.yaml"
    _SEQ_CLS_CONFIG_FILE = "seq_cls_config.json"

    def __init__(self) -> None:
        self.encodings: Optional[Encodings] = None
        self.collate_fn: Optional[Collate] = None
        self.model: Optional[Union[_BiLSTMNER, _BiLSTMSeqCls]] = None
        self.max_length: int = self.DEFAULT_MAX_LENGTH
        self.batch_size: int = self.BATCH_SIZE
        self._label_classes: Optional[List[str]] = None
        self._granular_classes: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prompt_to_tokens(prompt: str) -> List[str]:
        """Tokenize prompt text for the BiLSTM sequence classifier."""
        tokens = prompt.split()
        return tokens if tokens else ["<EMPTY>"]

    def _build_model(
        self, w2v_weights: Optional[torch.Tensor] = None
    ) -> _BiLSTMNER:
        """Instantiate ``_BiLSTMNER`` from the current encodings.

        Parameters
        ----------
        w2v_weights:
            Optional pre-trained word2vec weight matrix
            ``FloatTensor[w2v_vocab_size, emb_dim]``.
        """
        assert self.encodings is not None
        w2v_vocab_size = len(self.encodings.w2v_word2int) if self.encodings.w2v_word2int else _MIN_W2V_VOCAB_SIZE
        return _BiLSTMNER(
            vocab_size=len(self.encodings.word2int),
            w2v_vocab_size=w2v_vocab_size,
            token_vocab_size=len(self.encodings.token2int),
            emb_dim=self.EMB_DIM,
            hidden_dim=self.HIDDEN_DIM,
            num_labels=len(self.LABEL_LIST),
            num_layers=self.NUM_LAYERS,
            w2v_weights=w2v_weights,
        )

    def _build_seq_cls_model(
        self,
        num_label_classes: int,
        num_granular_classes: int = 0,
        w2v_weights: Optional[torch.Tensor] = None,
    ) -> _BiLSTMSeqCls:
        """Instantiate ``_BiLSTMSeqCls`` from current encodings."""
        assert self.encodings is not None
        w2v_vocab_size = (
            len(self.encodings.w2v_word2int)
            if self.encodings.w2v_word2int
            else _MIN_W2V_VOCAB_SIZE
        )
        return _BiLSTMSeqCls(
            vocab_size=len(self.encodings.word2int),
            w2v_vocab_size=w2v_vocab_size,
            token_vocab_size=len(self.encodings.token2int),
            emb_dim=self.EMB_DIM,
            hidden_dim=self.HIDDEN_DIM,
            num_label_classes=num_label_classes,
            num_granular_classes=num_granular_classes,
            num_layers=self.NUM_LAYERS,
            w2v_weights=w2v_weights,
        )

    @staticmethod
    def _compute_seq_metrics(
        all_true: List[List[str]], all_pred: List[List[str]]
    ) -> dict:
        """Compute strict and relaxed NER F1 scores."""
        from seqeval.metrics import f1_score, precision_score, recall_score
        from data.evaluation import calculate_f1_per_entity_covering_all

        per_entity = calculate_f1_per_entity_covering_all(all_true, all_pred)
        overall_relaxed = per_entity.get("Overall", {})
        strict_f1 = f1_score(all_true, all_pred, zero_division=0)
        relaxed_f1 = overall_relaxed.get("F1-Score", 0.0)
        return {
            "f1": strict_f1,
            "precision": precision_score(all_true, all_pred, zero_division=0),
            "recall": recall_score(all_true, all_pred, zero_division=0),
            "relaxed_f1": relaxed_f1,
            "relaxed_precision": overall_relaxed.get("Precision", 0.0),
            "relaxed_recall": overall_relaxed.get("Recall", 0.0),
            "avg_f1": (strict_f1 + relaxed_f1) / 2.0,
        }

    @staticmethod
    def _compute_classification_metrics(
        y_true: List[int], y_pred: List[int], num_classes: int
    ) -> dict:
        """Compute accuracy, precision, recall and macro F1 for sequence classification."""
        if not y_true:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "macro_f1": 0.0}

        accuracy = sum(int(t == p) for t, p in zip(y_true, y_pred)) / len(y_true)

        def _prf_for_class(cls_idx: int) -> Tuple[float, float, float]:
            tp = sum(int(t == cls_idx and p == cls_idx) for t, p in zip(y_true, y_pred))
            fp = sum(int(t != cls_idx and p == cls_idx) for t, p in zip(y_true, y_pred))
            fn = sum(int(t == cls_idx and p != cls_idx) for t, p in zip(y_true, y_pred))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                (2.0 * precision * recall / (precision + recall))
                if (precision + recall) > 0
                else 0.0
            )
            return precision, recall, f1

        all_prf = [_prf_for_class(i) for i in range(num_classes)]
        precision = sum(v[0] for v in all_prf) / num_classes
        recall = sum(v[1] for v in all_prf) / num_classes
        macro_f1 = sum(v[2] for v in all_prf) / num_classes

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "macro_f1": macro_f1,
        }

    def _predict_sequences(
        self, data: pd.DataFrame, device: torch.device
    ) -> List[List[str]]:
        """Run inference and return predicted BIO tag sequences."""
        assert self.model is not None and self.collate_fn is not None
        self.model.eval()
        results = []
        with torch.no_grad():
            for _, row in data.iterrows():
                tokens = (
                    ast.literal_eval(row["tokens"])
                    if isinstance(row["tokens"], str)
                    else list(row["tokens"])
                )
                word_ids = [self.collate_fn.encode_word(t) for t in tokens]
                w2v_word_ids = [self.collate_fn.encode_word_w2v(t) for t in tokens]
                subword_ids = [self.collate_fn.encode_subwords(t) for t in tokens]

                # Build single-example batch
                wt = torch.tensor([word_ids], dtype=torch.long).to(device)
                w2v_wt = torch.tensor([w2v_word_ids], dtype=torch.long).to(device)
                max_sub = max(len(s) for s in subword_ids) if subword_ids else 1
                pad_sub = self.encodings.token2int.get(Encodings.PAD, 0)
                st = torch.full((1, len(tokens), max_sub), pad_sub, dtype=torch.long).to(device)
                for j, subs in enumerate(subword_ids):
                    n = min(len(subs), max_sub)
                    if n > 0:
                        st[0, j, :n] = torch.tensor(subs[:n], dtype=torch.long)

                logits = self.model(wt, w2v_wt, st)  # [1, seq_len, num_labels]
                preds = logits.argmax(dim=-1).squeeze(0).tolist()
                ner_tags = self.autocorrect_labels(
                    [self.ID2LABEL[p] for p in preds]
                )
                results.append(ner_tags)
        return results

    # ------------------------------------------------------------------
    # Public interface (BaseNERModel)
    # ------------------------------------------------------------------

    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        output_path: str,
        metric: str = "f1",
        word2vec_file: Optional[str] = None,
        max_training_epochs: int = 50,
        alpha_ce: float = 0.5,
        alpha_dice: float = 0.5,
        **kwargs,
    ) -> None:
        """Train the BiLSTM model and save the best checkpoint.

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
        word2vec_file:
            Optional path to a word2vec text-format file used to
            initialise the frozen E_w2v embedding layer.  Words not
            found in the file receive zero vectors.
        max_training_epochs:
            Number of training epochs (default: 50).
        alpha_ce:
            Weight for the CrossEntropy loss component (default: 0.5).
        alpha_dice:
            Weight for the Dice loss component (default: 0.5).
        """
        log_device()
        device = get_device()

        self.batch_size = kwargs.get("batch_size", self.BATCH_SIZE)
        self.max_length = kwargs.get("max_sequence_len", self.DEFAULT_MAX_LENGTH)

        # 1. Build encodings from training data only
        self.encodings = Encodings()
        self.encodings.compute(train_data)
        self.collate_fn = Collate(self.encodings)

        # 2. Load word2vec weights — builds a dedicated w2v_word2int from the
        #    full embeddings file so that unseen words can be represented at
        #    inference time.
        w2v_weights, w2v_word2int = _load_w2v_weights(word2vec_file, self.EMB_DIM)
        self.encodings.w2v_word2int = w2v_word2int
        # Re-create collate_fn now that w2v_word2int is populated
        self.collate_fn = Collate(self.encodings)

        # 3. Build datasets and data loaders
        train_dataset = NERDataset(train_data, self.collate_fn)
        val_dataset = NERDataset(val_data, self.collate_fn)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

        # 4. Build model
        self.model = self._build_model(w2v_weights=w2v_weights).to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LR)
        criterion_ce = nn.CrossEntropyLoss(ignore_index=-100)
        criterion_dice = MulticlassDiceLoss(ignore_index=-100)

        best_metric_val = -1.0
        best_state: Optional[dict] = None

        for epoch in range(1, max_training_epochs + 1):
            # --- training ---
            self.model.train()
            total_loss = 0.0
            for word_t, w2v_word_t, sub_t, lbl_t, _lengths in train_loader:
                word_t = word_t.to(device)
                w2v_word_t = w2v_word_t.to(device)
                sub_t = sub_t.to(device)
                lbl_t = lbl_t.to(device)

                optimizer.zero_grad()
                logits = self.model(word_t, w2v_word_t, sub_t)  # [B, L, num_labels]
                loss = alpha_ce * criterion_ce(
                    logits.view(-1, len(self.LABEL_LIST)),
                    lbl_t.view(-1),
                ) + alpha_dice * criterion_dice(logits, lbl_t)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                total_loss += loss.item()

            # --- evaluation ---
            self.model.eval()
            avg_loss = total_loss / max(len(train_loader), 1)
            print(f"[hner] ── epoch {epoch} ──────────────────────  loss={avg_loss:.4f}")

            train_true: List[List[str]] = []
            train_pred: List[List[str]] = []
            val_true: List[List[str]] = []
            val_pred: List[List[str]] = []
            with torch.no_grad():
                for word_t, w2v_word_t, sub_t, lbl_t, lengths in train_loader:
                    word_t = word_t.to(device)
                    w2v_word_t = w2v_word_t.to(device)
                    sub_t = sub_t.to(device)
                    logits = self.model(word_t, w2v_word_t, sub_t)
                    preds = logits.argmax(dim=-1)

                    for i, seq_len in enumerate(lengths.tolist()):
                        true_seq = [
                            self.ID2LABEL[lbl_t[i, t].item()]
                            for t in range(seq_len)
                        ]
                        pred_seq = self.autocorrect_labels([
                            self.ID2LABEL[preds[i, t].item()]
                            for t in range(seq_len)
                        ])
                        train_true.append(true_seq)
                        train_pred.append(pred_seq)

                for word_t, w2v_word_t, sub_t, lbl_t, lengths in val_loader:
                    word_t = word_t.to(device)
                    w2v_word_t = w2v_word_t.to(device)
                    sub_t = sub_t.to(device)
                    logits = self.model(word_t, w2v_word_t, sub_t)
                    preds = logits.argmax(dim=-1)

                    for i, seq_len in enumerate(lengths.tolist()):
                        true_seq = [
                            self.ID2LABEL[lbl_t[i, t].item()]
                            for t in range(seq_len)
                        ]
                        pred_seq = self.autocorrect_labels([
                            self.ID2LABEL[preds[i, t].item()]
                            for t in range(seq_len)
                        ])
                        val_true.append(true_seq)
                        val_pred.append(pred_seq)

            train_metrics = self._compute_seq_metrics(train_true, train_pred)
            val_metrics = self._compute_seq_metrics(val_true, val_pred)
            current = val_metrics.get(metric, val_metrics["f1"])

            _print_metrics_block("Trainset", train_metrics)
            _print_metrics_block("Devset", val_metrics)

            if current > best_metric_val:
                best_metric_val = current
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                print(f"[hner]   → new best {metric}={best_metric_val:.4f}")

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.save(output_path)

    def train_seq_cls(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        output_path: str,
        max_training_epochs: int = 50,
        **kwargs,
    ) -> None:
        """Train BiLSTM sequence classification with label/granular heads."""
        if "label" not in train_data.columns:
            raise ValueError("Sequence-classification training requires a 'label' column.")

        log_device()
        device = get_device()
        os.makedirs(output_path, exist_ok=True)

        self.batch_size = kwargs.get("batch_size", self.BATCH_SIZE)
        self.max_length = kwargs.get("max_sequence_len", self.DEFAULT_MAX_LENGTH)
        word2vec_file = kwargs.get("word2vec_file")

        tokenized_train = [self._prompt_to_tokens(str(v)) for v in train_data["prompt"]]
        train_encoding_data = pd.DataFrame({"tokens": tokenized_train})

        self.encodings = Encodings()
        self.encodings.compute(train_encoding_data)
        self.collate_fn = Collate(self.encodings)

        w2v_weights, w2v_word2int = _load_w2v_weights(word2vec_file, self.EMB_DIM)
        self.encodings.w2v_word2int = w2v_word2int
        self.collate_fn = Collate(self.encodings)

        self._label_classes = sorted(str(v) for v in train_data["label"].unique())
        label2id = {c: i for i, c in enumerate(self._label_classes)}
        has_granular = "granular_label" in train_data.columns
        if has_granular and "granular_label" not in val_data.columns:
            raise ValueError(
                "Validation data must contain 'granular_label' when training includes it."
            )

        self._granular_classes = None
        granular2id = None
        if has_granular and train_data["granular_label"].isnull().any():
            raise ValueError("Training data contains null values in 'granular_label'.")
        if has_granular and val_data["granular_label"].isnull().any():
            raise ValueError("Validation data contains null values in 'granular_label'.")
        if has_granular:
            self._granular_classes = sorted(str(v) for v in train_data["granular_label"].unique())
            granular2id = {c: i for i, c in enumerate(self._granular_classes)}

        seq_collate = SequenceClassificationCollate(self.collate_fn, has_granular=has_granular)
        train_dataset = SequenceClassificationDataset(
            train_data,
            self.collate_fn,
            label2id,
            granular2id,
            max_length=self.max_length,
        )
        val_dataset = SequenceClassificationDataset(
            val_data,
            self.collate_fn,
            label2id,
            granular2id,
            max_length=self.max_length,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=seq_collate,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=seq_collate,
        )

        self.model = self._build_seq_cls_model(
            num_label_classes=len(self._label_classes),
            num_granular_classes=len(self._granular_classes) if self._granular_classes else 0,
            w2v_weights=w2v_weights,
        ).to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LR)
        criterion = nn.CrossEntropyLoss()
        best_metric_val = -1.0
        best_state: Optional[dict] = None

        for epoch in range(1, max_training_epochs + 1):
            self.model.train()
            total_loss = 0.0
            for word_t, w2v_word_t, sub_t, lengths, label_t, granular_t in train_loader:
                word_t = word_t.to(device)
                w2v_word_t = w2v_word_t.to(device)
                sub_t = sub_t.to(device)
                lengths = lengths.to(device)
                label_t = label_t.to(device)
                if granular_t is not None:
                    granular_t = granular_t.to(device)

                optimizer.zero_grad()
                label_logits, granular_logits = self.model(word_t, w2v_word_t, sub_t, lengths)
                loss = criterion(label_logits, label_t)
                if granular_logits is not None and granular_t is not None:
                    loss = loss + criterion(granular_logits, granular_t)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                total_loss += loss.item()

            self.model.eval()
            val_true: List[int] = []
            val_pred: List[int] = []
            gran_true: List[int] = []
            gran_pred: List[int] = []
            with torch.no_grad():
                for word_t, w2v_word_t, sub_t, lengths, label_t, granular_t in val_loader:
                    word_t = word_t.to(device)
                    w2v_word_t = w2v_word_t.to(device)
                    sub_t = sub_t.to(device)
                    lengths = lengths.to(device)
                    label_logits, granular_logits = self.model(word_t, w2v_word_t, sub_t, lengths)

                    pred = torch.argmax(label_logits, dim=-1).cpu().tolist()
                    val_pred.extend(pred)
                    val_true.extend(label_t.tolist())

                    if granular_logits is not None and granular_t is not None:
                        gpred = torch.argmax(granular_logits, dim=-1).cpu().tolist()
                        gran_pred.extend(gpred)
                        gran_true.extend(granular_t.tolist())

            label_metrics = self._compute_classification_metrics(
                val_true, val_pred, len(self._label_classes)
            )
            current = label_metrics["macro_f1"]
            avg_loss = total_loss / max(len(train_loader), 1)

            print(
                f"[hner] ── epoch {epoch} ────────────────────── "
                f"loss={avg_loss:.4f} label_macro_f1={label_metrics['macro_f1']:.4f}"
            )

            if has_granular and self._granular_classes:
                gran_metrics = self._compute_classification_metrics(
                    gran_true, gran_pred, len(self._granular_classes)
                )
                current = (label_metrics["macro_f1"] + gran_metrics["macro_f1"]) / 2.0
                print(
                    f"[hner]   granular_macro_f1={gran_metrics['macro_f1']:.4f} "
                    f"avg_macro_f1={current:.4f}"
                )

            if current > best_metric_val:
                best_metric_val = current
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                print(f"[hner]   → new best macro_f1={best_metric_val:.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.save_seq_cls(output_path)

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run inference and return a ``DataFrame`` with BIO predictions.

        Parameters
        ----------
        data:
            Input ``DataFrame`` with at least ``"ID"`` and ``"tokens"``
            columns.

        Returns
        -------
        pd.DataFrame
            The input columns plus ``"labels"`` and ``"ner_tags"``.
        """
        assert self.model is not None, "Model not loaded. Call load() or train() first."
        device = get_device()
        self.model.to(device)

        predicted_tags = self._predict_sequences(data, device)

        results = []
        for (_, row), ner_tags in zip(data.iterrows(), predicted_tags):
            labels = [self.bio_to_label(tag) for tag in ner_tags]
            results.append({
                "ID": row["ID"],
                "tokens": row["tokens"],
                "labels": str(labels),
                "ner_tags": str(ner_tags),
            })
        return pd.DataFrame(results)

    def predict_seq_cls(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict sequence labels for ``label`` and optional ``granular_label``."""
        if self.model is None or self.collate_fn is None or self._label_classes is None:
            raise RuntimeError("Model not loaded. Call load_seq_cls() first.")
        device = get_device()
        self.model.eval()
        self.model.to(device)
        pad_sub = self.encodings.token2int.get(Encodings.PAD)
        if pad_sub is None:
            raise RuntimeError("Encodings are missing the PAD token for subword IDs.")

        has_granular = self._granular_classes is not None
        results = []
        with torch.no_grad():
            for _, row in data.iterrows():
                tokens = self._prompt_to_tokens(str(row["prompt"]))[: self.max_length]
                if not tokens:
                    tokens = ["<EMPTY>"]
                word_ids = [self.collate_fn.encode_word(t) for t in tokens]
                w2v_word_ids = [self.collate_fn.encode_word_w2v(t) for t in tokens]
                subword_ids = [self.collate_fn.encode_subwords(t) for t in tokens]

                wt = torch.tensor([word_ids], dtype=torch.long).to(device)
                w2v_wt = torch.tensor([w2v_word_ids], dtype=torch.long).to(device)
                lengths = torch.tensor([len(tokens)], dtype=torch.long).to(device)
                max_sub = max(len(s) for s in subword_ids) if subword_ids else 1
                st = torch.full((1, len(tokens), max_sub), pad_sub, dtype=torch.long).to(device)
                for j, subs in enumerate(subword_ids):
                    n = min(len(subs), max_sub)
                    if n > 0:
                        st[0, j, :n] = torch.tensor(subs[:n], dtype=torch.long)

                label_logits, granular_logits = self.model(wt, w2v_wt, st, lengths)
                label_idx = torch.argmax(label_logits, dim=-1).item()

                result: dict = {col: row[col] for col in data.columns}
                result["prediction_label"] = self._label_classes[label_idx]
                if has_granular and granular_logits is not None:
                    gran_idx = torch.argmax(granular_logits, dim=-1).item()
                    result["prediction_granular_label"] = self._granular_classes[gran_idx]
                results.append(result)

        return pd.DataFrame(results)

    def save(self, path: str) -> None:
        """Save model weights, encodings, and config to *path*.

        Parameters
        ----------
        path:
            Directory that will be created if it does not exist.
        """
        os.makedirs(path, exist_ok=True)
        assert self.model is not None and self.encodings is not None

        # Model weights
        torch.save(
            self.model.state_dict(),
            os.path.join(path, self._WEIGHTS_FILE),
        )

        # Encodings
        self.encodings.save(os.path.join(path, self._ENCODINGS_FILE))

        # Architecture config
        w2v_vocab_size = len(self.encodings.w2v_word2int) if self.encodings.w2v_word2int else _MIN_W2V_VOCAB_SIZE
        config = {
            "vocab_size": len(self.encodings.word2int),
            "w2v_vocab_size": w2v_vocab_size,
            "token_vocab_size": len(self.encodings.token2int),
            "emb_dim": self.EMB_DIM,
            "hidden_dim": self.HIDDEN_DIM,
            "num_labels": len(self.LABEL_LIST),
            "num_layers": self.NUM_LAYERS,
        }
        with open(os.path.join(path, self._CONFIG_FILE), "w", encoding="utf-8") as fh:
            yaml.dump(config, fh, default_flow_style=False)

    def save_seq_cls(self, path: str) -> None:
        """Save sequence-classification weights, encodings and class mappings."""
        os.makedirs(path, exist_ok=True)
        assert self.model is not None and self.encodings is not None
        if self._label_classes is None:
            raise RuntimeError("Sequence classifier classes are not initialised.")

        torch.save(
            self.model.state_dict(),
            os.path.join(path, self._WEIGHTS_FILE),
        )
        self.encodings.save(os.path.join(path, self._ENCODINGS_FILE))

        w2v_vocab_size = len(self.encodings.w2v_word2int) if self.encodings.w2v_word2int else _MIN_W2V_VOCAB_SIZE
        config = {
            "vocab_size": len(self.encodings.word2int),
            "w2v_vocab_size": w2v_vocab_size,
            "token_vocab_size": len(self.encodings.token2int),
            "emb_dim": self.EMB_DIM,
            "hidden_dim": self.HIDDEN_DIM,
            "num_layers": self.NUM_LAYERS,
            "num_label_classes": len(self._label_classes),
            "num_granular_classes": len(self._granular_classes) if self._granular_classes else 0,
        }
        with open(os.path.join(path, self._CONFIG_FILE), "w", encoding="utf-8") as fh:
            yaml.dump(config, fh, default_flow_style=False)

        seq_cfg = {
            "label_classes": self._label_classes,
            "has_granular": self._granular_classes is not None,
        }
        if self._granular_classes is not None:
            seq_cfg["granular_classes"] = self._granular_classes
        with open(os.path.join(path, self._SEQ_CLS_CONFIG_FILE), "w", encoding="utf-8") as fh:
            json.dump(seq_cfg, fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BiLSTMModel":
        """Load a previously saved :class:`BiLSTMModel` from *path*.

        Parameters
        ----------
        path:
            Directory previously written by :meth:`save`.

        Returns
        -------
        BiLSTMModel
            A fully initialised model ready for inference.
        """
        instance = cls()
        instance.encodings = Encodings.load(os.path.join(path, cls._ENCODINGS_FILE))
        instance.collate_fn = Collate(instance.encodings)

        with open(os.path.join(path, cls._CONFIG_FILE), "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)

        instance.model = _BiLSTMNER(**config)
        weights_path = os.path.join(path, cls._WEIGHTS_FILE)
        instance.model.load_state_dict(
            torch.load(weights_path, map_location="cpu")
        )
        instance.model.eval()
        return instance

    @classmethod
    def load_seq_cls(cls, path: str) -> "BiLSTMModel":
        """Load a sequence-classification BiLSTM model from *path*."""
        config_path = os.path.join(path, cls._SEQ_CLS_CONFIG_FILE)
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Missing sequence-classification config: {cls._SEQ_CLS_CONFIG_FILE}"
            )
        with open(config_path, "r", encoding="utf-8") as fh:
            seq_config = json.load(fh)

        instance = cls()
        instance._label_classes = seq_config.get("label_classes")
        has_granular = seq_config.get("has_granular", False)
        instance._granular_classes = seq_config.get("granular_classes") if has_granular else None

        instance.encodings = Encodings.load(os.path.join(path, cls._ENCODINGS_FILE))
        instance.collate_fn = Collate(instance.encodings)

        with open(os.path.join(path, cls._CONFIG_FILE), "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)

        instance.model = _BiLSTMSeqCls(**config)
        weights_path = os.path.join(path, cls._WEIGHTS_FILE)
        instance.model.load_state_dict(
            torch.load(weights_path, map_location="cpu")
        )
        instance.model.eval()
        return instance
