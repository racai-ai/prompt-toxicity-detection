"""Vocabulary encodings and batch collation for the BiLSTM NER model.

The :class:`Encodings` class builds word-level and character n-gram
(subword) vocabularies from a dataset and supports saving/loading them
in YAML format.

The :class:`Collate` class is a callable that turns a list of raw
``(word_ids, subword_ids, label_ids)`` examples into padded PyTorch
tensors ready for consumption by the BiLSTM model.

Subword token extraction
------------------------
For each word the following transformation is applied before extracting
n-grams:

1. Prepend ``<`` and append ``>`` to mark word boundaries.
2. Slide a window of length 3, 4 and 5 over the padded character
   sequence and collect every sub-string.

Example
~~~~~~~
For the word ``"book"`` the padded sequence is ``"<book>"`` and the
resulting subword tokens are::

    trigrams  : '<bo', 'boo', 'ook', 'ok>'
    4-grams   : '<boo', 'book', 'ook>'
    5-grams   : '<book', 'book>'
"""

import ast
from collections import Counter
from typing import Dict, List

import yaml


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _extract_subword_tokens(word: str) -> List[str]:
    """Return all boundary-marked character n-grams (3-5) for *word*.

    Parameters
    ----------
    word:
        A single token (already lowercased by the caller).

    Returns
    -------
    list of str
        Trigrams, 4-grams and 5-grams extracted from ``"<{word}>"``.
    """
    padded = f"<{word}>"
    tokens: List[str] = []
    for n in (3, 4, 5):
        for i in range(len(padded) - n + 1):
            tokens.append(padded[i : i + n])
    return tokens


# ---------------------------------------------------------------------------
# Encodings
# ---------------------------------------------------------------------------

class Encodings:
    """Word-level and subword-level vocabulary encodings.

    Attributes
    ----------
    word2int : dict
        Maps words that appear at least twice in the dataset to unique
        integer indices.  Index 0 is reserved for the padding symbol
        ``"<PAD>"`` and index 1 for the unknown-word symbol ``"<UNK>"``.
    token2int : dict
        Same structure as :attr:`word2int` but for character n-gram
        subword tokens.
    w2v_word2int : dict
        Mapping from word strings to integer indices built from the
        external word2vec embeddings file.  This covers the entire
        vocabulary of the embeddings file, not just words seen in the
        training corpus.  Index 0 is ``"<PAD>"``, index 1 is ``"<UNK>"``.
        Empty when no word2vec file was provided.
    """

    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(self) -> None:
        self.word2int: Dict[str, int] = {}
        self.token2int: Dict[str, int] = {}
        self.w2v_word2int: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def compute(self, dataset) -> None:
        """Build :attr:`word2int` and :attr:`token2int` from *dataset*.

        Only words / subword tokens that appear **at least twice** across
        the entire dataset receive an entry.  All comparisons are
        case-insensitive (tokens are lower-cased before counting).

        Parameters
        ----------
        dataset:
            A ``pandas.DataFrame`` that must contain a ``"tokens"``
            column whose values are either Python lists or
            string-encoded lists (e.g. ``"['Hello', ',', 'world']"``).
        """
        word_counts: Counter = Counter()
        token_counts: Counter = Counter()

        for tokens in dataset["tokens"]:
            if isinstance(tokens, str):
                tokens = ast.literal_eval(tokens)
            for word in tokens:
                word = word.lower()
                word_counts[word] += 1
                for ngram in _extract_subword_tokens(word):
                    token_counts[ngram] += 1

        # Build word2int (PAD=0, UNK=1, then alphabetical order)
        self.word2int = {self.PAD: 0, self.UNK: 1}
        for word in sorted(word_counts):
            if word_counts[word] >= 2:
                self.word2int[word] = len(self.word2int)

        # Build token2int (same reserved indices)
        self.token2int = {self.PAD: 0, self.UNK: 1}
        for token in sorted(token_counts):
            if token_counts[token] >= 2:
                self.token2int[token] = len(self.token2int)

    def save(self, path: str) -> None:
        """Persist the encodings to a YAML file at *path*.

        Parameters
        ----------
        path:
            File path (including ``.yaml`` extension) where the data
            will be written.
        """
        data = {
            "word2int": self.word2int,
            "token2int": self.token2int,
            "w2v_word2int": self.w2v_word2int,
        }
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(data, fh, allow_unicode=True, default_flow_style=False)

    @classmethod
    def load(cls, path: str) -> "Encodings":
        """Load encodings from the YAML file at *path*.

        Parameters
        ----------
        path:
            Path to a YAML file previously written by :meth:`save`.

        Returns
        -------
        Encodings
            A fully populated :class:`Encodings` instance.
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        enc = cls()
        enc.word2int = data.get("word2int", {})
        enc.token2int = data.get("token2int", {})
        enc.w2v_word2int = data.get("w2v_word2int", {})
        return enc


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

class Collate:
    """Collate function for batching BiLSTM NER examples.

    Converts a list of ``(word_ids, subword_ids, label_ids)`` tuples
    produced by :class:`~models.bilstm.NERDataset` into padded
    PyTorch tensors.

    Parameters
    ----------
    encodings:
        A fitted :class:`Encodings` instance that supplies the
        vocabulary mappings.
    """

    def __init__(self, encodings: Encodings) -> None:
        self.encodings = encodings

    # ------------------------------------------------------------------
    # Encoding helpers (used when building the dataset)
    # ------------------------------------------------------------------

    def encode_word(self, word: str) -> int:
        """Return the integer index for *word* (lowercased).

        Falls back to the ``<UNK>`` index when the word is out-of-vocab.
        """
        unk = self.encodings.word2int.get(Encodings.UNK, 1)
        return self.encodings.word2int.get(word.lower(), unk)

    def encode_word_w2v(self, word: str) -> int:
        """Return the w2v integer index for *word* (lowercased).

        Uses :attr:`~Encodings.w2v_word2int` which covers the full
        vocabulary of the external word2vec file.  Falls back to the
        ``<UNK>`` index when the word is not present in the embeddings
        file (or when no embeddings file was loaded).
        """
        if not self.encodings.w2v_word2int:
            return 1  # UNK when no w2v vocabulary available
        unk = self.encodings.w2v_word2int.get(Encodings.UNK, 1)
        return self.encodings.w2v_word2int.get(word.lower(), unk)

    def encode_subwords(self, word: str) -> List[int]:
        """Return a list of integer indices for the subword tokens of *word*.

        Falls back to the ``<UNK>`` index for each out-of-vocab n-gram.
        """
        unk = self.encodings.token2int.get(Encodings.UNK, 1)
        return [
            self.encodings.token2int.get(ng, unk)
            for ng in _extract_subword_tokens(word.lower())
        ]

    # ------------------------------------------------------------------
    # __call__ — used as the DataLoader collate_fn
    # ------------------------------------------------------------------

    def __call__(self, batch):
        """Collate *batch* into padded tensors.

        Parameters
        ----------
        batch:
            A list of ``(word_ids, w2v_word_ids, subword_ids, label_ids)``
            tuples where:

            * ``word_ids``      – ``List[int]``, one per token (training
              vocabulary index).
            * ``w2v_word_ids``  – ``List[int]``, one per token (word2vec
              vocabulary index).
            * ``subword_ids``   – ``List[List[int]]``, one inner list per
              token, each inner list contains the n-gram indices for
              that token.
            * ``label_ids``     – ``List[int]``, one BIO label index per
              token.

        Returns
        -------
        tuple
            ``(word_tensor, w2v_word_tensor, subword_tensor, label_tensor, lengths)``

            * ``word_tensor``      – ``LongTensor[batch, max_len]``
            * ``w2v_word_tensor``  – ``LongTensor[batch, max_len]``
            * ``subword_tensor``   – ``LongTensor[batch, max_len, max_sub]``
            * ``label_tensor``     – ``LongTensor[batch, max_len]``
              (padding positions set to ``-100`` so they are ignored by
              cross-entropy loss)
            * ``lengths``          – ``LongTensor[batch]`` with the
              unpadded sequence length for each example.
        """
        import torch

        word_seqs, w2v_word_seqs, subword_seqs, label_seqs = zip(*batch)

        lengths = [len(w) for w in word_seqs]
        max_len = max(lengths)

        pad_word = self.encodings.word2int.get(Encodings.PAD, 0)
        pad_w2v = self.encodings.w2v_word2int.get(Encodings.PAD, 0) if self.encodings.w2v_word2int else 0
        pad_sub = self.encodings.token2int.get(Encodings.PAD, 0)

        # Maximum number of subword tokens any single word can have
        max_sub = max(
            len(subs)
            for sub_seq in subword_seqs
            for subs in sub_seq
        ) if any(sub_seq for sub_seq in subword_seqs) else 1

        batch_size = len(batch)
        word_tensor = torch.full((batch_size, max_len), pad_word, dtype=torch.long)
        w2v_word_tensor = torch.full((batch_size, max_len), pad_w2v, dtype=torch.long)
        subword_tensor = torch.full(
            (batch_size, max_len, max_sub), pad_sub, dtype=torch.long
        )
        label_tensor = torch.full((batch_size, max_len), -100, dtype=torch.long)

        for i, (word_ids, w2v_word_ids, sub_ids, lbl_ids) in enumerate(
            zip(word_seqs, w2v_word_seqs, subword_seqs, label_seqs)
        ):
            seq_len = len(word_ids)
            word_tensor[i, :seq_len] = torch.tensor(word_ids, dtype=torch.long)
            w2v_word_tensor[i, :seq_len] = torch.tensor(w2v_word_ids, dtype=torch.long)
            label_tensor[i, :seq_len] = torch.tensor(lbl_ids, dtype=torch.long)
            for j, subs in enumerate(sub_ids):
                n = min(len(subs), max_sub)
                if n > 0:
                    subword_tensor[i, j, :n] = torch.tensor(
                        subs[:n], dtype=torch.long
                    )

        return (
            word_tensor,
            w2v_word_tensor,
            subword_tensor,
            label_tensor,
            torch.tensor(lengths, dtype=torch.long),
        )
