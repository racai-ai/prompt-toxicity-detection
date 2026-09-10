"""LLM-based NER tagger using a locally hosted Ollama model.

During **training**, GEPA (via :mod:`gepa_ner.llm_prompt_adapter`) is used
to optimise a DSPy annotation program.  Each candidate program is evaluated
by running it on a sample of validation sentences and computing the relaxed
F1 against gold BIO tags.  The best-performing program code is saved to the
output directory.

During **inference**, the saved DSPy program is executed with DSPy + Ollama.
If no saved program exists (zero-shot mode) the built-in seed program is used.

Output format expected from the LLM (one line per input token)::

    TOKEN<TAB>LABEL

where ``LABEL`` is one of:

* ``-``               – token is not part of any entity
* ``ClinicalImpacts`` – clinical consequence of opioid use
* ``SocialImpacts``   – social consequence of opioid use

These flat labels are converted to BIO tags internally (using the standard
B-/I- continuation logic from :meth:`BaseNERModel.labels_to_bio`).
"""

from __future__ import annotations

import ast
import json
import os
import sys
from typing import Optional

import dspy
import pandas as pd

from models.base import BaseNERModel

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

DEFAULT_OLLAMA_MODEL = "gemma3:27b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

_FLAT_LABELS = frozenset({"-", "ClinicalImpacts", "SocialImpacts"})

_PROGRAM_FILE = "llm_ner_program.py"
_CONFIG_FILE = "llm_ner_config.json"


# ------------------------------------------------------------------
# Parsing helpers (used both here and in the GEPA adapter)
# ------------------------------------------------------------------

def parse_flat_output(response: str, tokens: list) -> list:
    """Parse ``TOKEN\\tLABEL`` lines returned by the LLM.

    Returns a list of flat labels (one per token in *tokens*).  Any
    unrecognised or missing labels fall back to ``"-"``.
    """
    flat: list = []
    lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
    for i, _ in enumerate(tokens):
        if i < len(lines):
            parts = lines[i].split("\t", 1)
            if len(parts) == 2:
                label = parts[1].strip()
            else:
                # Fall back to whitespace-split
                parts2 = lines[i].split(None, 1)
                label = parts2[1].strip() if len(parts2) == 2 else "-"
            if label not in _FLAT_LABELS:
                label = "-"
        else:
            label = "-"
        flat.append(label)
    return flat


def flat_to_bio(flat_labels: list) -> list:
    """Convert flat labels (-, ClinicalImpacts, SocialImpacts) to BIO tags."""
    bio: list = []
    prev = "-"
    for label in flat_labels:
        if label == "-":
            bio.append("O")
            prev = "-"
        elif label == prev:
            bio.append(f"I-{label}")
        else:
            bio.append(f"B-{label}")
            prev = label
    return bio


# ------------------------------------------------------------------
# Default (seed) DSPy program
# ------------------------------------------------------------------

SEED_NER_PROGRAM = r'''import dspy


class NERAnnotation(dspy.Signature):
    """You are an expert at Named Entity Recognition for first-person social
    media posts about opioid use and misuse.

    Task: Annotate each token with its entity label.

    Entity types:
      ClinicalImpacts  - self-reported clinical consequences of opioid use.
                         Examples: withdrawal, overdose, depression, anxiety,
                         rehab, side effects, hospitalization, detox, nausea.
      SocialImpacts    - self-reported social consequences of opioid use.
                         Examples: job loss, relationship breakdown, legal
                         troubles, family conflict, financial problems,
                         homelessness, custody loss, isolation.
      -                - the token is NOT part of any entity.

    Rules:
      1. Output EXACTLY one line per token in the input.
      2. Each output line format: TOKEN<TAB>LABEL
      3. LABEL must be exactly one of:  -  ClinicalImpacts  SocialImpacts
      4. Only annotate FIRST-PERSON, SELF-REPORTED impacts.
      5. Drug names are NEVER entities.
      6. Do NOT add any extra text, blank lines, or explanations.
    """

    tokens_input: str = dspy.InputField(
        desc="Tokens to annotate, one per line."
    )
    annotations: str = dspy.OutputField(
        desc=(
            "Annotations: one line per token, format  TOKEN<TAB>LABEL  "
            "where LABEL is one of: -, ClinicalImpacts, SocialImpacts. "
            "Output EXACTLY as many lines as there are input tokens."
        )
    )


class NERAnnotator(dspy.Module):
    def __init__(self):
        self.annotate = dspy.Predict(NERAnnotation)

    def forward(self, tokens_input: str):
        return self.annotate(tokens_input=tokens_input)


program = NERAnnotator()
'''


# ------------------------------------------------------------------
# LLMNERModel
# ------------------------------------------------------------------

class LLMNERModel(BaseNERModel):
    """NER tagger backed by a locally hosted Ollama LLM.

    Parameters
    ----------
    ollama_model:
        Ollama model identifier, e.g. ``"gemma3:27b"`` or ``"llama3:8b"``.
    ollama_base_url:
        Base URL of the Ollama server (default: ``http://localhost:11434``).
    program_code:
        Python source code (string) of the DSPy annotation program.
        If ``None`` the built-in :data:`SEED_NER_PROGRAM` is used.
    """

    def __init__(
        self,
        ollama_model: str = DEFAULT_OLLAMA_MODEL,
        ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
        program_code: Optional[str] = None,
    ):
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        self.program_code = program_code or SEED_NER_PROGRAM
        self._dspy_program: Optional[dspy.Module] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_lm(self) -> dspy.LM:
        """Return a configured DSPy LM pointing at Ollama."""
        return dspy.LM(
            model=f"ollama/{self.ollama_model}",
            api_base=self.ollama_base_url,
            api_key="ollama",
            temperature=0.0,
            cache=False,
        )

    def _load_program(self) -> dspy.Module:
        """Compile and return the DSPy annotation program."""
        if self._dspy_program is not None:
            return self._dspy_program

        ctx: dict = {}
        exec(self.program_code, ctx)  # noqa: S102
        prog = ctx.get("program")
        if prog is None or not isinstance(prog, dspy.Module):
            raise RuntimeError(
                "The LLM NER program code must assign a dspy.Module to `program`."
            )
        lm = self._build_lm()
        prog.set_lm(lm)
        self._dspy_program = prog
        return prog

    def _annotate_tokens(self, tokens: list) -> tuple:
        """Annotate a single sentence; return (flat_labels, ner_tags)."""
        prog = self._load_program()
        tokens_input = "\n".join(tokens)
        try:
            pred = prog(tokens_input=tokens_input)
            raw = getattr(pred, "annotations", "") or ""
        except Exception as exc:  # noqa: BLE001
            print(f"[llm_ner] LLM call failed: {exc}", file=sys.stderr)
            raw = ""
        flat = parse_flat_output(raw, tokens)
        ner_tags = self.autocorrect_labels(flat_to_bio(flat))
        return flat, ner_tags

    # ------------------------------------------------------------------
    # BaseNERModel interface
    # ------------------------------------------------------------------

    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        output_path: str,
        metric: str = "relaxed_f1",
        **kwargs,
    ) -> None:
        """Optimise the prompt program with GEPA and save the result.

        Accepted extra keyword arguments (forwarded to
        :func:`gepa_ner.llm_prompt_adapter.run_llm_prompt_evolution`):

        ``ollama_model`` : str
            Ollama model to use for optimisation (defaults to
            ``self.ollama_model``).
        ``ollama_base_url`` : str
            Ollama base URL (defaults to ``self.ollama_base_url``).
        ``max_metric_calls`` : int
            GEPA budget — total ``evaluate()`` calls (default: 30).
        ``eval_sample_size`` : int
            Number of validation sentences per evaluation (default: 50).
        ``run_dir`` : str
            Directory for GEPA checkpoints (default:
            ``<output_path>/gepa_runs``).
        """
        from gepa_ner.llm_prompt_adapter import run_llm_prompt_evolution

        ollama_model = kwargs.pop("ollama_model", self.ollama_model)
        ollama_base_url = kwargs.pop("ollama_base_url", self.ollama_base_url)
        run_dir = kwargs.pop("run_dir", os.path.join(output_path, "gepa_runs"))

        best_code = run_llm_prompt_evolution(
            train_data=train_data,
            val_data=val_data,
            ollama_model=ollama_model,
            ollama_base_url=ollama_base_url,
            run_dir=run_dir,
            **kwargs,
        )

        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        self.program_code = best_code
        self._dspy_program = None  # force reload on next predict()
        self.save(output_path)
        print(f"[llm_ner] Best program saved to: {output_path}")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run inference on *data* and return a DataFrame with predictions."""
        results = []
        total = len(data)
        for i, (_, row) in enumerate(data.iterrows(), 1):
            print(f"[llm_ner] Annotating {i}/{total} ...", end="\r", flush=True)
            tokens = (
                ast.literal_eval(row["tokens"])
                if isinstance(row["tokens"], str)
                else list(row["tokens"])
            )
            sentence_id = row["ID"]
            _, ner_tags = self._annotate_tokens(tokens)
            labels = [self.bio_to_label(tag) for tag in ner_tags]
            results.append(
                {
                    "ID": sentence_id,
                    "tokens": str(tokens),
                    "labels": str(labels),
                    "ner_tags": str(ner_tags),
                }
            )
        print()
        return pd.DataFrame(results)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, _PROGRAM_FILE), "w", encoding="utf-8") as fh:
            fh.write(self.program_code)
        config = {
            "ollama_model": self.ollama_model,
            "ollama_base_url": self.ollama_base_url,
        }
        with open(os.path.join(path, _CONFIG_FILE), "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "LLMNERModel":
        config_path = os.path.join(path, _CONFIG_FILE)
        program_path = os.path.join(path, _PROGRAM_FILE)

        config: dict = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as fh:
                config = json.load(fh)

        program_code: Optional[str] = None
        if os.path.exists(program_path):
            with open(program_path, "r", encoding="utf-8") as fh:
                program_code = fh.read()

        return cls(
            ollama_model=config.get("ollama_model", DEFAULT_OLLAMA_MODEL),
            ollama_base_url=config.get("ollama_base_url", DEFAULT_OLLAMA_BASE_URL),
            program_code=program_code,
        )
