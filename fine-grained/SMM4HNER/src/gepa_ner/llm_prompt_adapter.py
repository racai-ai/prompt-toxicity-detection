"""GEPA adapter for LLM NER prompt optimisation.

Unlike the augmentation adapter (:class:`gepa_ner.adapter.NERSyntheticAdapter`),
this adapter directly evaluates how well the LLM — driven by the candidate
DSPy program — annotates a sample of validation sentences.

Evaluation pipeline
-------------------
1. Build the candidate DSPy annotation program from its Python code string.
2. Run the program on a random sample of validation sentences.
3. Parse the LLM's flat-label output (-, ClinicalImpacts, SocialImpacts).
4. Convert flat labels to BIO tags and auto-correct any invalid sequences.
5. Compute the relaxed F1 against gold BIO tags.
6. Return the relaxed F1 as a uniform score for every example in the batch.

The high-level entry point :func:`run_llm_prompt_evolution` wires GEPA,
DSPy, and Ollama together and returns the code string of the best program
found during the optimisation run.
"""

from __future__ import annotations

import ast
import os
import sys
import traceback
from typing import Any

import dspy
import pandas as pd

import gepa
from gepa import EvaluationBatch, GEPAAdapter

from data.evaluation import calculate_f1_per_entity_covering_all
from models.base import BaseNERModel
from models.llm_ner_model import (
    SEED_NER_PROGRAM,
    flat_to_bio,
    parse_flat_output,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _compute_relaxed_f1(
    gold_batch: list,
    pred_batch: list,
) -> float:
    """Overall relaxed F1 over a batch of BIO-tag sequences."""
    try:
        result = calculate_f1_per_entity_covering_all(gold_batch, pred_batch)
        return result.get("Overall", {}).get("F1-Score", 0.0)
    except Exception:  # noqa: BLE001
        return 0.0


# ------------------------------------------------------------------
# GEPA Adapter
# ------------------------------------------------------------------

class LLMNERPromptAdapter(GEPAAdapter):
    """GEPAAdapter that scores a DSPy NER annotation program by running it on
    the validation set and computing the relaxed F1 against gold labels.

    Parameters
    ----------
    task_lm:
        DSPy LM used to *run* the annotation program (the Ollama model).
    reflection_lm:
        DSPy LM used by GEPA to *propose* improved programs.
    val_df:
        Validation DataFrame with ``tokens`` and ``ner_tags`` columns.
    eval_sample_size:
        Number of validation sentences sampled per evaluation.
    failure_score:
        Score returned when a candidate program fails to compile or run.
    """

    def __init__(
        self,
        task_lm: dspy.LM,
        reflection_lm: dspy.LM,
        val_df: pd.DataFrame,
        eval_sample_size: int = 50,
        failure_score: float = 0.0,
    ) -> None:
        self.task_lm = task_lm
        self.reflection_lm = reflection_lm
        self.val_df = val_df
        self.eval_sample_size = min(eval_sample_size, len(val_df))
        self.failure_score = failure_score
        self._eval_count = 0

    # ------------------------------------------------------------------

    def build_program(
        self, candidate: dict[str, str]
    ) -> tuple[dspy.Module, None] | tuple[None, str]:
        """Compile and instantiate the candidate's Python program string."""
        src = candidate["program"]
        try:
            compile(src, "<candidate>", "exec")
        except SyntaxError as exc:
            return None, f"SyntaxError: {exc}\n{traceback.format_exc()}"

        ctx: dict[str, Any] = {}
        try:
            exec(src, ctx)  # noqa: S102
        except Exception as exc:  # noqa: BLE001
            return None, f"Exec error: {exc}\n{traceback.format_exc()}"

        prog = ctx.get("program")
        if prog is None:
            return None, "Code did not define a `program` variable."
        if not isinstance(prog, dspy.Module):
            return None, f"`program` is {type(prog)!r}, expected dspy.Module."
        prog.set_lm(self.task_lm)
        return prog, None

    # ------------------------------------------------------------------

    def _run_program_on_sample(self, program: dspy.Module) -> float:
        """Run *program* on a random sample of validation sentences.

        Returns the relaxed F1 against gold BIO tags.
        """
        sample = self.val_df.sample(
            n=self.eval_sample_size,
            replace=False,
            random_state=self._eval_count,
        )

        gold_batch: list = []
        pred_batch: list = []

        for _, row in sample.iterrows():
            tokens = (
                ast.literal_eval(row["tokens"])
                if isinstance(row["tokens"], str)
                else list(row["tokens"])
            )
            gold_tags = (
                ast.literal_eval(row["ner_tags"])
                if isinstance(row["ner_tags"], str)
                else list(row["ner_tags"])
            )

            tokens_input = "\n".join(tokens)
            try:
                pred = program(tokens_input=tokens_input)
                raw = getattr(pred, "annotations", "") or ""
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[llm_prompt eval #{self._eval_count}]   inference error: {exc}",
                    file=sys.stderr,
                )
                raw = ""

            flat = parse_flat_output(raw, tokens)
            bio = BaseNERModel.autocorrect_labels(flat_to_bio(flat))
            gold_batch.append(gold_tags)
            pred_batch.append(bio)

        return _compute_relaxed_f1(gold_batch, pred_batch)

    # ------------------------------------------------------------------
    # GEPAAdapter protocol
    # ------------------------------------------------------------------

    def evaluate(
        self,
        batch: list,
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        self._eval_count += 1
        n = len(batch)

        program, err = self.build_program(candidate)
        if program is None:
            print(
                f"[llm_prompt eval #{self._eval_count}] BUILD FAILED: "
                f"{(err or '').splitlines()[0]}",
                file=sys.stderr,
            )
            return EvaluationBatch(
                outputs=[None] * n,
                scores=[self.failure_score] * n,
                trajectories=err if capture_traces else None,
            )

        relaxed_f1 = self._run_program_on_sample(program)
        print(
            f"[llm_prompt eval #{self._eval_count}] "
            f"relaxed_f1={relaxed_f1:.4f} "
            f"(sample={self.eval_sample_size})"
        )

        outputs = [{"relaxed_f1": relaxed_f1}] * n
        scores = [relaxed_f1] * n
        trajectories = {"relaxed_f1": relaxed_f1} if capture_traces else None

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )


# ------------------------------------------------------------------
# High-level entry point
# ------------------------------------------------------------------

def run_llm_prompt_evolution(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    *,
    ollama_model: str = "gemma3:27b",
    ollama_base_url: str = "http://localhost:11434",
    max_metric_calls: int = 30,
    eval_sample_size: int = 50,
    run_dir: str = "gepa_runs/llm_ner_prompt",
) -> str:
    """Run GEPA to optimise the NER annotation prompt program.

    Parameters
    ----------
    train_data:
        Training DataFrame (currently unused by the evaluation; reserved for
        future use, e.g. few-shot example selection).
    val_data:
        Validation DataFrame with ``tokens`` and ``ner_tags`` columns.
    ollama_model:
        Ollama model identifier (e.g. ``"gemma3:27b"``).
    ollama_base_url:
        Ollama server base URL (default: ``http://localhost:11434``).
    max_metric_calls:
        GEPA budget — total number of ``evaluate()`` calls.
    eval_sample_size:
        Number of validation sentences sampled per evaluation call.
    run_dir:
        Directory for GEPA checkpoints and the best program file.

    Returns
    -------
    str
        Python source code of the best NER annotation program found.
    """
    task_lm = dspy.LM(
        model=f"ollama/{ollama_model}",
        api_base=ollama_base_url,
        api_key="ollama",
        temperature=0.0,
        cache=False,
    )
    reflection_lm = dspy.LM(
        model=f"ollama/{ollama_model}",
        api_base=ollama_base_url,
        api_key="ollama",
        temperature=0.7,
        cache=False,
    )

    # Build dspy.Example list for GEPA trainset / valset
    dev_examples: list[dspy.Example] = []
    for _, row in val_data.iterrows():
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
        ex = dspy.Example(
            tokens=tokens,
            ner_tags=ner_tags,
            ID=row["ID"],
        ).with_inputs("tokens", "ner_tags", "ID")
        dev_examples.append(ex)

    adapter = LLMNERPromptAdapter(
        task_lm=task_lm,
        reflection_lm=reflection_lm,
        val_df=val_data,
        eval_sample_size=min(eval_sample_size, len(val_data)),
    )

    os.makedirs(run_dir, exist_ok=True)

    print(
        f"[llm_prompt] Starting GEPA optimisation — "
        f"budget={max_metric_calls}, "
        f"eval_sample={eval_sample_size}, "
        f"model={ollama_model}"
    )

    result = gepa.optimize(
        seed_candidate={"program": SEED_NER_PROGRAM},
        trainset=dev_examples,
        valset=dev_examples,
        adapter=adapter,
        max_metric_calls=max_metric_calls,
        candidate_selection_strategy="pareto",
        frontier_type="instance",
        module_selector="all",
        reflection_minibatch_size=3,
        run_dir=run_dir,
        use_cloudpickle=True,
        display_progress_bar=True,
    )

    best_code: str = result.best_candidate["program"]
    best_idx = result.best_idx
    best_score = result.val_aggregate_scores[best_idx]

    best_path = os.path.join(run_dir, "best_ner_program.py")
    with open(best_path, "w", encoding="utf-8") as fh:
        fh.write(best_code)

    print(
        f"[llm_prompt] Evolution complete.  "
        f"Best relaxed_f1={best_score:.4f}  "
        f"program → {best_path}"
    )

    return best_code
