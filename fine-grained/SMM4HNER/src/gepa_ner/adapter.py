"""Custom GEPAAdapter that evolves a DSPy NER-data-augmentation program.

A baseline BERT adapter is trained once on the real training data at
initialisation.  Each candidate is then evaluated as follows:

  1. Build the DSPy program from the candidate code string.
  2. Run the program on training examples to produce augmented NER examples.
  3. Parse the program outputs into tokens + BIO tags.
  4. Fine-tune the baseline checkpoint on the augmented data only.
  5. Evaluate on the full dev set and compute the relaxed-F1
     improvement (delta) over the baseline.
  6. Return the delta as a uniform score for every example in the
     GEPA batch.
"""

from __future__ import annotations

import random
import sys
import traceback
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import dspy
import pandas as pd
import yaml
from data.evaluation import calculate_f1_per_entity_covering_all

from gepa import EvaluationBatch, GEPAAdapter
from gepa.proposer.reflective_mutation.base import Signature

from gepa_ner.evaluator import (
    LABEL_LIST,
    LABEL2ID,
    EvalResult,
    autocorrect_labels,
    finetune_from_checkpoint,
    train_baseline,
)
from gepa_ner.guidelines import ANNOTATION_GUIDELINES


# ------------------------------------------------------------------
# Parsing helpers
# ------------------------------------------------------------------

def _parse_annotated_output(raw: str) -> tuple[list[str], list[str]] | None:
    """Parse ``TOKEN\\tNER_TAG`` lines into (tokens, ner_tags).

    Returns *None* when the output cannot be parsed into at least one
    valid token.
    """
    tokens: list[str] = []
    tags: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        tok = parts[0].strip()
        tag = parts[1].strip()
        if tag not in LABEL2ID:
            tag = "O"
        tokens.append(tok)
        tags.append(tag)
    if not tokens:
        return None
    tags = autocorrect_labels(tags)
    return tokens, tags


def _per_example_relaxed_f1(gold: list[str], pred: list[str]) -> float:
    """Relaxed overlap-based F1 for a single example."""
    try:
        result = calculate_f1_per_entity_covering_all([gold], [pred])
        return result.get("Overall", {}).get("F1-Score", 0.0)
    except Exception:
        return 0.0


# ------------------------------------------------------------------
# DSPy program proposal signature (inlined to avoid broken import
# chain through gepa.adapters.dspy_full_program_adapter.__init__
# which pulls in dspy.teleprompt.bootstrap_trace)
# ------------------------------------------------------------------

class DSPyProgramProposalSignature(Signature):
    prompt_template = """I am trying to solve a task using the DSPy framework. Here's a comprehensive overview of DSPy concepts to guide your improvements:

Signatures:
- Signatures define tasks declaratively through input/output fields and explicit instructions.
- They serve as blueprints for what the LM needs to accomplish.

Signature Types:
- Simple signatures: Specified as strings like "input1, ..., inputN -> output1, ..., outputM" (e.g., "topic -> tweet").
- Typed signatures: Create a subclass of dspy.Signature with a detailed docstring that includes task instructions, common pitfalls, edge cases, and successful strategies. Define fields using dspy.InputField(desc="...", type=...) and dspy.OutputField(desc="...", type=...) with pydantic types such as str, List[str], Literal["option1", "option2"], or custom classes.

Modules:
- Modules specify __how__ to solve the task defined by a signature.
- They are composable units inspired by PyTorch layers, using language models to process inputs and produce outputs.
- Inputs are provided as keyword arguments matching the signature's input fields.
- Outputs are returned as dspy.Prediction objects containing the signature's output fields.
- Key built-in modules:
  - dspy.Predict(signature): Performs a single LM call to directly generate the outputs from the inputs.
  - dspy.ChainOfThought(signature): Performs a single LM call that first generates a reasoning chain, then the outputs (adds a 'reasoning' field to the prediction).
  - Other options: dspy.ReAct(signature) for reasoning and acting, or custom chains.
- Custom modules: Subclass dspy.Module. In __init__, compose sub-modules (e.g., other Predict or ChainOfThought instances). In forward(self, **kwargs), define the data flow: call sub-modules, execute Python logic if needed, and return dspy.Prediction with the output fields.

Example Usage:
```
# Simple signature
simple_signature = "question -> answer"

# Typed signature
class ComplexSignature(dspy.Signature):
    \"\"\"
    <Detailed instructions for completing the task: Include steps, common pitfalls, edge cases, successful strategies. Include domain knowledge...>
    \"\"\"
    question: str = dspy.InputField(desc="The question to answer")
    answer: str = dspy.OutputField(desc="Concise and accurate answer")

# Built-in module
simple_program = dspy.Predict(simple_signature)  # or dspy.ChainOfThought(ComplexSignature)

# Custom module
class ComplexModule(dspy.Module):
    def __init__(self):
        self.reasoner = dspy.ChainOfThought("question -> intermediate_answer")
        self.finalizer = dspy.Predict("intermediate_answer -> answer")

    def forward(self, question: str):
        intermediate = self.reasoner(question=question)
        final = self.finalizer(intermediate_answer=intermediate.intermediate_answer)
        return dspy.Prediction(answer=final.answer, reasoning=intermediate.reasoning) # dspy.ChainOfThought returns 'reasoning' in addition to the signature outputs.

complex_program = ComplexModule()
```

DSPy Improvement Strategies:
1. Analyze traces for LM overload: If a single call struggles (e.g., skips steps or hallucinates), decompose into multi-step modules with ChainOfThought or custom logic for stepwise reasoning.
2. Avoid over-decomposition: If the program is too fragmented, consolidate related steps into fewer modules for efficiency and coherence.
3. Refine signatures: Enhance docstrings with actionable guidance from traces-address specific errors, incorporate domain knowledge, document edge cases, and suggest reasoning patterns. Ensure docstrings are self-contained, as the LM won't have access external traces during runtime.
4. Balance LM and Python: Use Python for symbolic/logical operations (e.g., loops, conditionals); delegate complex reasoning or generation to LM calls.
5. Incorporate control flow: Add loops, conditionals, sub-modules in custom modules if the task requires iteration (e.g., multi-turn reasoning, selection, voting, etc.).
6. Leverage LM strengths: For code-heavy tasks, define signatures with 'code' outputs, extract and execute the generated code in the module's forward pass.

Here's my current code:
```
<curr_program>
```

Here is the execution trace of the current code on example inputs, their outputs, and detailed feedback on improvements:
```
<dataset_with_feedback>
```

Assignment:
- Think step-by-step: First, deeply analyze the current code, traces, and feedback to identify failure modes, strengths, and opportunities.
- Create a concise checklist (3-7 bullets) outlining your high-level improvement plan, focusing on conceptual changes (e.g., "Decompose step X into a multi-stage module").
- Then, propose a drop-in replacement code that instantiates an improved 'program' object.
- Ensure the code is modular, efficient, and directly addresses feedback.
- Output everything in a single code block using triple backticks-no additional explanations, comments, or language markers outside the block.
- The code must be a valid, self-contained Python script with all necessary imports, definitions, and assignment to 'program'.

Output Format:
- Start with the checklist in plain text (3-7 short bullets).
- Follow immediately with one code block in triple backticks containing the complete Python code, including assigning a `program` object."""
    input_keys: ClassVar[list[str]] = ["curr_program", "dataset_with_feedback"]
    output_keys: ClassVar[list[str]] = ["new_program"]

    @classmethod
    def prompt_renderer(cls, input_dict: dict[str, Any]) -> str:
        curr_program = input_dict["curr_program"]
        if not isinstance(curr_program, str):
            raise TypeError("curr_program must be a string")

        dataset = input_dict["dataset_with_feedback"]
        if not isinstance(dataset, list):
            raise TypeError("dataset_with_feedback must be a list")

        def format_samples(samples):
            yaml_str = yaml.dump(samples, sort_keys=False, default_flow_style=False, indent=2)
            return yaml_str

        prompt = cls.prompt_template
        prompt = prompt.replace("<curr_program>", curr_program)
        prompt = prompt.replace("<dataset_with_feedback>", format_samples(dataset))
        return prompt

    @staticmethod
    def _strip_lang_tag(code: str) -> str:
        """Remove a leading language tag (e.g. ``python``) from extracted code."""
        first_nl = code.find("\n")
        if first_nl == -1:
            return code
        first_line = code[:first_nl].strip()
        if first_line.isalpha():
            return code[first_nl + 1:]
        return code

    @staticmethod
    def output_extractor(lm_out: str) -> dict[str, str]:
        new_instruction = None
        if lm_out.count("```") >= 2:
            start = lm_out.find("```")
            end = lm_out.rfind("```")
            if start >= end:
                new_instruction = lm_out
            if start == -1 or end == -1:
                new_instruction = lm_out
            else:
                raw = lm_out[start + 3 : end].strip()
                new_instruction = DSPyProgramProposalSignature._strip_lang_tag(raw)
        else:
            lm_out = lm_out.strip()
            if lm_out.startswith("```"):
                lm_out = lm_out[3:]
            if lm_out.endswith("```"):
                lm_out = lm_out[:-3]
            lm_out = DSPyProgramProposalSignature._strip_lang_tag(lm_out.strip())
            new_instruction = lm_out

        return {"new_program": new_instruction}


# ------------------------------------------------------------------
# Adapter
# ------------------------------------------------------------------

class NERSyntheticAdapter:
    """GEPAAdapter that evolves a DSPy program for NER data augmentation.

    Conforms to the ``GEPAAdapter`` protocol.
    """

    def __init__(
        self,
        task_lm: dspy.LM,
        reflection_lm: dspy.LM,
        train_df: pd.DataFrame,
        dev_df: pd.DataFrame,
        synth_dataset_size: int = 200,
        bert_epochs: int = 30,
        bert_batch_size: int = 16,
        baseline_epochs: int = 30,
        baseline_dir: str = "gepa_runs/baseline_checkpoint",
        failure_score: float = 0.0,
        rng: random.Random | None = None,
    ):
        self.task_lm = task_lm
        self.reflection_lm = reflection_lm
        self.train_df = train_df
        self.dev_df = dev_df
        self.synth_dataset_size = synth_dataset_size
        self.bert_epochs = bert_epochs
        self.bert_batch_size = bert_batch_size
        self.failure_score = failure_score
        self.rng = rng or random.Random(0)
        self._eval_count = 0

        print("[adapter] Training baseline adapter on real training data ...")
        self._baseline_relaxed_f1, self._baseline_checkpoint_path = train_baseline(
            train_df,
            dev_df,
            baseline_dir,
            num_epochs=baseline_epochs,
            batch_size=bert_batch_size,
        )
        print(
            f"[adapter] Baseline relaxed F1: {self._baseline_relaxed_f1:.4f}  "
            f"checkpoint: {self._baseline_checkpoint_path}"
        )

    # ---------------------------------------------------------------
    # build_program  (shared helper, same pattern as DspyAdapter)
    # ---------------------------------------------------------------

    def build_program(
        self, candidate: dict[str, str],
    ) -> tuple[dspy.Module, None] | tuple[None, str]:
        src = candidate["program"]
        try:
            compile(src, "<candidate>", "exec")
        except SyntaxError as exc:
            return None, f"SyntaxError: {exc}\n{traceback.format_exc()}"

        ctx: dict[str, Any] = {}
        try:
            exec(src, ctx)
        except Exception as exc:
            return None, f"Exec error: {exc}\n{traceback.format_exc()}"

        prog = ctx.get("program")
        if prog is None:
            return None, (
                "Code did not define a `program` variable. "
                "It must be an instance of dspy.Module."
            )
        if not isinstance(prog, dspy.Module):
            return None, (
                f"`program` is {type(prog)}, expected dspy.Module."
            )
        prog.set_lm(self.task_lm)
        return prog, None

    # ---------------------------------------------------------------
    # _generate_augmented_data
    # ---------------------------------------------------------------

    def _generate_augmented_data(
        self,
        program: dspy.Module,
    ) -> tuple[pd.DataFrame, list[dict]]:
        """Generate augmented NER data by applying the program to training examples."""

        import ast

        rows: list[dict] = []
        traces: list[dict] = []
        failed_count = 0

        # Sample training examples (with replacement if train is smaller than target)
        n_train = len(self.train_df)
        n_target = min(self.synth_dataset_size, n_train) if n_train > 0 else 0
        if n_target == 0:
            print(f"[eval #{self._eval_count}] No training examples available")
            return pd.DataFrame(columns=["tokens", "labels", "ner_tags", "ID"]), []

        indices = self.rng.choices(
            range(n_train),
            k=self.synth_dataset_size,
        ) if n_train < self.synth_dataset_size else self.rng.sample(
            range(n_train),
            min(n_train, self.synth_dataset_size),
        )

        print(f"[eval #{self._eval_count}] Augmenting {len(indices)} training examples "
              f"(target: {self.synth_dataset_size}) ...")

        batch_inputs: list[dspy.Example] = []
        for idx in indices:
            row = self.train_df.iloc[idx]
            tokens = ast.literal_eval(row["tokens"]) if isinstance(row["tokens"], str) else row["tokens"]
            ner_tags = ast.literal_eval(row["ner_tags"]) if isinstance(row["ner_tags"], str) else row["ner_tags"]
            annotated_input = "\n".join(f"{t}\t{tag}" for t, tag in zip(tokens, ner_tags))
            ex = dspy.Example(
                annotated_input=annotated_input,
                source_id=row["ID"],
                original_tokens=tokens,
                original_ner_tags=ner_tags,
            ).with_inputs("annotated_input")
            batch_inputs.append(ex)

        try:
            preds = program.batch(batch_inputs)
        except Exception as exc:
            traces.append({"error": str(exc)})
            print(f"[eval #{self._eval_count}]   batch error: {exc}")
            return pd.DataFrame(columns=["tokens", "labels", "ner_tags", "ID"]), traces

        for i, pred in enumerate(preds):
            ex = batch_inputs[i]
            source_id = getattr(ex, "source_id", f"idx_{i}")
            orig_tokens = getattr(ex, "original_tokens", [])
            orig_ner_tags = getattr(ex, "original_ner_tags", [])

            trace_entry: dict[str, Any] = {
                "source_id": source_id,
                "original_tokens": orig_tokens,
                "original_ner_tags": orig_ner_tags,
            }

            try:
                raw_output = getattr(pred, "augmented_output", "") or getattr(pred, "annotated_output", "")

                trace_entry["raw_augmented_output"] = raw_output

                parsed = _parse_annotated_output(raw_output)

                if parsed is None:
                    trace_entry["error"] = "Could not parse augmented_output"
                    traces.append(trace_entry)
                    failed_count += 1
                    continue

                tokens, ner_tags = parsed

                labels = [
                    tag.split("-", 1)[1] if tag != "O" else "_"
                    for tag in ner_tags
                ]

                rows.append({
                    "tokens": str(tokens),
                    "labels": str(labels),
                    "ner_tags": str(ner_tags),
                    "ID": f"aug_{source_id}_{len(rows)}",
                })

                trace_entry["tokens"] = tokens
                trace_entry["ner_tags"] = ner_tags
                trace_entry["success"] = True

            except Exception as exc:
                trace_entry["error"] = str(exc)
                failed_count += 1

            traces.append(trace_entry)

        synth_df = pd.DataFrame(
            rows,
            columns=["tokens", "labels", "ner_tags", "ID"],
        )

        success_traces = [t for t in traces if t.get("success")]
        total_attempted = len(rows) + failed_count
        rate = len(rows) / total_attempted * 100 if total_attempted else 0
        print(f"[eval #{self._eval_count}] Augmentation done: "
              f"{len(rows)}/{total_attempted} valid ({rate:.1f}% success)")

        self._log_augmented_samples(success_traces)
        self._log_entity_distribution(success_traces)

        return synth_df, traces
    # ---------------------------------------------------------------
    # Console logging helpers
    # ---------------------------------------------------------------

    @staticmethod
    def _format_augmented_pair(
        orig_tokens: list[str],
        orig_ner_tags: list[str],
        aug_tokens: list[str],
        aug_ner_tags: list[str],
    ) -> str:
        """Format an original vs augmented pair as a readable string."""
        orig_text = " ".join(orig_tokens)
        aug_text = " ".join(aug_tokens)
        aug_entities = [
            f"{tag}({aug_tokens[i]})"
            for i, tag in enumerate(aug_ner_tags)
            if tag != "O"
        ]
        ent_str = " ".join(aug_entities) if aug_entities else "(no entities)"
        return (
            f'original: "{orig_text}"\n'
            f'             augmented: "{aug_text}"\n'
            f'             entities: {ent_str}'
        )

    def _log_augmented_samples(self, success_traces: list[dict], n: int = 3) -> None:
        """Print *n* sample original vs augmented pairs to the console."""
        if not success_traces:
            return
        samples = success_traces[:n]
        for i, s in enumerate(samples, 1):
            formatted = self._format_augmented_pair(
                s.get("original_tokens", []),
                s.get("original_ner_tags", []),
                s.get("tokens", []),
                s.get("ner_tags", []),
            )
            print(f"  [sample {i}] {formatted}")

    def _log_entity_distribution(self, success_traces: list[dict]) -> None:
        """Print entity span counts and all-O example count."""
        clinical = 0
        social = 0
        all_o = 0
        for s in success_traces:
            tags = s.get("ner_tags", [])
            has_entity = False
            for tag in tags:
                if tag == "B-ClinicalImpacts":
                    clinical += 1
                    has_entity = True
                elif tag == "B-SocialImpacts":
                    social += 1
                    has_entity = True
                elif tag.startswith(("I-ClinicalImpacts", "I-SocialImpacts")):
                    has_entity = True
            if not has_entity:
                all_o += 1
        print(f"  entities: {clinical} ClinicalImpacts spans, "
              f"{social} SocialImpacts spans, {all_o} all-O examples")

    def _log_evaluation_summary(
        self,
        num_valid: int,
        total_calls: int,
        result: EvalResult,
        delta: float,
    ) -> None:
        """Print a structured per-evaluation summary block."""
        rate = num_valid / total_calls * 100 if total_calls else 0
        sign = "+" if delta >= 0 else ""

        lines = [
            f"=== GEPA Eval #{self._eval_count} ===",
            f"  augmented examples: {num_valid}/{self.synth_dataset_size} "
            f"({rate:.1f}% success)",
            f"  baseline F1:      {self._baseline_relaxed_f1:.4f}",
            f"  candidate F1:     {result.relaxed_f1:.4f}",
            f"  delta:           {sign}{delta:.4f}",
            f"  strict F1:        {result.f1:.4f}",
        ]

        per_entity = result.per_entity_results
        if per_entity:
            lines.append("  per-entity:")
            for etype, metrics in per_entity.items():
                if etype == "Overall":
                    continue
                ef1 = metrics.get("F1-Score", 0.0)
                lines.append(f"    {etype:<20s} relaxed_f1={ef1:.4f}")

        lines.append("=" * len(lines[0]))
        print("\n".join(lines))

    # ---------------------------------------------------------------
    # evaluate  (GEPAAdapter protocol)
    # ---------------------------------------------------------------

    def evaluate(
        self,
        batch: list,
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        program, err = self.build_program(candidate)

        if program is None:
            print(f"[eval #{self._eval_count}] BUILD FAILED: "
                  f"{err.splitlines()[0] if err else 'unknown'}")
            n = len(batch)
            self._eval_count += 1
            return EvaluationBatch(
                outputs=[None] * n,
                scores=[self.failure_score] * n,
                trajectories=err if capture_traces else None,
            )

        # 1. Generate augmented data
        synth_df, gen_traces = self._generate_augmented_data(program)
        num_valid = len(synth_df)

        if num_valid == 0:
            print(f"[eval #{self._eval_count}] AUGMENTATION FAILED: 0 valid examples")
            n = len(batch)
            traj = {
                "generation_traces": gen_traces,
                "bert_eval": None,
                "error": "No valid augmented examples generated",
            }
            self._eval_count += 1
            return EvaluationBatch(
                outputs=[None] * n,
                scores=[self.failure_score] * n,
                trajectories=traj if capture_traces else None,
            )

        # 2. Fine-tune baseline checkpoint on synthetic data only
        try:
            result = finetune_from_checkpoint(
                self._baseline_checkpoint_path,
                synth_df,
                self.dev_df,
                num_epochs=self.bert_epochs,
                batch_size=self.bert_batch_size,
            )
        except Exception as exc:
            print(f"[eval #{self._eval_count}] FINETUNE FAILED: {exc}",
                  file=sys.stderr)
            n = len(batch)
            traj = {
                "generation_traces": gen_traces,
                "bert_eval": None,
                "error": f"BERT failed: {exc}",
            }
            self._eval_count += 1
            return EvaluationBatch(
                outputs=[None] * n,
                scores=[self.failure_score] * n,
                trajectories=traj if capture_traces else None,
            )

        # 3. Score = relaxed-F1 improvement over baseline (uniform for all batch examples)
        delta = result.relaxed_f1 - self._baseline_relaxed_f1
        n = len(batch)
        scores = [delta] * n
        outputs = [{"delta": delta, "relaxed_f1": result.relaxed_f1}] * n

        trajectories = None
        if capture_traces:
            trajectories = {
                "generation_traces": gen_traces,
                "bert_eval": {
                    "f1_strict": result.f1,
                    "precision_strict": result.precision,
                    "recall_strict": result.recall,
                    "f1_relaxed": result.relaxed_f1,
                    "precision_relaxed": result.relaxed_precision,
                    "recall_relaxed": result.relaxed_recall,
                    "per_entity_results": result.per_entity_results,
                    "per_example": result.per_example,
                },
                "baseline_relaxed_f1": self._baseline_relaxed_f1,
                "delta": delta,
                "num_augmented": num_valid,
            }

        self._log_evaluation_summary(num_valid, len(gen_traces), result, delta)
        self._eval_count += 1

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    # ---------------------------------------------------------------
    # make_reflective_dataset  (GEPAAdapter protocol)
    # ---------------------------------------------------------------

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        assert "program" in components_to_update

        # Handle build-failure case where trajectories is a plain error string
        if isinstance(eval_batch.trajectories, str):
            return {
                "program": [
                    {
                        "Feedback": eval_batch.trajectories
                        + "\n\n--- Annotation Guidelines (always follow) ---\n"
                        + ANNOTATION_GUIDELINES,
                    }
                ],
            }

        traj = eval_batch.trajectories or {}
        items: list[dict[str, Any]] = []

        # --- A) Augmentation summary ---
        gen_traces = traj.get("generation_traces", [])
        successes = [g for g in gen_traces if g.get("success")]
        failures = [g for g in gen_traces if g.get("error")]

        gen_summary: dict[str, Any] = {
            "Section": "Augmentation Summary",
            "total_augmentations": len(gen_traces),
            "successful_augmentations": len(successes),
            "failed_augmentations": len(failures),
        }
        if failures:
            gen_summary["failure_samples"] = [
                {
                    "source_id": f.get("source_id"),
                    "error": f.get("error", ""),
                    "original_tokens": f.get("original_tokens", []),
                }
                for f in failures[:5]
            ]
        if successes:
            samples = successes[:3]
            gen_summary["success_samples"] = [
                {
                    "source_id": s.get("source_id"),
                    "original_tokens": s.get("original_tokens", []),
                    "original_ner_tags": s.get("original_ner_tags", []),
                    "augmented_tokens": s.get("tokens", []),
                    "augmented_ner_tags": s.get("ner_tags", []),
                }
                for s in samples
            ]
        items.append(gen_summary)

        # --- B) BERT evaluation (fine-tuned on synth) vs baseline ---
        bert_eval = traj.get("bert_eval")
        if bert_eval is not None:
            bert_summary: dict[str, Any] = {
                "Section": "BERT Evaluation on Dev Set (fine-tuned on augmented data)",
                "baseline_relaxed_f1": traj.get("baseline_relaxed_f1", self._baseline_relaxed_f1),
                "candidate_relaxed_f1": bert_eval["f1_relaxed"],
                "delta_relaxed_f1": traj.get("delta", 0.0),
                "strict_f1": bert_eval["f1_strict"],
                "strict_precision": bert_eval["precision_strict"],
                "strict_recall": bert_eval["recall_strict"],
                "relaxed_precision": bert_eval["precision_relaxed"],
                "relaxed_recall": bert_eval["recall_relaxed"],
                "per_entity_results": bert_eval.get("per_entity_results", {}),
            }
            items.append(bert_summary)

            wrong = [
                pe for pe in bert_eval["per_example"] if not pe["correct"]
            ]
            if wrong:
                sample_wrong = wrong[:8]
                failure_details: dict[str, Any] = {
                    "Section": "BERT Failure Cases (dev examples the model got wrong)",
                    "num_failures": len(wrong),
                    "samples": [],
                }
                for w in sample_wrong:
                    failure_details["samples"].append({
                        "tokens": w["tokens"],
                        "gold_tags": w["gold_tags"],
                        "pred_tags": w["pred_tags"],
                    })
                items.append(failure_details)

        # --- C) Annotation guidelines (constant) ---
        items.append({
            "Section": "Annotation Guidelines (always follow when augmenting; output must have correct BIO tags for the augmented text)",
            "guidelines": ANNOTATION_GUIDELINES,
        })

        return {"program": items}

    # ---------------------------------------------------------------
    # propose_new_texts  (GEPAAdapter protocol — optional override)
    # ---------------------------------------------------------------

    def _wrap_lm(self, lm: dspy.LM, *, log: bool = False):
        """Wrap a dspy.LM into the ``(str) -> str`` callable GEPA expects."""
        def _call(prompt: str | list[dict[str, Any]]) -> str:
            if isinstance(prompt, str):
                result = lm(prompt)
            else:
                result = lm(messages=prompt)
            if isinstance(result, list):
                out = result[0] if result else ""
            else:
                out = str(result)
            if log:
                print(f"[reflection] Critic response:\n{out}")
            return out
        return _call

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        lm_callable = self._wrap_lm(self.reflection_lm, log=True)
        new_texts: dict[str, str] = {}
        for name in components_to_update:
            print(f'[reflection] Proposing new program for component "{name}" ...')
            base = candidate[name]
            feedback = reflective_dataset[name]
            new_texts[name] = DSPyProgramProposalSignature.run(
                lm=lm_callable,
                input_dict={
                    "curr_program": base,
                    "dataset_with_feedback": feedback,
                },
            )["new_program"]
        return new_texts
