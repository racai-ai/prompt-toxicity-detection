"""Entry point for running the GEPA NER evolution loop.

The evolved DSPy program augments training examples (input: tokens + ner_tags
-> output: different example with its own correct BIO tags). BERT is
fine-tuned on augmentations; delta relaxed F1 is the fitness score.

Usage (programmatic)::

    from gepa_ner.run import run_evolution
    result = run_evolution("data/new_train_data.csv", "data/new_dev_data.csv")

Usage (CLI — wired through ``hner evolve``)::

    hner evolve data/new_train_data.csv data/new_dev_data.csv \\
        --ollama-model gemma3:27b \\
        --max-metric-calls 100 \\
        --bert-epochs 30 \\
        --run-dir gepa_runs/ner_evolution
"""

from __future__ import annotations

import ast
import os

import dspy
import pandas as pd

import gepa
from gepa_ner.adapter import NERSyntheticAdapter
from gepa_ner.seed_program import SEED_PROGRAM
from utils.device import set_device

os.environ["HOSTED_VLLM_API_BASE"] = "http://localhost:8000/v1"



def _load_dev_examples(dev_csv: str) -> list[dspy.Example]:
    """Load the dev CSV into a list of ``dspy.Example`` objects."""
    df = pd.read_csv(dev_csv)
    examples: list[dspy.Example] = []
    for _, row in df.iterrows():
        tokens = ast.literal_eval(row["tokens"]) if isinstance(row["tokens"], str) else row["tokens"]
        ner_tags = ast.literal_eval(row["ner_tags"]) if isinstance(row["ner_tags"], str) else row["ner_tags"]
        ex = dspy.Example(
            tokens=tokens,
            ner_tags=ner_tags,
            ID=row["ID"],
        ).with_inputs("tokens", "ner_tags", "ID")
        examples.append(ex)
    return examples


def run_evolution(
    train_csv: str,
    dev_csv: str,
    *,
    ollama_model: str = "gemma3:27b",
    ollama_base_url: str = "http://localhost:11434",
    max_metric_calls: int = 100,
    bert_epochs: int = 3,
    bert_batch_size: int = 16,
    baseline_epochs: int = 8,
    synth_dataset_size: int = 200,
    run_dir: str = "gepa_runs/ner_evolution",
    device: str | None = None,
) -> "gepa.GEPAResult":
    """Run the full GEPA evolution loop and return the result.

    Parameters
    ----------
    train_csv : str
        Path to the real training CSV (tokens, labels, ner_tags, ID).
    dev_csv : str
        Path to the dev/validation CSV — used as GEPA train & val set.
    ollama_model : str
        Ollama model name for both the task LM and the reflection LM.
    ollama_base_url : str
        Base URL for the Ollama server.
    max_metric_calls : int
        GEPA budget (total number of evaluate() calls).
    bert_epochs : int
        Number of training epochs for the lightweight BERT evaluator.
    bert_batch_size : int
        Batch size for BERT training.
    baseline_epochs : int
        Number of epochs to train the baseline adapter on real training
        data (done once before the evolution loop starts).
    synth_dataset_size : int
        Target number of augmented examples per evaluation (sampled from train).
    run_dir : str
        Directory for GEPA checkpoints and logs.  If it already exists
        GEPA will resume from the last saved state.
    device : str | None
        Runtime device override (for example: ``cpu``, ``cuda``, ``rocm``,
        ``mps``, ``xpu``).  ``None`` enables automatic detection.

    Returns
    -------
    gepa.GEPAResult
        Contains ``best_candidate["program"]`` — the evolved DSPy code.
    """
    if device is not None:
        set_device(device)

    # --- LMs ---
    task_lm = dspy.LM(
        f"hosted_vllm/{ollama_model}",
        temperature=0.7,
        cache=False,
    )
    reflection_lm = dspy.LM(
        f"hosted_vllm/{ollama_model}",
        temperature=0.7,
        cache=False,
    )

    # --- Data ---
    train_df = pd.read_csv(train_csv)
    dev_df = pd.read_csv(dev_csv)
    dev_examples = _load_dev_examples(dev_csv)

    # --- Adapter (trains baseline before GEPA loop starts) ---
    baseline_dir = os.path.join(run_dir, "baseline_checkpoint")
    adapter = NERSyntheticAdapter(
        task_lm=task_lm,
        reflection_lm=reflection_lm,
        train_df=train_df,
        dev_df=dev_df,
        synth_dataset_size=synth_dataset_size,
        bert_epochs=bert_epochs,
        bert_batch_size=bert_batch_size,
        baseline_epochs=baseline_epochs,
        baseline_dir=baseline_dir,
    )

    # --- Run GEPA ---
    print(f"[gepa_ner] Starting evolution (augmentation mode) — budget={max_metric_calls}, "
          f"aug_size={synth_dataset_size}, bert_epochs={bert_epochs}")
    print(f"[gepa_ner] Run directory: {run_dir}")

    os.makedirs(run_dir, exist_ok=True)

    result = gepa.optimize(
        seed_candidate={"program": SEED_PROGRAM},
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

    # --- Save best program ---
    best_code = result.best_candidate["program"]
    best_path = os.path.join(run_dir, "best_program.py")
    with open(best_path, "w", encoding="utf-8") as f:
        f.write(best_code)

    best_idx = result.best_idx
    best_score = result.val_aggregate_scores[best_idx]
    print(f"[gepa_ner] Evolution complete.")
    print(f"[gepa_ner] Best validation score: {best_score:.4f}")
    print(f"[gepa_ner] Best program saved to: {best_path}")

    return result


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for ``hner evolve``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evolve a DSPy NER data-augmentation program with GEPA.",
    )
    parser.add_argument("train_csv", help="Path to the training CSV file.")
    parser.add_argument("dev_csv", help="Path to the dev/validation CSV file.")
    parser.add_argument(
        "--ollama-model", default="gemma3:27b",
        help="Ollama model name (default: gemma3:27b).",
    )
    parser.add_argument(
        "--ollama-base-url", default="http://localhost:11434",
        help="Ollama API base URL.",
    )
    parser.add_argument(
        "--max-metric-calls", type=int, default=100,
        help="GEPA budget: total evaluate() calls (default: 100).",
    )
    parser.add_argument(
        "--bert-epochs", type=int, default=30,
        help="BERT adapter training epochs per evaluation (default: 30).",
    )
    parser.add_argument(
        "--bert-batch-size", type=int, default=16,
        help="BERT batch size (default: 16).",
    )
    parser.add_argument(
        "--baseline-epochs", type=int, default=30,
        help="Epochs for baseline adapter training on real data (default: 30).",
    )
    parser.add_argument(
        "--synth-dataset-size", type=int, default=200,
        help="Target number of augmented examples per evaluation (default: 200).",
    )
    parser.add_argument(
        "--run-dir", default="gepa_runs/ner_evolution",
        help="Directory for GEPA checkpoints/logs.",
    )
    parser.add_argument(
        "--device", default=None,
        help=(
            "Runtime device to use (for example: cpu, cuda, rocm, mps, xpu). "
            "Defaults to automatic detection."
        ),
    )

    args = parser.parse_args(argv)
    run_evolution(
        args.train_csv,
        args.dev_csv,
        ollama_model=args.ollama_model,
        ollama_base_url=args.ollama_base_url,
        max_metric_calls=args.max_metric_calls,
        bert_epochs=args.bert_epochs,
        bert_batch_size=args.bert_batch_size,
        baseline_epochs=args.baseline_epochs,
        synth_dataset_size=args.synth_dataset_size,
        run_dir=args.run_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
