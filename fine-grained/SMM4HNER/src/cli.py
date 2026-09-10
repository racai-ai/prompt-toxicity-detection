"""CLI entry point for the SMM4H NER system.

Usage
-----
Train::

    hner train <input_file> <validation_file> <output_model> [--config-file <cfg.yaml>]
        [--batch-size 16] [--max-training-epochs 50]

    The config YAML (required) supplies model_type and any architecture / loss /
    optimiser hyperparameters, for example::

        # conf/bert-sequence-classification.yaml
        model_type: bert_finetuned
        metric: macro_f1
        max_sequence_len: 512
        learning_rate: 2.0e-5
        warmup_ratio: 0.1
        weight_decay: 0.01

Test (inference)::

    hner test <input_file> <output_file> --type <model_type> [--model <model_file>]
        [--batch-size 16] [--max-sequence-len 512]

    # For the llm_ner model type in zero-shot mode:
    hner test <input_file> <output_file> --type llm_ner
        [--ollama-model gemma3:27b] [--ollama-base-url http://localhost:11434]

Augment::

    hner augment <input_file> <output_file> [--config-file conf/augment.yaml]

    Config keys: strategy, model.

Evolve (GEPA + DSPy data augmentation)::

    hner evolve <train_csv> <dev_csv> [--config-file conf/evolve.yaml]

    Config keys: ollama_model, ollama_base_url, max_metric_calls, bert_epochs,
    bert_batch_size, baseline_epochs, run_dir.

Evaluate (official SMM4H strict + relaxed metrics)::

    hner evaluate <input_file> --gold-col <col> --pred-col <col>

Analyze (dataset entity analysis)::

    hner analyze <trainset_file> <devset_file> <report_file> [--fuzzy-threshold 0.5]

Supported model types: bert_adapter, bert_finetuned, gliner, bert_variational, bert_fixmatch, lstm, llm_ner
"""

import argparse
import sys

import yaml

from data.augmenter import STRATEGY_REGISTRY, DEFAULT_MODEL, augment
from data.analyzer import analyze
from data.comparator import compare
from training.trainer import MODEL_REGISTRY, Trainer
from utils.device import set_device


def _load_config(path: str) -> dict:
    """Load a YAML config file and return it as a dict (empty dict if path is None)."""
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hner",
        description="SMM4H NER — train and test NER models for the shared task.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # ------------------------------------------------------------------ train
    train_parser = subparsers.add_parser(
        "train",
        help="Train a model and save it to disk.",
    )
    train_parser.add_argument("input_file", help="Path to the training CSV file.")
    train_parser.add_argument("validation_file", help="Path to the validation CSV file.")
    train_parser.add_argument("output_model", help="Directory where the trained model will be saved.")
    train_parser.add_argument(
        "--config-file",
        dest="config_file",
        default=None,
        help=(
            "Path to a YAML config file that supplies model_type and "
            "hyperparameters (metric, learning_rate, warmup_ratio, weight_decay, "
            "alpha_ce, alpha_dice, max_sequence_len, unlabeled_file, word2vec_file, "
            "ollama_model, ollama_base_url, max_metric_calls, eval_sample_size, …)."
        ),
    )
    train_parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=None,
        help=(
            "Mini-batch size for training (default: model-specific, typically 16 or 32). "
            "Overrides the value in the config file."
        ),
    )
    train_parser.add_argument(
        "--max-training-epochs",
        dest="max_training_epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: 50). Overrides the value in the config file.",
    )
    train_parser.add_argument(
        "--device",
        dest="device",
        default=None,
        help=(
            "Runtime device to use (for example: cpu, cuda, rocm, mps, xpu). "
            "Defaults to automatic detection."
        ),
    )

    # ------------------------------------------------------------------ test
    test_parser = subparsers.add_parser(
        "test",
        help="Run inference with a trained model and write predictions.",
    )
    test_parser.add_argument("input_file", help="Path to the test CSV file.")
    test_parser.add_argument("output_file", help="Path where the output CSV will be written.")
    test_parser.add_argument(
        "--type",
        dest="model_type",
        required=True,
        choices=sorted(MODEL_REGISTRY.keys()),
        help="Model architecture to use.",
    )
    test_parser.add_argument(
        "--model",
        dest="model_file",
        default=None,
        help="Path to a trained model directory. If omitted, the default zero-shot model is used.",
    )
    test_parser.add_argument(
        "--ollama-model",
        dest="ollama_model",
        default=None,
        help=(
            "Ollama model identifier for zero-shot llm_ner inference "
            "(e.g. 'gemma3:27b').  Ignored for other model types."
        ),
    )
    test_parser.add_argument(
        "--ollama-base-url",
        dest="ollama_base_url",
        default=None,
        help=(
            "Ollama server base URL for zero-shot llm_ner inference "
            "(default: http://localhost:11434).  Ignored for other model types."
        ),
    )
    test_parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=None,
        help=(
            "Mini-batch size for inference (default: model-specific). "
            "Overrides the model's built-in default batch size."
        ),
    )
    test_parser.add_argument(
        "--max-sequence-len",
        dest="max_sequence_len",
        type=int,
        default=None,
        help=(
            "Maximum input sequence length in tokens for inference "
            "(default: value stored in model.yaml, or 512 for BERT-based models, "
            "4096 for LSTM).  Sequences longer than this limit will be truncated."
        ),
    )
    test_parser.add_argument(
        "--device",
        dest="device",
        default=None,
        help=(
            "Runtime device to use (for example: cpu, cuda, rocm, mps, xpu). "
            "Defaults to automatic detection."
        ),
    )

    # --------------------------------------------------------------- augment
    augment_parser = subparsers.add_parser(
        "augment",
        help="Augment a dataset using an LLM via Ollama and write results to a new CSV.",
    )
    augment_parser.add_argument("input_file", help="Path to the source CSV file.")
    augment_parser.add_argument("output_file", help="Path where the augmented CSV will be written.")
    augment_parser.add_argument(
        "--config-file",
        dest="config_file",
        default=None,
        help=(
            "Path to a YAML config file supplying augmentation parameters. "
            "Recognised keys: strategy (default: negate), model (Ollama model name)."
        ),
    )

    # -------------------------------------------------------------- compare
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare two annotated CSV files and write diverging examples to a text file.",
    )
    compare_parser.add_argument("file1", help="Path to the first (gold-standard) CSV file.")
    compare_parser.add_argument("file2", help="Path to the second (predicted) CSV file.")
    compare_parser.add_argument("output_file", help="Path to the output text file.")

    # -------------------------------------------------------------- evolve
    evolve_parser = subparsers.add_parser(
        "evolve",
        help="Evolve a DSPy NER data-generation program with GEPA.",
    )
    evolve_parser.add_argument("train_csv", help="Path to the training CSV file.")
    evolve_parser.add_argument("dev_csv", help="Path to the dev/validation CSV file.")
    evolve_parser.add_argument(
        "--config-file",
        dest="config_file",
        default=None,
        help=(
            "Path to a YAML config file supplying GEPA parameters. "
            "Recognised keys: ollama_model, ollama_base_url, max_metric_calls, "
            "bert_epochs, bert_batch_size, baseline_epochs, run_dir."
        ),
    )
    evolve_parser.add_argument(
        "--device",
        dest="device",
        default=None,
        help=(
            "Runtime device to use (for example: cpu, cuda, rocm, mps, xpu). "
            "Defaults to automatic detection."
        ),
    )

    # -------------------------------------------------------------- analyze
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze train and dev datasets and write an entity analysis report.",
    )
    analyze_parser.add_argument("trainset_file", help="Path to the training CSV file.")
    analyze_parser.add_argument("devset_file", help="Path to the dev/validation CSV file.")
    analyze_parser.add_argument("report_file", help="Path to the output text report file.")
    analyze_parser.add_argument(
        "--fuzzy-threshold",
        dest="fuzzy_threshold",
        type=float,
        default=0.5,
        help=(
            "Minimum Jaccard similarity (bag-of-words) to consider two entity "
            "strings a fuzzy match (default: 0.5)."
        ),
    )

    # ------------------------------------------------------------ evaluate
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run official SMM4H strict + relaxed NER evaluation on a CSV/Excel file.",
    )
    evaluate_parser.add_argument("input_file", help="Path to a CSV or Excel file with gold and predicted BIO tags.")
    evaluate_parser.add_argument(
        "--gold-col", default="ner_tags_str",
        help="Column name for gold BIO tags (default: ner_tags_str).",
    )
    evaluate_parser.add_argument(
        "--pred-col", default="prediction",
        help="Column name for predicted BIO tags (default: prediction).",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "augment":
        cfg = _load_config(getattr(args, "config_file", None))
        strategy = cfg.get("strategy", "negate")
        model = cfg.get("model", DEFAULT_MODEL)
        augment(args.input_file, args.output_file, strategy, model)
        return

    if args.command == "analyze":
        from data.analyzer import FUZZY_THRESHOLD, build_report
        threshold = getattr(args, "fuzzy_threshold", FUZZY_THRESHOLD)
        report = build_report(args.trainset_file, args.devset_file, threshold=threshold)
        with open(args.report_file, "w", encoding="utf-8") as fh:
            fh.write(report)
            fh.write("\n")
        print(f"[analyze] Report written to {args.report_file}")
        return

    if args.command == "compare":
        compare(args.file1, args.file2, args.output_file)
        return

    if args.command == "evaluate":
        from data.evaluation import (
            _to_list,
            evaluate_test_strict_ner,
            calculate_f1_per_entity_covering_all,
        )
        import pandas as pd

        path = args.input_file
        if path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)

        strict = evaluate_test_strict_ner(
            df, gold_col=args.gold_col, pred_col=args.pred_col, print_report=True,
        )

        gold_tags = df[args.gold_col].apply(_to_list).tolist()
        pred_tags = df[args.pred_col].apply(_to_list).tolist()
        relaxed = calculate_f1_per_entity_covering_all(gold_tags, pred_tags)

        print("\n=== RELAXED OVERLAP NER METRICS ===")
        for entity_type, metrics in relaxed.items():
            print(f"Entity Type: {entity_type}")
            for k, v in metrics.items():
                print(f"  {k}: {v}")
            print()

        print("=" * 50)
        print(f"Micro F1 (Strict):  {strict['f1_strict']:.4f}")
        print(f"Overall F1 (Relaxed): {relaxed['Overall']['F1-Score']:.4f}")
        print("=" * 50)
        return

    if args.command == "evolve":
        from gepa_ner.run import run_evolution

        cfg = _load_config(getattr(args, "config_file", None))
        if getattr(args, "device", None) is not None:
            set_device(args.device)

        run_evolution(
            args.train_csv,
            args.dev_csv,
            ollama_model=cfg.get("ollama_model", "gemma3:27b"),
            ollama_base_url=cfg.get("ollama_base_url", "http://localhost:11434"),
            max_metric_calls=cfg.get("max_metric_calls", 100),
            bert_epochs=cfg.get("bert_epochs", 5),
            bert_batch_size=cfg.get("bert_batch_size", 16),
            baseline_epochs=cfg.get("baseline_epochs", 30),
            run_dir=cfg.get("run_dir", "gepa_runs/ner_evolution"),
            device=getattr(args, "device", None),
        )
        return

    if args.command == "train":
        cfg = _load_config(getattr(args, "config_file", None))

        model_type = cfg.get("model_type")
        if model_type is None:
            print(
                "error: model_type is required. Provide it in the config file "
                "(--config-file) under the key 'model_type'.",
                file=sys.stderr,
            )
            sys.exit(1)
        if model_type not in MODEL_REGISTRY:
            print(
                f"error: unsupported model_type '{model_type}'. "
                f"Choose one of: {sorted(MODEL_REGISTRY.keys())}",
                file=sys.stderr,
            )
            sys.exit(1)

        if getattr(args, "device", None) is not None:
            set_device(args.device)

        trainer = Trainer(model_type=model_type)

        # CLI --batch-size and --max-training-epochs override config values.
        batch_size = args.batch_size if args.batch_size is not None else cfg.get("batch_size")
        max_epochs = (
            args.max_training_epochs
            if args.max_training_epochs is not None
            else cfg.get("max_training_epochs", 50)
        )

        # Build extra kwargs from config (all remaining keys not yet consumed).
        _consumed = {"model_type", "batch_size", "max_training_epochs"}
        extra_kwargs: dict = {k: v for k, v in cfg.items() if k not in _consumed}

        trainer.train(
            args.input_file,
            args.validation_file,
            args.output_model,
            metric=cfg.get("metric", "f1"),
            unlabeled_file=cfg.get("unlabeled_file"),
            word2vec_file=cfg.get("word2vec_file"),
            alpha_ce=cfg.get("alpha_ce", 0.5),
            alpha_dice=cfg.get("alpha_dice", 0.5),
            max_training_epochs=max_epochs,
            batch_size=batch_size,
            max_sequence_len=cfg.get("max_sequence_len"),
            **{
                k: v for k, v in extra_kwargs.items()
                if k not in {
                    "metric", "unlabeled_file", "word2vec_file",
                    "alpha_ce", "alpha_dice", "max_sequence_len",
                }
            },
        )
        return

    if args.command == "test":
        trainer = Trainer(model_type=args.model_type)
        if getattr(args, "device", None) is not None:
            set_device(args.device)
        test_kwargs: dict = {}
        if getattr(args, "ollama_model", None) is not None:
            test_kwargs["ollama_model"] = args.ollama_model
        if getattr(args, "ollama_base_url", None) is not None:
            test_kwargs["ollama_base_url"] = args.ollama_base_url
        trainer.test(
            args.input_file,
            args.output_file,
            getattr(args, "model_file", None),
            batch_size=getattr(args, "batch_size", None),
            max_sequence_len=getattr(args, "max_sequence_len", None),
            **test_kwargs,
        )


if __name__ == "__main__":
    main()
