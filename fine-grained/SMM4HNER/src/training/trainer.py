"""Training and inference orchestrator for SMM4H NER models."""

import os

import pandas as pd
import yaml

from models.bert_adapter import BertAdapterModel
from models.bert_finetuned import BertFinetunedModel
from models.gliner_model import GlinerModel
from models.bert_variational import BertVariational
from models.bert_fixmatch import BertFixMatchModel
from models.bilstm import BiLSTMModel
from models.llm_ner_model import LLMNERModel
from models.pretrained import resolve_model_name_or_path

MODEL_REGISTRY = {
    "bert_adapter": BertAdapterModel,
    "bert_finetuned": BertFinetunedModel,
    "gliner": GlinerModel,
    "bert_variational": BertVariational,
    "bert_fixmatch": BertFixMatchModel,
    "lstm": BiLSTMModel,
    "llm_ner": LLMNERModel,
}

TASK_NER = "ner"
TASK_SEQ_CLS = "sequence_classification"

_MODEL_YAML = "model.yaml"


def detect_dataset_type(df: pd.DataFrame) -> str:
    """Detect whether *df* is a token-classification (NER) or sequence-classification dataset.

    Detection is based on column names:

    * If the dataframe contains a ``prompt`` column (with or without a
      ``label`` column) and no ``tokens`` column, it is treated as a
      **sequence classification** dataset.  This covers both training files
      (which include labels) and inference/test files (which may not).
    * If it contains a ``tokens`` column it is treated as an **NER** dataset.

    Parameters
    ----------
    df:
        The loaded dataframe to inspect.

    Returns
    -------
    str
        Either :data:`TASK_NER` or :data:`TASK_SEQ_CLS`.

    Raises
    ------
    ValueError
        If the column layout does not match either expected format.
    """
    columns = set(df.columns)
    if "tokens" in columns:
        return TASK_NER
    if "prompt" in columns and "tokens" not in columns:
        return TASK_SEQ_CLS
    raise ValueError(
        f"Cannot determine dataset type from columns: {sorted(columns)}. "
        "Expected either NER columns (at minimum 'tokens') or sequence "
        "classification columns (at minimum 'prompt')."
    )


def save_model_yaml(output_path: str, task_type: str, max_sequence_len: int | None = None) -> None:
    """Write a ``model.yaml`` file recording the classifier type and hyperparameters.

    Parameters
    ----------
    output_path:
        Directory where ``model.yaml`` will be written.
    task_type:
        One of :data:`TASK_NER` or :data:`TASK_SEQ_CLS`.
    max_sequence_len:
        Maximum input sequence length used during training.  When provided
        this value is stored in the file so that inference can reproduce the
        same truncation behaviour.
    """
    os.makedirs(output_path, exist_ok=True)
    config = {"task": task_type}
    if max_sequence_len is not None:
        config["max_sequence_len"] = max_sequence_len
    with open(os.path.join(output_path, _MODEL_YAML), "w", encoding="utf-8") as fh:
        yaml.dump(config, fh, default_flow_style=False)


def load_model_yaml(model_path: str) -> dict:
    """Read the ``model.yaml`` written by :func:`save_model_yaml`.

    Returns a dictionary with at least a ``task`` key.  When the file does
    not exist (e.g. for models trained before this feature was introduced)
    ``{"task": TASK_NER}`` is returned so that existing models keep working.

    Parameters
    ----------
    model_path:
        Directory that was previously passed as *output_model* during training.

    Returns
    -------
    dict
        The parsed YAML content, or ``{"task": TASK_NER}`` as a fallback.
    """
    model_path = resolve_model_name_or_path(model_path)
    yaml_path = os.path.join(model_path, _MODEL_YAML)
    if not os.path.exists(yaml_path):
        return {"task": TASK_NER}
    with open(yaml_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_data(file_path: str) -> pd.DataFrame:
    """Load and validate the task CSV file.

    Supports two dataset formats:

    **NER / token classification**
        Expected columns: ``ID``, ``tokens`` (plus ``labels``, ``ner_tags``).
        The ``tokens``, ``labels`` and ``ner_tags`` columns may be stored as
        string-encoded Python lists (e.g. ``"['Hello', ',', 'world']"``).

    **Sequence classification**
        Expected columns: ``prompt``.  Optional columns:
        ``label``, ``dialect``, ``granular_label``.
    """
    df = pd.read_csv(file_path)
    task_type = detect_dataset_type(df)

    if task_type == TASK_NER:
        required_columns = {"ID", "tokens"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Input file is missing required NER columns: {missing}")

    return df


class Trainer:
    """Orchestrates training and inference across all supported model types."""

    def __init__(self, model_type: str):
        if model_type not in MODEL_REGISTRY:
            raise ValueError(
                f"Unsupported model type '{model_type}'. "
                f"Choose one of: {sorted(MODEL_REGISTRY.keys())}"
            )
        self.model_type = model_type
        self.model_class = MODEL_REGISTRY[model_type]

    # ------------------------------------------------------------------

    def train(self, input_file: str, validation_file: str, output_model: str, metric: str = "f1", unlabeled_file: str = None, word2vec_file: str = None, max_training_epochs: int = 50, alpha_ce: float = 0.5, alpha_dice: float = 0.5, batch_size: int | None = None, max_sequence_len: int | None = None, **kwargs) -> None:
        """Load data, train the model, and save it to *output_model*.

        The dataset format is detected automatically from the column names of
        *input_file*:

        * **NER** datasets contain a ``tokens`` column.
        * **Sequence-classification** datasets contain ``prompt`` and ``label``
          columns.

        The detected type is printed to stdout and persisted as a ``model.yaml``
        file inside *output_model*.

        Parameters
        ----------
        input_file:
            Path to the training CSV.
        validation_file:
            Path to the validation CSV.
        output_model:
            Directory where the trained model will be saved.
        metric:
            Metric used for best-model selection.  For NER tasks: ``"f1"``
            (strict seqeval F1, default), ``"relaxed_f1"``, or ``"avg_f1"``.
            For sequence-classification tasks this parameter is ignored and
            ``"f1"`` (binary F1) is always used.
        unlabeled_file:
            Optional path to unlabeled data for models that use it (e.g., fixmatch).
        word2vec_file:
            Optional path to a word2vec text-format file (used by the lstm model).
        max_training_epochs:
            Number of training epochs (default: 50).
        alpha_ce:
            Weight for the CrossEntropy loss component (default: 0.5).
        alpha_dice:
            Weight for the Dice loss component (default: 0.5).
        batch_size:
            Mini-batch size for training.  When ``None`` the model's built-in
            default is used.
        max_sequence_len:
            Maximum input sequence length in tokens.  When ``None`` the
            model's built-in default is used (512 for BERT-based models, 4096
            for LSTM).  The effective value is stored in ``model.yaml``.
        """
        print(f"[hner] Loading training data from: {input_file}")
        train_data = load_data(input_file)

        print(f"[hner] Loading validation data from: {validation_file}")
        val_data = load_data(validation_file)

        task_type = detect_dataset_type(train_data)
        print(f"[hner] Detected classification type: {task_type}")

        # Resolve the effective max_sequence_len so it can be stored in model.yaml.
        # If not specified by the caller, fall back to the model class default.
        effective_max_seq = max_sequence_len
        if effective_max_seq is None:
            effective_max_seq = getattr(self.model_class, "DEFAULT_MAX_LENGTH", None)

        # Persist the task type and max_sequence_len so test/inference can use them later.
        save_model_yaml(output_model, task_type, max_sequence_len=effective_max_seq)

        import inspect
        model_name = kwargs.pop("model_name", None)
        print(f"[hner] Training model type '{self.model_type}' (task: {task_type})...")
        sig = inspect.signature(self.model_class.__init__)
        if "model_name" in sig.parameters and model_name is not None:
            print(f"[hner] Using custom model name: {model_name}")
            model = self.model_class(model_name=model_name)
        else:
            model = self.model_class()

        if task_type == TASK_NER:
            train_kwargs = {
                "metric": metric,
                "max_training_epochs": max_training_epochs,
                "alpha_ce": alpha_ce,
                "alpha_dice": alpha_dice,
            }
            if batch_size is not None:
                train_kwargs["batch_size"] = batch_size
            if effective_max_seq is not None:
                train_kwargs["max_sequence_len"] = effective_max_seq
            if unlabeled_file is not None:
                train_kwargs["unlabeled_file"] = unlabeled_file
            if word2vec_file is not None:
                train_kwargs["word2vec_file"] = word2vec_file
            train_kwargs.update(kwargs)
            model.train(train_data, val_data, output_model, **train_kwargs)
        else:
            seq_kwargs = {
                "max_training_epochs": max_training_epochs,
            }
            if batch_size is not None:
                seq_kwargs["batch_size"] = batch_size
            if effective_max_seq is not None:
                seq_kwargs["max_sequence_len"] = effective_max_seq
            seq_kwargs.update(kwargs)
            model.train_seq_cls(train_data, val_data, output_model, **seq_kwargs)

        print(f"[hner] Model saved to: {output_model}")

    def test(self, input_file: str, output_file: str, model_file: str | None = None, batch_size: int | None = None, max_sequence_len: int | None = None, **kwargs) -> None:
        """Load a model (or use zero-shot default), run inference, and write results.

        The task type (NER vs. sequence classification) is always determined by
        reading ``model.yaml`` from *model_file*.  When *model_file* is ``None``
        (zero-shot inference), the task type defaults to :data:`TASK_NER` because
        zero-shot models only support token-level NER.

        Parameters
        ----------
        input_file:
            Path to the test CSV.
        output_file:
            Path where the predictions CSV will be written.
        model_file:
            Optional path to a trained model directory.  When ``None``, a
            default zero-shot model is used.
        batch_size:
            Mini-batch size for inference.  When ``None`` the model's built-in
            default is used.
        max_sequence_len:
            Maximum input sequence length in tokens for inference.  When
            ``None``, the value stored in ``model.yaml`` is used (if
            available), falling back to the model's built-in default.
        **kwargs:
            Extra keyword arguments forwarded to the model constructor when
            instantiating the default zero-shot model (e.g. ``ollama_model``
            or ``ollama_base_url`` for the ``llm_ner`` model type).
        """
        print(f"[hner] Loading test data from: {input_file}")
        test_data = pd.read_csv(input_file)

        # Task type is always read from model.yaml so that inference behaviour
        # is determined by how the model was trained, not by guessing from the
        # input file columns.  When no model file is provided (zero-shot mode)
        # we default to NER because zero-shot classifiers only support NER.
        if model_file:
            model_file = resolve_model_name_or_path(model_file)
            model_config = load_model_yaml(model_file)
            task_type = model_config.get("task", TASK_NER)
            # Use max_sequence_len from model.yaml when not explicitly overridden.
            if max_sequence_len is None:
                max_sequence_len = model_config.get("max_sequence_len", None)
        else:
            task_type = TASK_NER

        # Validate required columns for NER inference.
        if task_type == TASK_NER:
            missing = {"ID", "tokens"} - set(test_data.columns)
            if missing:
                raise ValueError(f"Input file is missing required NER columns: {missing}")

        print(f"[hner] Classification type: {task_type}")

        if model_file:
            print(f"[hner] Loading model from: {model_file}")
            if task_type == TASK_SEQ_CLS:
                if not hasattr(self.model_class, "load_seq_cls"):
                    raise NotImplementedError(
                        f"{self.model_class.__name__} does not support sequence "
                        "classification.  Use the 'bert_finetuned', 'bert_adapter', or 'lstm' "
                        "model type."
                    )
                model = self.model_class.load_seq_cls(model_file)
            else:
                model = self.model_class.load(model_file)
        else:
            print(f"[hner] No model file specified — using default zero-shot model.")
            model = self.model_class(**kwargs)

        # Apply inference hyperparameters to the loaded model so that predict()
        # can honour them without needing extra arguments.
        if max_sequence_len is not None:
            model.max_length = max_sequence_len
        if batch_size is not None:
            model.batch_size = batch_size

        print(f"[hner] Running inference...")
        if task_type == TASK_NER:
            predictions = model.predict(test_data)
        else:
            predictions = model.predict_seq_cls(test_data)

        predictions.to_csv(output_file, index=False)
        print(f"[hner] Predictions written to: {output_file}")
