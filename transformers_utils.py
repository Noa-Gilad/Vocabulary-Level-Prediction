"""
Vocabulary Level Prediction — Hugging Face Transformers helpers.

Weighted regression training (e.g. RoBERTa with a scalar head), batch collation with
``sample_weight``, and evaluation metrics aligned with ``modeling_utils.evaluate_fold``.
Depends optionally on ``torch``, ``transformers``, ``datasets``, and ``optuna``; functions
that need missing packages raise ``ImportError`` at call time.
"""

# -----------------------------------------------------------------------------
# Environment setting
# -----------------------------------------------------------------------------
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

# Project metrics / sample weights (same package as classical CV; no circular import).
from modeling_utils import compute_balanced_sample_weights, evaluate_fold

# Hugging Face ``datasets`` (tokenized rows for ``Trainer``).
try:
    from datasets import Dataset
except ImportError:  # pragma: no cover
    Dataset = None  # type: ignore[misc, assignment]

# Optuna (SQLite study resume, HP search backend).
try:
    import optuna
except ImportError:  # pragma: no cover
    optuna = None  # type: ignore[misc, assignment]

# PyTorch and Hugging Face ``transformers`` (optional for environments without a GPU stack).
try:
    import torch
    import torch.nn.functional as torch_functional
    from transformers import (
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
        EvalPrediction,
        Trainer,
    )
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[misc, assignment]
    torch_functional = None  # type: ignore[misc, assignment]
    AutoModelForSequenceClassification = None  # type: ignore[misc, assignment]
    DataCollatorWithPadding = None  # type: ignore[misc, assignment]
    EvalPrediction = Any  # type: ignore[misc, assignment]
    Trainer = None  # type: ignore[misc, assignment]


# -----------------------------------------------------------------------------
# Weighted regression loss
# -----------------------------------------------------------------------------


def weighted_mean_squared_error_loss(
    logits: Any,
    labels: Any,
    sample_weight: Any,
) -> Any:
    """
    Scalar weighted MSE: sum_i w_i (logits_i - y_i)^2 / sum_i w_i.

    Uses ``weights`` only from the training batch (no leakage). All weights must be
    non-negative; sum is clamped below to avoid division by zero.

    Parameters
    ----------
    logits : torch.Tensor
        Model predictions (1-D after squeeze, same length as labels).
    labels : torch.Tensor
        Ground-truth targets as float (same device as logits).
    sample_weight : torch.Tensor
        Non-negative per-example weights, same shape as logits.

    Returns
    -------
    torch.Tensor
        Scalar loss (0-D tensor).
    """
    if torch is None:
        raise ImportError("torch is required for weighted_mean_squared_error_loss.")
    logits_f = logits.float()
    labels_f = labels.float()
    weights_f = sample_weight.float()
    squared_errors = (logits_f - labels_f) ** 2
    weight_sum = weights_f.sum().clamp_min(1e-8)
    return (squared_errors * weights_f).sum() / weight_sum


# -----------------------------------------------------------------------------
# Trainer (weighted MSE) and prediction step
# -----------------------------------------------------------------------------


if Trainer is not None:

    class WeightedRegressionTrainer(Trainer):
        """
        Regression ``Trainer`` with optional per-sample weights (balanced rare classes).

        Expects each batch to include ``labels`` (float tensor) and ``sample_weight``
        (float tensor). ``sample_weight`` is removed before ``model(**inputs)`` so the
        model forward signature stays standard. Validation should include
        ``sample_weight`` (e.g. all ones) so batches are collated consistently.

        One-line purpose
            Apply sklearn-style balanced ``sample_weight`` to MSE in PyTorch.

        Parameters
        ----------
        Inherits all ``transformers.Trainer`` arguments.
        """

        def compute_loss(
            self,
            model: Any,
            inputs: Dict[str, Any],
            return_outputs: bool = False,
            **kwargs: Any,
        ) -> Any:
            """
            Weighted MSE when ``sample_weight`` is present; else plain MSE.

            Parameters
            ----------
            model : Any
                Hugging Face model.
            inputs : Dict[str, Any]
                Batch dict; ``sample_weight`` is popped and not forwarded to ``model``.
            return_outputs : bool
                If True, return ``(loss, outputs)``.
            **kwargs : Any
                Absorbs extra keyword args from newer ``Trainer`` versions
                (e.g. ``num_items_in_batch``).

            Returns
            -------
            Any
                Scalar loss, or ``(loss, outputs)`` if ``return_outputs``.
            """
            labels = inputs.pop("labels")
            sample_weight = inputs.pop("sample_weight", None)
            outputs = model(**inputs)
            logits = outputs.logits
            if logits.dim() > 1 and logits.size(-1) == 1:
                logits = logits.squeeze(-1)
            labels = labels.to(logits.device).float()
            if sample_weight is not None:
                sample_weight = sample_weight.to(logits.device)
                loss = weighted_mean_squared_error_loss(logits, labels, sample_weight)
            else:
                loss = torch_functional.mse_loss(logits, labels)
            return (loss, outputs) if return_outputs else loss

        def prediction_step(
            self,
            model: Any,
            inputs: Dict[str, Any],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
            **kwargs: Any,
        ) -> Tuple[Any, Any, Any]:
            """
            Drop ``sample_weight`` before the forward pass (encoder does not accept it).

            Parameters
            ----------
            model : Any
                HF model.
            inputs : Dict[str, Any]
                Batch possibly containing ``sample_weight``.
            prediction_loss_only : bool
                Forwarded to ``Trainer.prediction_step``.
            ignore_keys : list of str | None
                Forwarded to ``Trainer.prediction_step``.
            **kwargs : Any
                Absorbs extra keyword args from newer ``Trainer`` versions.

            Returns
            -------
            tuple
                Same as ``Trainer.prediction_step`` (loss, logits, labels).
            """
            inputs = dict(inputs)
            inputs.pop("sample_weight", None)
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only,
                ignore_keys=ignore_keys,
                **kwargs,
            )

else:

    class WeightedRegressionTrainer:  # type: ignore[misc]
        """
        Placeholder when ``transformers`` is not installed; do not instantiate.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "transformers and torch are required for WeightedRegressionTrainer."
            )


# -----------------------------------------------------------------------------
# Metrics factory and data collation (HF integration)
# -----------------------------------------------------------------------------


def build_hf_regression_compute_metrics_fn(
    min_score: int = 0,
    max_score: int = 5,
) -> Callable[[EvalPrediction], Dict[str, float]]:
    """
    Factory for ``Trainer`` ``compute_metrics`` using project ``evaluate_fold`` (QWK, RMSE, MAE).

    Parameters
    ----------
    min_score : int
        Lower clip for QWK rounding in ``evaluate_fold``.
    max_score : int
        Upper clip for QWK rounding in ``evaluate_fold``.

    Returns
    -------
    Callable[[EvalPrediction], Dict[str, float]]
        Function compatible with ``TrainingArguments(metric_for_best_model=...)``.
    """

    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        predictions, labels = eval_pred
        predictions = np.asarray(predictions, dtype=float).squeeze()
        # QWK requires integer y_true; consensus targets (e.g. 2.5) must be rounded
        # before ``evaluate_fold`` passes them to ``cohen_kappa_score``.
        labels = np.rint(np.asarray(labels, dtype=float).squeeze()).astype(int)
        metrics = evaluate_fold(
            labels, predictions, min_score=min_score, max_score=max_score
        )
        return {
            "qwk": float(metrics["qwk"]),
            "rmse": float(metrics["rmse"]),
            "mae": float(metrics["mae"]),
        }

    return compute_metrics


def build_weighted_regression_data_collator(tokenizer: Any) -> Any:
    """
    Build a batch collator that pads text fields and stacks ``labels`` + ``sample_weight``.

    ``DataCollatorWithPadding`` alone does not stack custom float columns; this wrapper
    pops ``labels`` and ``sample_weight`` before padding, then concatenates them as tensors.

    Requires ``TrainingArguments(remove_unused_columns=False)``: the default ``True`` strips
    any column not in the model's ``forward`` (including ``sample_weight``), which causes
    a ``KeyError`` in this collator.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerBase
        Tokenizer used for padding (must match training tokenization).

    Returns
    -------
    Callable[[List[Dict[str, Any]]], Dict[str, Any]]
        Collator suitable for ``TrainingArguments`` / ``Trainer``.
    """
    if torch is None or DataCollatorWithPadding is None:
        raise ImportError(
            "torch and transformers are required for build_weighted_regression_data_collator."
        )

    pad = DataCollatorWithPadding(tokenizer, padding=True, return_tensors="pt")

    def collate(features: List[Dict[str, Any]]) -> Dict[str, Any]:
        label_list = [
            torch.tensor(feature.pop("labels"), dtype=torch.float32)
            for feature in features
        ]
        weight_list = [
            torch.tensor(feature.pop("sample_weight"), dtype=torch.float32)
            for feature in features
        ]
        batch = pad(features)
        batch["labels"] = torch.stack(label_list)
        batch["sample_weight"] = torch.stack(weight_list)
        return batch

    return collate


def prepare_balanced_regression_weights(y_train: np.ndarray) -> np.ndarray:
    """
    Per-sample weights for regression (alias to ``compute_balanced_sample_weights``).

    Use these as ``sample_weight`` in HF datasets for ``WeightedRegressionTrainer``.

    Parameters
    ----------
    y_train : np.ndarray
        Training targets (ordinal integers 0..K); same rule as sklearn folds.

    Returns
    -------
    np.ndarray
        One positive weight per row, same length as ``y_train``.
    """
    return compute_balanced_sample_weights(y_train)


# -----------------------------------------------------------------------------
# Hugging Face ``datasets`` (tokenized regression rows for Trainer)
# -----------------------------------------------------------------------------


def build_tokenized_regression_dataset(
    tokenizer: Any,
    texts: List[str],
    labels: np.ndarray,
    sample_weights: np.ndarray,
    max_length: int,
) -> Any:
    """
    Build a tokenized ``datasets.Dataset`` with ``labels`` and ``sample_weight`` columns.

    Keeps notebook cells free of ``def`` blocks: notebooks call this from
    ``transformers_utils`` only. Does **not** alter ``modeling_utils`` or classical CV.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerBase
        Tokenizer matching the model (e.g. RoBERTa).
    texts : list of str
        Raw essay strings (e.g. ``Text_cleaned``), one per row.
    labels : np.ndarray
        Scalar regression targets, same length as ``texts``.
    sample_weights : np.ndarray
        Per-row weights for ``WeightedRegressionTrainer`` (balanced or ones).
    max_length : int
        Maximum sequence length for truncation (no padding here; collator pads batches).

    Returns
    -------
    datasets.Dataset
        Rows with tokenizer outputs plus ``labels`` and ``sample_weight``.
    """
    if Dataset is None:
        raise ImportError(
            "datasets is required for build_tokenized_regression_dataset."
        )

    # Map step: closure captures tokenizer and max_length so the notebook stays declarative.
    def tokenize_text_batch(examples: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    raw_dataset = Dataset.from_dict(
        {
            "text": texts,
            "labels": np.asarray(labels, dtype=np.float32),
            "sample_weight": np.asarray(sample_weights, dtype=np.float32),
        }
    )
    return raw_dataset.map(
        tokenize_text_batch,
        batched=True,
        remove_columns=["text"],
    )


# -----------------------------------------------------------------------------
# Suppress verbose LOAD REPORT table from transformers model loading
# -----------------------------------------------------------------------------

# ``log_state_dict_report`` logs via the logger passed from ``modeling_utils``
# (see ``transformers.modeling_utils``), not only ``loading_report``. Both must be
# raised to ERROR or the table still appears (e.g. in Colab / Jupyter).
_LOAD_REPORT_SUPPRESSION_LOGGER_NAMES: Tuple[str, ...] = (
    "transformers.modeling_utils",
    "transformers.utils.loading_report",
)


@contextmanager
def _suppress_load_report_logging_levels() -> Iterator[None]:
    """
    One-line purpose
        Temporarily set all LOAD REPORT-related loggers to ERROR and restore after.

    Parameters
    ----------
    None

    Returns
    -------
    contextmanager
        Yields once; restores previous levels in ``finally``.
    """
    previous_levels: Dict[str, int] = {
        name: logging.getLogger(name).level
        for name in _LOAD_REPORT_SUPPRESSION_LOGGER_NAMES
    }
    for name in _LOAD_REPORT_SUPPRESSION_LOGGER_NAMES:
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        yield
    finally:
        for name, level in previous_levels.items():
            logging.getLogger(name).setLevel(level)


@contextmanager
def suppress_hf_loading_report():
    """
    Temporarily silence the ``LOAD REPORT`` table printed by ``transformers``
    when loading a model whose checkpoint has expected missing / unexpected keys
    (e.g. classification head vs. LM head on ``roberta-base``).

    The report is emitted as a ``WARNING`` on ``transformers.modeling_utils`` (the
    logger passed into ``log_state_dict_report``) and may also use
    ``transformers.utils.loading_report`` when no logger is supplied. This context
    manager raises both to ``ERROR`` for the duration of the block, then restores
    the previous levels.

    Returns
    -------
    contextmanager
        Use with ``with suppress_hf_loading_report(): ...``.
    """
    with _suppress_load_report_logging_levels():
        yield


# -----------------------------------------------------------------------------
# Optuna resume helper
# -----------------------------------------------------------------------------


def get_completed_optuna_trial_count(storage: str, study_name: str) -> int:
    """
    Return the number of **COMPLETE** trials in an existing Optuna study.

    Safe to call when the study or database does not yet exist (returns ``0``).

    Parameters
    ----------
    storage : str
        Optuna storage URL (e.g. ``"sqlite:///optuna_study.db"``).
    study_name : str
        Name passed to ``optuna.create_study(study_name=...)``.

    Returns
    -------
    int
        Count of completed trials, or ``0`` if the study is not found.
    """
    if optuna is None:
        return 0
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        return len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        )
    except Exception:
        return 0


# -----------------------------------------------------------------------------
# Optuna hyperparameter search helpers
# -----------------------------------------------------------------------------


def build_model_init_fn(
    model_name: str,
    num_labels: int = 1,
    problem_type: str = "regression",
    n_trials: Optional[int] = None,
    suppress_load_report: bool = True,
) -> Callable:
    """
    Factory for ``model_init`` required by ``Trainer.hyperparameter_search``.

    Each Optuna trial needs a **freshly initialised** model so weights are not
    carried over from the previous trial. Optionally suppresses the verbose
    ``LOAD REPORT`` table and prints a trial progress line.

    Parameters
    ----------
    model_name : str
        Pretrained checkpoint name (e.g. ``"roberta-base"``).
    num_labels : int
        Number of output logits (1 for regression).
    problem_type : str
        HF model ``problem_type`` (``"regression"`` for scalar MSE head).
    n_trials : int or None
        Total target trial count for display (e.g. ``15``). When set together
        with a valid ``trial`` object, prints ``Trial k / n_trials``.
    suppress_load_report : bool
        If ``True`` (default), silence the ``LOAD REPORT`` table on every
        ``from_pretrained`` call (same loggers as ``suppress_hf_loading_report``).

    Returns
    -------
    Callable
        ``model_init(trial)`` suitable for ``Trainer(model_init=...)``.
    """
    if AutoModelForSequenceClassification is None:
        raise ImportError("transformers is required for build_model_init_fn.")

    def model_init(trial: Any = None) -> Any:
        # Print trial progress banner when Optuna trial info is available.
        if trial is not None and n_trials is not None:
            print(f"\n{'=' * 50}")
            print(f"  Trial {trial.number + 1} / {n_trials}")
            print(f"{'=' * 50}")

        if suppress_load_report:
            with _suppress_load_report_logging_levels():
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=num_labels,
                    problem_type=problem_type,
                )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                problem_type=problem_type,
            )

        return model

    return model_init


def build_optuna_hp_space_fn(
    lr_min: float = 1e-5,
    lr_max: float = 5e-5,
    epochs_min: int = 2,
    epochs_max: int = 5,
    batch_sizes: Optional[List[int]] = None,
    weight_decay_min: float = 0.0,
    weight_decay_max: float = 0.1,
    warmup_ratio_min: float = 0.0,
    warmup_ratio_max: float = 0.15,
) -> Callable:
    """
    Factory for the ``hp_space`` callable passed to ``Trainer.hyperparameter_search``.

    Returns a function that maps an Optuna ``trial`` to a dict of
    ``TrainingArguments`` overrides. Ranges default to a **narrow** search
    around standard BERT/RoBERTa fine-tuning values.

    Parameters
    ----------
    lr_min, lr_max : float
        Learning-rate bounds (log-uniform).
    epochs_min, epochs_max : int
        Epoch count bounds (integer).
    batch_sizes : list of int or None
        Categorical choices for ``per_device_train_batch_size``.
        Defaults to ``[8, 16]``.
    weight_decay_min, weight_decay_max : float
        Weight-decay bounds (uniform).
    warmup_ratio_min, warmup_ratio_max : float
        Warmup-ratio bounds (uniform).

    Returns
    -------
    Callable
        ``optuna_hp_space(trial)`` compatible with ``hyperparameter_search``.
    """
    if batch_sizes is None:
        batch_sizes = [8, 16]

    def optuna_hp_space(trial: Any) -> Dict[str, Any]:
        return {
            "learning_rate": trial.suggest_float(
                "learning_rate",
                lr_min,
                lr_max,
                log=True,
            ),
            "num_train_epochs": trial.suggest_int(
                "num_train_epochs",
                epochs_min,
                epochs_max,
            ),
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size",
                batch_sizes,
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay",
                weight_decay_min,
                weight_decay_max,
            ),
            "warmup_ratio": trial.suggest_float(
                "warmup_ratio",
                warmup_ratio_min,
                warmup_ratio_max,
            ),
        }

    return optuna_hp_space


def build_compute_objective_fn(
    metric_key: str = "eval_qwk",
) -> Callable[[Dict[str, float]], float]:
    """
    Factory for ``compute_objective`` used by ``Trainer.hyperparameter_search``.

    Maps the evaluation metrics dict to **one scalar** that Optuna optimises.

    Parameters
    ----------
    metric_key : str
        Key in the metrics dict to maximise (e.g. ``"eval_qwk"``).

    Returns
    -------
    Callable[[Dict[str, float]], float]
        Function returning the objective value for one trial.
    """

    def compute_objective(metrics: Dict[str, float]) -> float:
        return metrics[metric_key]

    return compute_objective
