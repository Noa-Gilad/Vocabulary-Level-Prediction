"""
Vocabulary Level Prediction — Hugging Face Transformers helpers.

Weighted regression training (e.g. RoBERTa with a scalar head), batch collation with
``sample_weight``, and evaluation metrics aligned with ``modeling_utils.evaluate_fold``.
Depends optionally on ``torch`` and ``transformers``; import this module only in notebooks
or code paths that fine-tune transformers.
"""

# -----------------------------------------------------------------------------
# Environment setting
# -----------------------------------------------------------------------------
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# PyTorch / Hugging Face are optional so environments without GPU stacks can still import
# ``modeling_utils`` and classical baselines without installing ``transformers``.
try:
    import torch
    import torch.nn.functional as torch_functional
    from transformers import EvalPrediction, Trainer
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[misc, assignment]
    torch_functional = None  # type: ignore[misc, assignment]
    EvalPrediction = Any  # type: ignore[misc, assignment]
    Trainer = None  # type: ignore[misc, assignment]

# ``evaluate_fold`` / ``compute_balanced_sample_weights`` are imported lazily inside helpers
# so importing this module does not load all of ``modeling_utils`` (and matplotlib, etc.).


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
    from modeling_utils import evaluate_fold

    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        predictions, labels = eval_pred
        predictions = np.asarray(predictions, dtype=float).squeeze()
        labels = np.asarray(labels, dtype=float).squeeze()
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

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerBase
        Tokenizer used for padding (must match training tokenization).

    Returns
    -------
    Callable[[List[Dict[str, Any]]], Dict[str, Any]]
        Collator suitable for ``TrainingArguments`` / ``Trainer``.
    """
    if torch is None:
        raise ImportError(
            "torch is required for build_weighted_regression_data_collator."
        )
    try:
        from transformers import DataCollatorWithPadding
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "transformers is required for build_weighted_regression_data_collator."
        ) from exc

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
    from modeling_utils import compute_balanced_sample_weights

    return compute_balanced_sample_weights(y_train)
