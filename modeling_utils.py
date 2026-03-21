"""
Modeling utilities for vocabulary-level prediction.

Functions for target creation, rater-agreement filtering,
cross-validation evaluation, sample-weight generation, experiment
orchestration (``run_all_experiments_cv`` for the full registry; ``run_registry_experiments_cv`` for row subsets),
leaderboard aggregation, CV visualization, and Optuna helpers
(``optuna_optimize_with_stratified_cv`` plus thin TF-IDF+SVD+MLP wrappers).
Feature extraction for text/embeddings stays in ``utils.py``; this module orchestrates models.
"""

from __future__ import annotations

import os
import warnings
from contextlib import nullcontext
from typing import Any, Callable, Collection, Dict, List, Mapping, Optional, Sequence, Tuple

# Limit BLAS/OpenMP oversubscription during sklearn fits so the kernel stays responsive to
# interrupts (multi-threaded native code can otherwise ignore Ctrl+C for long stretches).
try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover
    threadpool_limits = None  # type: ignore[misc, assignment]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    cohen_kappa_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError:  # pragma: no cover
    optuna = None  # type: ignore[misc, assignment]
    MedianPruner = None  # type: ignore[misc, assignment]
    TPESampler = None  # type: ignore[misc, assignment]


# ---------------------------------------------------------------------------
# Target creation
# ---------------------------------------------------------------------------


def filter_by_rater_agreement(
    data_frame: pd.DataFrame,
    rater_1_column: str = "Vocabulary_1",
    rater_2_column: str = "Vocabulary_2",
    max_allowed_gap: int = 2,
) -> pd.DataFrame:
    """
    Remove rows where raters disagree by >= max_allowed_gap points.

    Parameters
    ----------
    data_frame : pd.DataFrame
        Input dataframe containing two rater columns.
    rater_1_column : str
        Name of first rater column.
    rater_2_column : str
        Name of second rater column.
    max_allowed_gap : int
        Rows with abs(rater1 - rater2) >= this value are dropped.

    Returns
    -------
    pd.DataFrame
        Filtered copy with reset index.
    """
    gap = (data_frame[rater_1_column] - data_frame[rater_2_column]).abs()
    return data_frame.loc[gap < max_allowed_gap].reset_index(drop=True)


def create_consensus_target(
    data_frame: pd.DataFrame,
    rater_1_column: str = "Vocabulary_1",
    rater_2_column: str = "Vocabulary_2",
    target_column: str = "target_vocab",
    min_score: int = 0,
    max_score: int = 5,
) -> pd.DataFrame:
    """
    Build consensus target: mean of two raters, rounded and clipped.

    Rounding uses np.rint (banker's rounding) — fixed rule everywhere.

    Parameters
    ----------
    data_frame : pd.DataFrame
        Dataframe containing the two rater columns.
    rater_1_column : str
        Name of first rater column.
    rater_2_column : str
        Name of second rater column.
    target_column : str
        Name of the new target column to create.
    min_score : int
        Lower bound of valid score range.
    max_score : int
        Upper bound of valid score range.

    Returns
    -------
    pd.DataFrame
        Same dataframe with an added target_column.
    """
    mean_scores = data_frame[[rater_1_column, rater_2_column]].mean(axis=1)
    data_frame[target_column] = (
        np.rint(mean_scores).astype(int).clip(min_score, max_score)
    )
    return data_frame


def build_text_target_dataset(
    data_frame: pd.DataFrame,
    text_column: str = "Text_cleaned",
    target_column: str = "target_vocab",
) -> pd.DataFrame:
    """
    Create a clean modeling dataset with only essay text and final target.

    Parameters
    ----------
    data_frame : pd.DataFrame
        Input dataframe containing text and target columns.
    text_column : str
        Name of the essay text column to keep.
    target_column : str
        Name of the final target column to keep.

    Returns
    -------
    pd.DataFrame
        Two-column dataframe: [text_column, target_column], index reset.
    """
    required_columns = [text_column, target_column]
    missing_columns = [
        column_name
        for column_name in required_columns
        if column_name not in data_frame.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required columns for text-target dataset: {missing_columns}"
        )

    text_target_data_frame = data_frame[[text_column, target_column]].copy()
    text_target_data_frame = text_target_data_frame.reset_index(drop=True)
    return text_target_data_frame


# ---------------------------------------------------------------------------
# Stratification helper
# ---------------------------------------------------------------------------


def build_stratification_bins(
    target_values: np.ndarray,
    merge_below: int = 2,
) -> np.ndarray:
    """
    Create binned labels for StratifiedKFold when rare classes exist.

    Merges all labels < merge_below into one bin so every bin has enough
    samples for stratified splitting. Original labels are unchanged for
    training and evaluation — bins are used only for fold assignment.

    Parameters
    ----------
    target_values : np.ndarray
        Original ordinal target labels.
    merge_below : int
        Labels strictly below this value are merged into one bin.

    Returns
    -------
    np.ndarray
        Binned labels (same length as target_values).
    """
    bins = target_values.copy()
    bins[bins < merge_below] = merge_below
    return bins


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def round_and_clip_predictions(
    predictions: np.ndarray,
    min_score: int = 0,
    max_score: int = 5,
) -> np.ndarray:
    """
    Convert continuous predictions to valid ordinal labels.

    Parameters
    ----------
    predictions : np.ndarray
        Raw continuous model output.
    min_score : int
        Lower bound of valid score range.
    max_score : int
        Upper bound of valid score range.

    Returns
    -------
    np.ndarray
        Rounded and clipped integer predictions.
    """
    return np.clip(np.rint(predictions).astype(int), min_score, max_score)


def compute_qwk(
    y_true: np.ndarray,
    y_pred_continuous: np.ndarray,
    min_score: int = 0,
    max_score: int = 5,
) -> float:
    """
    Quadratic weighted kappa on rounded-and-clipped predictions.

    Uses ``labels=np.arange(min_score, max_score + 1)`` so every fold uses the **same**
    0..K ordinal frame as sklearn's weight matrix. Relying on inferred labels per fold
    can skew QWK when a validation fold omits some score levels in ``y_true`` or in
    rounded predictions.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth ordinal labels.
    y_pred_continuous : np.ndarray
        Continuous predictions (rounded internally).
    min_score : int
        Lower bound for rounding clip.
    max_score : int
        Upper bound for rounding clip.

    Returns
    -------
    float
        QWK score; may be ``nan`` if sklearn cannot compute kappa (degenerate fold).
    """
    y_pred_rounded = round_and_clip_predictions(y_pred_continuous, min_score, max_score)
    ordinal_labels = np.arange(min_score, max_score + 1)
    kappa = cohen_kappa_score(
        y_true,
        y_pred_rounded,
        labels=ordinal_labels,
        weights="quadratic",
    )
    return float(kappa)


def evaluate_fold(
    y_true: np.ndarray,
    y_pred_continuous: np.ndarray,
    min_score: int = 0,
    max_score: int = 5,
) -> Dict[str, float]:
    """
    Compute QWK, RMSE, MAE for one set of predictions.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred_continuous : np.ndarray
        Continuous predictions.
    min_score : int
        Lower bound for QWK rounding.
    max_score : int
        Upper bound for QWK rounding.

    Returns
    -------
    Dict[str, float]
        Keys: qwk, rmse, mae.
        **QWK** uses rounded/clipped predictions; **RMSE** and **MAE** use raw continuous
        ``y_pred_continuous`` vs integer ``y_true`` (standard regression errors).
    """
    return {
        "qwk": compute_qwk(y_true, y_pred_continuous, min_score, max_score),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred_continuous))),
        "mae": float(mean_absolute_error(y_true, y_pred_continuous)),
    }


# ---------------------------------------------------------------------------
# Sample weights
# ---------------------------------------------------------------------------


def compute_balanced_sample_weights(y_train: np.ndarray) -> np.ndarray:
    """
    Balanced per-sample weights derived from train-fold labels only.

    Parameters
    ----------
    y_train : np.ndarray
        Training-fold target labels.

    Returns
    -------
    np.ndarray
        Weight array aligned with y_train.
    """
    return compute_sample_weight(class_weight="balanced", y=y_train)


class UniformRandomScoreRegressor(BaseEstimator, RegressorMixin):
    """
    Ignores features; predicts continuous scores uniform on [min_score, max_score].

    Use as a sanity-check baseline: QWK should stay near zero and RMSE/MAE worse than
    any reasonable model when the label scale is 0-5.
    """

    def __init__(
        self,
        *,
        random_state: Optional[int] = None,
        min_score: float = 0.0,
        max_score: float = 5.0,
    ) -> None:
        self.random_state = random_state
        self.min_score = min_score
        self.max_score = max_score

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Any = None,
    ) -> UniformRandomScoreRegressor:
        """
        Learn input dimensionality only; random predictions do not use ``y``.

        Parameters
        ----------
        X : Any
            Training design matrix (only shape is used).
        y : Any
            Training targets (ignored for prediction logic).
        sample_weight : Any
            Ignored; accepted for Pipeline compatibility.

        Returns
        -------
        UniformRandomScoreRegressor
            Fitted self.
        """
        # sample_weight is ignored; Pipeline may still pass model__sample_weight.
        _ = sample_weight
        X, y = check_X_y(
            X,
            y,
            accept_sparse=True,
            y_numeric=True,
            multi_output=False,
            estimator=self,
        )
        self.rng_ = np.random.RandomState(self.random_state)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """
        Draw one uniform score per row in ``[min_score, max_score]``.

        Parameters
        ----------
        X : Any
            Feature matrix; only ``n_samples`` is used.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples,)``, dtype float64.
        """
        check_is_fitted(self, "rng_")
        X = check_array(X, accept_sparse=True)
        n_samples = X.shape[0]
        return self.rng_.uniform(
            self.min_score,
            self.max_score,
            size=n_samples,
        )


# ---------------------------------------------------------------------------
# Pipeline builders (modular — all hyperparameters passed in by caller)
# ---------------------------------------------------------------------------

# Reference defaults only; notebooks may copy or override when calling builders.
DEFAULT_TFIDF_VECTORIZER_KWARGS: Dict[str, Any] = {
    "max_features": 20_000,
    "ngram_range": (1, 2),
    "sublinear_tf": True,
    "min_df": 1,
    "max_df": 0.95,
}

# Default ``max_iter`` search range for Optuna MLP trials (early stopping still applies).
OPTUNA_MLP_MAX_ITER_BOUNDS: Tuple[int, int] = (300, 800)

# Tree / boosting short names: on the embedding track we skip StandardScaler (plan).
DEFAULT_EMBEDDING_TREE_MODEL_SHORT_NAMES: frozenset[str] = frozenset(
    {
        "RF",
        "XGB",
        "LGBM",
    }
)


def make_handcrafted_regression_pipeline(final_estimator: Any) -> Pipeline:
    """
    Pipeline: StandardScaler + regressor for numeric handcrafted features.

    Parameters
    ----------
    final_estimator : Any
        sklearn regressor (last step name will be "model" for sample_weight routing).

    Returns
    -------
    Pipeline
        Fitted per CV fold on train indices only.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", final_estimator),
        ]
    )


def make_embeddings_regression_pipeline(
    final_estimator: Any,
    model_short_name: str,
    tree_model_short_names: Collection[str],
) -> Any:
    """
    Frozen-embedding track: scale linear-like models; leave tree/boosting estimators unscaled.

    Matches the model-selection plan: StandardScaler inside the pipeline for Ridge, Lasso,
    ElasticNet, LinearSVR, MLP; no scaler for RandomForest, XGBoost, LightGBM.

    Parameters
    ----------
    final_estimator : Any
        Unfitted regressor for this experiment.
    model_short_name : str
        Short label for the experiment (e.g. "Ridge", "RF") used to decide scaling.
    tree_model_short_names : Collection[str]
        Short names that skip StandardScaler (tree or gradient-boosting regressors).

    Returns
    -------
    Any
        ``Pipeline([StandardScaler, model])`` or the bare ``final_estimator``.
    """
    if model_short_name in tree_model_short_names:
        return final_estimator
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", final_estimator),
        ]
    )


def make_tfidf_regression_pipeline(
    final_estimator: Any,
    vectorizer_kwargs: Mapping[str, Any],
) -> Pipeline:
    """
    Pipeline: TfidfVectorizer + regressor (vectorizer fits inside each CV fold).

    Parameters
    ----------
    final_estimator : Any
        sklearn regressor compatible with sparse input (or dense after transform).
    vectorizer_kwargs : Mapping[str, Any]
        Keyword arguments for TfidfVectorizer (caller owns all choices).

    Returns
    -------
    Pipeline
        Full text pipeline for one experiment.
    """
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**dict(vectorizer_kwargs))),
            ("model", final_estimator),
        ]
    )


def make_tfidf_mlp_pipeline(
    vectorizer_kwargs: Mapping[str, Any],
    n_svd_components: int = 300,
    mlp_estimator: Any | None = None,
    random_state: int = 42,
) -> Pipeline:
    """
    Pipeline: TfidfVectorizer + TruncatedSVD + MLPRegressor for sparse TF-IDF.

    Parameters
    ----------
    vectorizer_kwargs : Mapping[str, Any]
        Keyword arguments for TfidfVectorizer.
    n_svd_components : int
        TruncatedSVD output dimension before MLP.
    mlp_estimator : Any | None
        If None, builds a small MLP with ``early_stopping`` (matches typical TF-IDF+SVD screening
        in ``Model_selection.ipynb``).
    random_state : int
        Seed for SVD and default MLP.

    Returns
    -------
    Pipeline
        Dense-friendly path for neural net on TF-IDF.

    Notes
    -----
    We use ``TruncatedSVD`` (fit per fold) to produce a dense, lower-dimensional input for
    the MLP. An alternative is ``StandardScaler(with_mean=False)`` on sparse TF-IDF; SVD is
    preferred here for stability and dimensionality control.
    """
    if mlp_estimator is None:
        mlp_estimator = MLPRegressor(
            hidden_layer_sizes=(256,),
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=random_state,
        )
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**dict(vectorizer_kwargs))),
            (
                "svd",
                TruncatedSVD(n_components=n_svd_components, random_state=random_state),
            ),
            ("mlp", mlp_estimator),
        ]
    )


def build_pipeline_for_track(
    feature_source: str,
    model_short_name: str,
    final_estimator: Any,
    *,
    tfidf_vectorizer_kwargs: Mapping[str, Any],
    svd_n_components: int = 300,
    tfidf_mlp_short_name: str = "MLP",
    embedding_tree_model_short_names: Collection[str] | None = None,
) -> Any:
    """
    Return one sklearn estimator or Pipeline for a (feature track, model) pair.

    Dispatches on feature_source and model_short_name (e.g. MLP on tfidf uses SVD).

    Parameters
    ----------
    feature_source : str
        One of "handcrafted", "tfidf", "embeddings", or "random_uniform".
    model_short_name : str
        Short label used in experiment names (e.g. "Ridge", "MLP").
    final_estimator : Any
        Unfitted regressor for this experiment.
    tfidf_vectorizer_kwargs : Mapping[str, Any]
        Passed to TfidfVectorizer when feature_source is "tfidf".
    svd_n_components : int
        Used when feature_source is "tfidf" and model_short_name matches MLP branch.
    tfidf_mlp_short_name : str
        Which short name triggers TF-IDF + SVD + MLP path (default "MLP").
    embedding_tree_model_short_names : Collection[str] | None
        For ``feature_source=="embeddings"``, short names that skip ``StandardScaler``.
        If None, uses ``DEFAULT_EMBEDDING_TREE_MODEL_SHORT_NAMES``.

    Returns
    -------
    Any
        Pipeline or bare estimator ready for run_stratified_cv.
    """
    if embedding_tree_model_short_names is None:
        embedding_tree_model_short_names = DEFAULT_EMBEDDING_TREE_MODEL_SHORT_NAMES

    # No scaler: baseline ignores features; scaling would be meaningless noise.
    if feature_source == "random_uniform":
        return Pipeline([("model", final_estimator)])

    if feature_source == "handcrafted":
        return make_handcrafted_regression_pipeline(final_estimator)
    if feature_source == "embeddings":
        return make_embeddings_regression_pipeline(
            final_estimator,
            model_short_name=model_short_name,
            tree_model_short_names=embedding_tree_model_short_names,
        )
    if feature_source == "tfidf":
        if model_short_name == tfidf_mlp_short_name:
            return make_tfidf_mlp_pipeline(
                vectorizer_kwargs=tfidf_vectorizer_kwargs,
                n_svd_components=svd_n_components,
                mlp_estimator=final_estimator,
            )
        return make_tfidf_regression_pipeline(
            final_estimator,
            vectorizer_kwargs=tfidf_vectorizer_kwargs,
        )
    raise ValueError(
        f"Unknown feature_source: {feature_source!r}. "
        "Expected 'handcrafted', 'tfidf', 'embeddings', or 'random_uniform'."
    )


def build_experiments_from_grid(
    feature_tracks: Sequence[Mapping[str, str]],
    named_estimators: Sequence[Tuple[str, Any]],
    *,
    tfidf_vectorizer_kwargs: Mapping[str, Any],
    svd_n_components: int = 300,
    tfidf_mlp_short_name: str = "MLP",
    embedding_tree_model_short_names: Collection[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Cross product of feature tracks and (name, estimator) pairs into experiment dicts.

    For asymmetric layouts (e.g. Ridge-only on handcrafted, all models on TF-IDF),
    use ``build_experiments_from_track_specs`` instead.

    Each experiment has keys: name, feature_source, pipeline.
    Pipelines are built via build_pipeline_for_track (no hardcoded model hyperparameters here).

    Parameters
    ----------
    feature_tracks : Sequence[Mapping[str, str]]
        Each mapping must include "feature_source" and "name_prefix" (for experiment names).
    named_estimators : Sequence[Tuple[str, Any]]
        Pairs of (model_short_name, unfitted_estimator).
    tfidf_vectorizer_kwargs : Mapping[str, Any]
        Passed through to TF-IDF pipelines.
    svd_n_components : int
        TruncatedSVD size for TF-IDF + MLP branch.
    tfidf_mlp_short_name : str
        Short name that selects TF-IDF + SVD + MLP pipeline.
    embedding_tree_model_short_names : Collection[str] | None
        Passed to ``build_pipeline_for_track`` for embedding tracks.

    Returns
    -------
    List[Dict[str, Any]]
        Ordered experiment specifications for run_registry_experiments_cv.
    """
    experiments: List[Dict[str, Any]] = []
    for track in feature_tracks:
        feature_source = track["feature_source"]
        name_prefix = track["name_prefix"]
        for model_short_name, estimator in named_estimators:
            pipeline = build_pipeline_for_track(
                feature_source=feature_source,
                model_short_name=model_short_name,
                final_estimator=estimator,
                tfidf_vectorizer_kwargs=tfidf_vectorizer_kwargs,
                svd_n_components=svd_n_components,
                tfidf_mlp_short_name=tfidf_mlp_short_name,
                embedding_tree_model_short_names=embedding_tree_model_short_names,
            )
            experiments.append(
                {
                    "name": f"{name_prefix}_{model_short_name}",
                    "feature_source": feature_source,
                    "pipeline": pipeline,
                }
            )
    return experiments


def build_experiments_from_track_specs(
    track_specs: Sequence[Mapping[str, Any]],
    *,
    tfidf_vectorizer_kwargs: Mapping[str, Any],
    svd_n_components: int = 300,
    tfidf_mlp_short_name: str = "MLP",
    embedding_tree_model_short_names: Collection[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Build an experiment registry where each feature track has its own estimator list.

    Use this when tracks are not a full cross-product (e.g. handcrafted Ridge-only
    baseline, while TF-IDF and embeddings run against every model).

    Parameters
    ----------
    track_specs : Sequence[Mapping[str, Any]]
        Each mapping must include:
        - feature_source: str — "handcrafted", "tfidf", or "embeddings".
        - name_prefix: str — prefix for experiment names.
        - named_estimators: Sequence[Tuple[str, Any]] — (short_name, unfitted_estimator)
          pairs **for this track only**.
    tfidf_vectorizer_kwargs : Mapping[str, Any]
        Passed to TF-IDF pipelines when a track uses feature_source "tfidf".
    svd_n_components : int
        TruncatedSVD size for TF-IDF + MLP branch.
    tfidf_mlp_short_name : str
        Short name that selects TF-IDF + SVD + MLP pipeline.
    embedding_tree_model_short_names : Collection[str] | None
        Tree short names that skip scaling on the embedding track.

    Returns
    -------
    List[Dict[str, Any]]
        Ordered experiment dicts (name, feature_source, pipeline) for
        run_registry_experiments_cv.
    """
    experiments: List[Dict[str, Any]] = []

    for track_index, track in enumerate(track_specs):
        try:
            feature_source = track["feature_source"]
            name_prefix = track["name_prefix"]
            named_estimators = track["named_estimators"]
        except KeyError as exc:
            missing_key = exc.args[0]
            raise KeyError(
                f"track_specs[{track_index}] missing required key {missing_key!r}. "
                "Expected keys: feature_source, name_prefix, named_estimators."
            ) from exc

        if not named_estimators:
            raise ValueError(
                f"track_specs[{track_index}] has empty named_estimators for "
                f"feature_source={feature_source!r}."
            )

        for model_short_name, estimator in named_estimators:
            pipeline = build_pipeline_for_track(
                feature_source=feature_source,
                model_short_name=model_short_name,
                final_estimator=estimator,
                tfidf_vectorizer_kwargs=tfidf_vectorizer_kwargs,
                svd_n_components=svd_n_components,
                tfidf_mlp_short_name=tfidf_mlp_short_name,
                embedding_tree_model_short_names=embedding_tree_model_short_names,
            )
            experiments.append(
                {
                    "name": f"{name_prefix}_{model_short_name}",
                    "feature_source": feature_source,
                    "pipeline": pipeline,
                }
            )

    return experiments


def build_experiment_registry(
    track_specs: Sequence[Mapping[str, Any]],
    *,
    tfidf_vectorizer_kwargs: Mapping[str, Any],
    svd_n_components: int = 300,
    tfidf_mlp_short_name: str = "MLP",
    embedding_tree_model_short_names: Collection[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Build the Section 4 experiment list (name, feature_source, pipeline) from track specs.

    Project-plan name for ``build_experiments_from_track_specs`` (same behavior).

    Parameters
    ----------
    track_specs : Sequence[Mapping[str, Any]]
        Per-track ``named_estimators`` lists (see ``build_experiments_from_track_specs``).
    tfidf_vectorizer_kwargs : Mapping[str, Any]
        Keyword arguments for ``TfidfVectorizer`` on the TF-IDF track.
    svd_n_components : int
        TruncatedSVD dimension for TF-IDF + MLP.
    tfidf_mlp_short_name : str
        Short name selecting the TF-IDF + SVD + MLP pipeline.
    embedding_tree_model_short_names : Collection[str] | None
        Embedding-track tree short names that skip ``StandardScaler``.

    Returns
    -------
    List[Dict[str, Any]]
        Experiment registry for ``run_registry_experiments_cv`` / ``run_all_experiments_cv``.
    """
    return build_experiments_from_track_specs(
        track_specs,
        tfidf_vectorizer_kwargs=tfidf_vectorizer_kwargs,
        svd_n_components=svd_n_components,
        tfidf_mlp_short_name=tfidf_mlp_short_name,
        embedding_tree_model_short_names=embedding_tree_model_short_names,
    )


# ---------------------------------------------------------------------------
# Cross-validation runner
# ---------------------------------------------------------------------------


def _pipeline_first_step_is_tfidf_vectorizer(model: Any) -> bool:
    """
    Return True if model is a Pipeline whose first step is a TfidfVectorizer.

    Parameters
    ----------
    model : Any
        Estimator or Pipeline.

    Returns
    -------
    bool
        True when TF-IDF must see raw text and fit only inside each CV fold.
    """
    if not isinstance(model, Pipeline) or not model.steps:
        return False
    _, first_transformer = model.steps[0]
    return isinstance(first_transformer, TfidfVectorizer)


def _validate_raw_text_features_for_tfidf_pipeline(
    model: Any,
    feature_storage: Any,
) -> None:
    """
    Fail fast if a TF-IDF pipeline would receive numeric precomputed features.

    TF-IDF vocabulary and IDF must be learned only from fold training text;
    passing a dense/sparse numeric matrix would usually mean prior global fit.

    Parameters
    ----------
    model : Any
        Estimator passed to run_stratified_cv.
    feature_storage : Any
        Full-column Series, single-column DataFrame, or 1D array aligned with y.

    Returns
    -------
    None
        Raises ValueError on invalid input.
    """
    if not _pipeline_first_step_is_tfidf_vectorizer(model):
        return

    if isinstance(feature_storage, pd.DataFrame):
        if feature_storage.shape[1] != 1:
            raise ValueError(
                "TF-IDF CV: pass a single text column as a pandas Series (or one-column DataFrame), "
                "not a wide numeric matrix."
            )
        series = feature_storage.iloc[:, 0]
    elif isinstance(feature_storage, pd.Series):
        series = feature_storage
    else:
        arr = np.asarray(feature_storage)
        if arr.ndim != 1:
            raise ValueError(
                "TF-IDF CV: expected a 1D array of document strings per row. "
                "Do not pass precomputed TF-IDF matrices — fit TfidfVectorizer inside each fold via Pipeline."
            )
        if arr.size == 0:
            return
        if np.issubdtype(arr.dtype, np.number):
            raise ValueError(
                "TF-IDF CV: numeric array detected. Use raw text strings so the vectorizer "
                "fits on training-fold documents only (avoids leakage)."
            )
        return

    if series.dtype.kind in ("i", "u", "f", "c") or np.issubdtype(
        series.dtype, np.number
    ):
        raise ValueError(
            "TF-IDF CV: numeric Series dtype. Pass object/string text (e.g. Text_cleaned)."
        )


def _materialize_fold_features(feature_storage: Any, row_indices: np.ndarray) -> Any:
    """
    Subset features by fold indices without copying the full corpus logic elsewhere.

    Parameters
    ----------
    feature_storage : Any
        Series, one-column DataFrame, or ndarray (same row order as y).
    row_indices : np.ndarray
        Integer indices for one fold (train or validation).

    Returns
    -------
    Any
        Slice suitable for Pipeline.fit or predict (Series, DataFrame, or ndarray).
    """
    if isinstance(feature_storage, pd.Series):
        # TF-IDF pipeline expects a Series of strings; keep as Series.
        return feature_storage.iloc[row_indices].reset_index(drop=True)
    if isinstance(feature_storage, pd.DataFrame):
        # Convert to numpy so downstream estimators (XGBoost, LightGBM) never see pandas
        # column names during fit() and then miss them during predict() — that mismatch is
        # the root cause of the "X does not have valid feature names" UserWarning.
        return feature_storage.iloc[row_indices].to_numpy()
    arr = np.asarray(feature_storage)
    return arr[row_indices]


def _get_sample_weight_param_name(pipeline_or_model: Any) -> str:
    """
    Resolve the correct sample_weight parameter name for fit().

    For a Pipeline the final step needs the double-underscore prefix.
    For a bare estimator it's just 'sample_weight'.

    Parameters
    ----------
    pipeline_or_model : Any
        Sklearn estimator or Pipeline.

    Returns
    -------
    str
        Parameter name to pass in fit(**{name: weights}).
    """
    if isinstance(pipeline_or_model, Pipeline):
        final_step_name = pipeline_or_model.steps[-1][0]
        return f"{final_step_name}__sample_weight"
    return "sample_weight"


def stratified_subsample_for_optuna_objective(
    raw_text_series: pd.Series,
    target_values: np.ndarray,
    stratification_bins: np.ndarray | None,
    n_samples: int,
    random_state: int,
) -> Tuple[pd.Series, np.ndarray, np.ndarray | None]:
    """
    Draw a stratified random subset of rows for fast Optuna objectives (approximate ranking).

    One-line purpose
        Shrink TF-IDF + MLP CV cost per trial while preserving class balance in fold splits.

    Parameters
    ----------
    raw_text_series : pd.Series
        Full-corpus cleaned text (same order as ``target_values``).
    target_values : np.ndarray
        Ordinal targets (length ``n``).
    stratification_bins : np.ndarray | None
        Labels for stratification; if None, uses ``target_values``.
    n_samples : int
        Target subset size (capped at ``n``).
    random_state : int
        RNG seed for reproducible subsampling.

    Returns
    -------
    Tuple[pd.Series, np.ndarray, np.ndarray | None]
        ``(text_subset, y_subset, strat_subset_or_none)`` with ``len == min(n_samples, n)``.
    """
    y = np.asarray(target_values)
    n_total = int(y.shape[0])
    if n_total == 0:
        raise ValueError("target_values is empty.")
    strat = stratification_bins if stratification_bins is not None else y
    strat = np.asarray(strat)
    if strat.shape[0] != n_total:
        raise ValueError("stratification_bins must align with target_values length.")

    take = min(int(n_samples), n_total)
    if take == n_total:
        return raw_text_series.reset_index(drop=True), y, stratification_bins

    # One split yields a stratified train set of exactly ``take`` rows (when feasible).
    try:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            train_size=take,
            random_state=random_state,
        )
        train_idx, _ = next(splitter.split(np.zeros(n_total), strat))
    except ValueError as exc:
        warnings.warn(
            "Stratified subsample failed; falling back to unstratified indices. "
            f"Reason: {exc}",
            UserWarning,
            stacklevel=2,
        )
        rng = np.random.RandomState(random_state)
        train_idx = rng.choice(n_total, size=take, replace=False)

    idx = np.sort(train_idx.astype(int, copy=False))
    text_sub = raw_text_series.iloc[idx].reset_index(drop=True)
    y_sub = y[idx]
    strat_sub = strat[idx]
    if stratification_bins is None:
        return text_sub, y_sub, None
    return text_sub, y_sub, strat_sub


def run_stratified_cv(
    model: Any,
    feature_matrix: np.ndarray | pd.DataFrame | pd.Series,
    target_values: np.ndarray,
    stratification_bins: np.ndarray | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    use_sample_weights: bool = True,
    min_score: int = 0,
    max_score: int = 5,
    after_each_fold: Optional[Callable[[int, Dict[str, float]], None]] = None,
    *,
    verbose: bool = False,
    limit_blas_threads: bool = False,
) -> pd.DataFrame:
    """
    Stratified K-fold CV for one regressor; returns per-fold metrics.

    Leakage control: each fold uses ``clone(model).fit(X_train, y_train)``.
    If the first Pipeline step is ``TfidfVectorizer``, vocabulary/IDF are fit only
    on training-fold documents; validation is transformed with that same vectorizer.

    Parameters
    ----------
    model : Any
        Scikit-learn-compatible estimator or Pipeline.
    feature_matrix : np.ndarray | pd.DataFrame | pd.Series
        Features row-aligned with ``target_values``. For TF-IDF pipelines pass **raw text**
        (e.g. ``Text_cleaned`` Series) — never a precomputed TF-IDF matrix.
    target_values : np.ndarray
        Ordinal target labels for evaluation.
    stratification_bins : np.ndarray | None
        Binned labels for fold assignment (e.g. rare-class merged).
        If None, target_values are used directly.
    n_splits : int
        Number of stratified folds.
    random_state : int
        Seed for reproducible fold splitting.
    use_sample_weights : bool
        Whether to pass balanced sample weights to fit().
    min_score : int
        Lower bound for QWK rounding.
    max_score : int
        Upper bound for QWK rounding.
    after_each_fold : callable, optional
        If set, called as ``after_each_fold(fold_number, metrics_dict)`` after each
        fold with keys qwk, rmse, mae (for Optuna pruning / logging).
    verbose : bool
        If True, print each fold before/after fit so long runs show progress (flush stdout).
    limit_blas_threads : bool
        If True, wrap each ``fit`` in ``threadpoolctl.threadpool_limits(1)`` when available
        so BLAS/OpenMP does not use all cores; this often makes Jupyter **Interrupt** / Ctrl+C
        responsive again. If ``threadpoolctl`` is not installed, this flag has no effect.

    Returns
    -------
    pd.DataFrame
        Per-fold metrics with columns: fold, qwk, rmse, mae.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y = np.asarray(target_values)
    strat_bins = stratification_bins if stratification_bins is not None else y
    n_samples = int(y.shape[0])

    if isinstance(feature_matrix, pd.Series):
        n_x = int(feature_matrix.shape[0])
    elif isinstance(feature_matrix, pd.DataFrame):
        n_x = int(feature_matrix.shape[0])
    else:
        n_x = int(np.asarray(feature_matrix).shape[0])

    if n_x != n_samples:
        raise ValueError(
            f"feature_matrix has {n_x} rows but target_values has {n_samples}."
        )

    _validate_raw_text_features_for_tfidf_pipeline(model, feature_matrix)

    # Stratify on labels only; do not use feature dtype for splitting.
    split_placeholder = np.zeros(n_samples)
    fold_rows: List[Dict[str, Any]] = []

    for fold_number, (train_idx, valid_idx) in enumerate(
        cv.split(split_placeholder, strat_bins), start=1
    ):
        X_train = _materialize_fold_features(feature_matrix, train_idx)
        X_valid = _materialize_fold_features(feature_matrix, valid_idx)
        y_train, y_valid = y[train_idx], y[valid_idx]

        fold_model = clone(model)

        if verbose:
            print(
                f"  CV fold {fold_number}/{n_splits}: train n={len(train_idx)} "
                f"→ fitting...",
                flush=True,
            )

        # Single-thread BLAS during fit reduces oversubscription and helps the IPython kernel
        # process SIGINT/Interrupt between native calls (optional; requires threadpoolctl).
        _blas_limit_ctx = (
            threadpool_limits(limits=1)
            if (limit_blas_threads and threadpool_limits is not None)
            else nullcontext()
        )

        # LightGBM + TfidfVectorizer Pipeline quirk: LGBMRegressor records sparse-
        # matrix column indices as "feature names" during fit(), then warns on
        # predict() when the validation sparse matrix lacks the same metadata.
        # This is cosmetic (no data leakage, no accuracy impact), so we filter
        # only this specific message.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
                category=UserWarning,
            )

            with _blas_limit_ctx:
                if use_sample_weights:
                    weights = compute_balanced_sample_weights(y_train)
                    weight_param = _get_sample_weight_param_name(fold_model)
                    fold_model.fit(X_train, y_train, **{weight_param: weights})
                else:
                    fold_model.fit(X_train, y_train)

            y_pred = fold_model.predict(X_valid)

        metrics = evaluate_fold(y_valid, y_pred, min_score, max_score)
        metrics["fold"] = fold_number
        fold_rows.append(metrics)
        if verbose:
            print(
                f"  CV fold {fold_number}/{n_splits}: "
                f"QWK={metrics['qwk']:.4f} RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f}",
                flush=True,
            )
        if after_each_fold is not None:
            after_each_fold(
                fold_number,
                {"qwk": metrics["qwk"], "rmse": metrics["rmse"], "mae": metrics["mae"]},
            )

    return pd.DataFrame(fold_rows)[["fold", "qwk", "rmse", "mae"]]


def _mean_fold_metrics_line(fold_metrics: pd.DataFrame) -> str:
    """
    Build one log line with mean QWK, RMSE, and MAE across CV folds.

    Parameters
    ----------
    fold_metrics : pd.DataFrame
        Columns must include qwk, rmse, mae (one row per fold).

    Returns
    -------
    str
        Human-readable summary for ``print`` during long CV runs.
    """
    means = fold_metrics[["qwk", "rmse", "mae"]].mean()
    return (
        f"mean QWK={means['qwk']:.4f}, RMSE={means['rmse']:.4f}, MAE={means['mae']:.4f}"
    )


def run_registry_experiments_cv(
    experiments: List[Dict[str, Any]],
    run_experiment: Sequence[bool] | np.ndarray,
    target_values: np.ndarray,
    feature_data_by_source: Mapping[str, Any],
    stratification_bins: np.ndarray | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    use_sample_weights: bool = True,
) -> pd.DataFrame:
    """
    Run stratified CV only for registry rows marked True in ``run_experiment``.

    TF-IDF experiments must use raw text under ``feature_data_by_source["tfidf"]``;
    the vectorizer is never precomputed globally — it fits inside each fold via Pipeline.

    Parameters
    ----------
    experiments : List[Dict[str, Any]]
        Registry from ``build_experiments_from_grid`` (name, feature_source, pipeline).
    run_experiment : Sequence[bool] | np.ndarray
        Same length as ``experiments``; True = run this index now.
    target_values : np.ndarray
        Ordinal target labels.
    feature_data_by_source : Mapping[str, Any]
        Keys: handcrafted, tfidf, embeddings, random_uniform — tfidf value must be raw text (Series), not TF-IDF matrix; random_uniform only needs row-aligned columns (ignored by the model).
    stratification_bins : np.ndarray | None
        Binned labels for fold assignment.
    n_splits : int
        Number of stratified folds.
    random_state : int
        Seed for fold splitting.
    use_sample_weights : bool
        Whether to pass balanced sample weights.

    Returns
    -------
    pd.DataFrame
        Per-fold rows for **selected** experiments only: experiment, fold, qwk, rmse, mae.
    """
    run_flags = np.asarray(run_experiment, dtype=bool)
    if run_flags.shape != (len(experiments),):
        raise ValueError(
            f"run_experiment length {run_flags.shape[0]} != len(experiments) {len(experiments)}."
        )
    if not np.any(run_flags):
        raise ValueError(
            "run_experiment is all False; set at least one True to run CV."
        )

    all_fold_results: List[pd.DataFrame] = []

    for run_flag, experiment in zip(run_flags, experiments):
        if not run_flag:
            continue

        experiment_name = experiment["name"]
        feature_source = experiment["feature_source"]
        pipeline = experiment["pipeline"]

        if feature_source not in feature_data_by_source:
            raise KeyError(
                f"feature_data_by_source missing key {feature_source!r}. "
                f"Keys present: {list(feature_data_by_source.keys())}"
            )
        feature_input = feature_data_by_source[feature_source]

        print(f"  Running: {experiment_name} ...", end=" ", flush=True)
        fold_df = run_stratified_cv(
            model=pipeline,
            feature_matrix=feature_input,
            target_values=target_values,
            stratification_bins=stratification_bins,
            n_splits=n_splits,
            random_state=random_state,
            use_sample_weights=use_sample_weights,
        )
        fold_df.insert(0, "experiment", experiment_name)
        all_fold_results.append(fold_df)

        print(_mean_fold_metrics_line(fold_df))

    return pd.concat(all_fold_results, ignore_index=True)


def run_all_experiments_cv(
    experiments: List[Dict[str, Any]],
    target_values: np.ndarray,
    feature_data_by_source: Mapping[str, Any],
    stratification_bins: np.ndarray | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    use_sample_weights: bool = True,
) -> pd.DataFrame:
    """
    Run stratified CV for every row in the experiment registry.

    Thin wrapper: sets ``run_experiment`` to all True and calls
    ``run_registry_experiments_cv`` (project plan orchestrator name).

    Parameters
    ----------
    experiments : List[Dict[str, Any]]
        Registry entries with name, feature_source, pipeline.
    target_values : np.ndarray
        Ordinal target labels.
    feature_data_by_source : Mapping[str, Any]
        Features or raw text per feature_source key.
    stratification_bins : np.ndarray | None
        Binned labels for stratified fold assignment.
    n_splits : int
        Number of folds.
    random_state : int
        RNG seed for fold splits.
    use_sample_weights : bool
        If True, pass balanced sample weights to the final regressor step.

    Returns
    -------
    pd.DataFrame
        Columns: experiment, fold, qwk, rmse, mae.
    """
    run_every_row = np.ones(len(experiments), dtype=bool)
    return run_registry_experiments_cv(
        experiments=experiments,
        run_experiment=run_every_row,
        target_values=target_values,
        feature_data_by_source=feature_data_by_source,
        stratification_bins=stratification_bins,
        n_splits=n_splits,
        random_state=random_state,
        use_sample_weights=use_sample_weights,
    )


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


def build_cv_leaderboard(all_fold_results: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-fold CV results into a ranked leaderboard.

    Parameters
    ----------
    all_fold_results : pd.DataFrame
        Output of run_registry_experiments_cv (experiment, fold, qwk, rmse, mae).

    Returns
    -------
    pd.DataFrame
        One row per experiment with mean/std for each metric,
        sorted by qwk_mean descending then mae_mean ascending.
    """
    grouped = all_fold_results.groupby("experiment")

    leaderboard = pd.DataFrame(
        {
            "qwk_mean": grouped["qwk"].mean(),
            "qwk_std": grouped["qwk"].std(),
            "rmse_mean": grouped["rmse"].mean(),
            "rmse_std": grouped["rmse"].std(),
            "mae_mean": grouped["mae"].mean(),
            "mae_std": grouped["mae"].std(),
        }
    )

    leaderboard = leaderboard.sort_values(
        by=["qwk_mean", "mae_mean"],
        ascending=[False, True],
    ).reset_index()

    return leaderboard


def style_cv_leaderboard(leaderboard: pd.DataFrame) -> Any:
    """
    Format CV leaderboard numeric columns for notebook display (pandas Styler).

    Parameters
    ----------
    leaderboard : pd.DataFrame
        Output of ``build_cv_leaderboard`` (includes an ``experiment`` column).

    Returns
    -------
    Any
        ``pandas.io.formats.style.Styler`` with fixed decimal formatting.
    """
    numeric_columns = [
        column_name
        for column_name in leaderboard.columns
        if column_name != "experiment"
    ]
    format_map = {column_name: "{:.4f}" for column_name in numeric_columns}
    return leaderboard.style.format(format_map, na_rep="—")


# ---------------------------------------------------------------------------
# CV visualization
# ---------------------------------------------------------------------------


def plot_cv_metric_bars(
    leaderboard: pd.DataFrame,
    metric_mean_col: str,
    metric_std_col: str,
    title: str,
    ylabel: str,
    figsize: Tuple[int, int] = (12, 5),
    higher_is_better: bool = True,
) -> plt.Figure:
    """
    Bar plot of one metric across experiments with std error bars.

    Parameters
    ----------
    leaderboard : pd.DataFrame
        Output of build_cv_leaderboard.
    metric_mean_col : str
        Column name for metric mean values.
    metric_std_col : str
        Column name for metric std values.
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    figsize : Tuple[int, int]
        Figure size.
    higher_is_better : bool
        If True sort descending, otherwise ascending.

    Returns
    -------
    plt.Figure
        The figure object.
    """
    sorted_df = leaderboard.sort_values(
        metric_mean_col,
        ascending=not higher_is_better,
    )

    fig, ax = plt.subplots(figsize=figsize)
    x_positions = range(len(sorted_df))
    ax.bar(
        x_positions,
        sorted_df[metric_mean_col],
        yerr=sorted_df[metric_std_col],
        capsize=4,
        edgecolor="black",
        alpha=0.8,
    )
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(sorted_df["experiment"], rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_cv_fold_boxplot(
    all_fold_results: pd.DataFrame,
    metric: str = "qwk",
    title: str = "Fold-wise QWK distribution by experiment",
    ylabel: str = "QWK",
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Boxplot of per-fold metric values across experiments.

    Parameters
    ----------
    all_fold_results : pd.DataFrame
        Output of run_registry_experiments_cv.
    metric : str
        Metric column to plot.
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    figsize : Tuple[int, int]
        Figure size.

    Returns
    -------
    plt.Figure
        The figure object.
    """
    # Order experiments by median metric (best first).
    experiment_order = (
        all_fold_results.groupby("experiment")[metric]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=figsize)
    box_data = [
        all_fold_results.loc[all_fold_results["experiment"] == exp, metric].values
        for exp in experiment_order
    ]
    ax.boxplot(box_data, labels=experiment_order, vert=True)
    ax.set_xticklabels(experiment_order, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Hyperparameter tuning: generic Optuna + stratified CV; TF-IDF+SVD+MLP wrappers
# ---------------------------------------------------------------------------


def _require_optuna() -> None:
    """
    Raise ImportError if Optuna is not installed.

    One-line purpose
        Guard Optuna-only entry points when the optional dependency is missing.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    if optuna is None:
        raise ImportError(
            "optuna is required for hyperparameter tuning. Install with: pip install optuna"
        )


def _apply_blas_thread_env_defaults() -> None:
    """
    Set common BLAS/OpenMP thread env vars to 1 when unset.

    One-line purpose
        Reduce many-core oversubscription so notebook interrupts and Optuna behave better.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Side effect: may set ``OMP_NUM_THREADS``, ``OPENBLAS_NUM_THREADS``, etc.
    """
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(key, "1")


def _optuna_study_to_trials_dataframe(study: Any) -> pd.DataFrame:
    """
    Flatten an Optuna ``Study`` into a pandas table (params + mean QWK/RMSE/MAE user attrs).

    One-line purpose
        Shared export format for any study created by ``optuna_optimize_with_stratified_cv``.

    Parameters
    ----------
    study : optuna.study.Study
        Completed or partially completed study.

    Returns
    -------
    pd.DataFrame
        One row per trial: ``number``, ``state``, ``qwk_mean``, ``rmse_mean``, ``mae_mean``,
        ``value``, plus flattened ``trial.params``.
    """
    rows: List[Dict[str, Any]] = []
    for t in study.trials:
        row: Dict[str, Any] = {
            "number": t.number,
            "state": str(t.state),
            "qwk_mean": t.user_attrs.get("qwk_mean"),
            "rmse_mean": t.user_attrs.get("rmse_mean"),
            "mae_mean": t.user_attrs.get("mae_mean"),
            "value": t.value,
        }
        row.update(t.params)
        rows.append(row)
    return pd.DataFrame(rows)


def optuna_optimize_with_stratified_cv(
    build_estimator: Callable[[Any], Any],
    raw_text_series: pd.Series,
    target_values: np.ndarray,
    stratification_bins: np.ndarray | None,
    *,
    n_trials: int = 30,
    n_splits: int = 5,
    random_state: int = 42,
    use_sample_weights: bool = True,
    optuna_seed: int = 42,
    min_score: int = 0,
    max_score: int = 5,
    use_pruner: bool = True,
    verbose_cv: bool = False,
    limit_blas_threads: bool = True,
    show_progress_bar: bool = True,
    objective_subsample_n: Optional[int] = None,
) -> Tuple[Any, pd.DataFrame]:
    """
    Generic Bayesian (TPE) loop: each trial builds an estimator, then ``run_stratified_cv``.

    The objective is **maximize mean validation QWK** (same metric stack as Section 4).
    ``build_estimator(trial)`` must return an **unfitted** sklearn estimator or Pipeline
    (e.g. suggest hyperparameters via ``trial.suggest_*`` and construct the model).

    One-line purpose
        Reusable Optuna driver for any model that fits the project's stratified CV protocol.

    Parameters
    ----------
    build_estimator : Callable[[optuna.trial.Trial], Any]
        Maps a trial to an unfitted estimator. Use ``trial.suggest_*`` inside for search space.
    raw_text_series : pd.Series
        Row-aligned cleaned text when the model is a TF-IDF pipeline; otherwise aligned features.
    target_values : np.ndarray
        Ordinal targets for evaluation.
    stratification_bins : np.ndarray | None
        Stratification labels for ``StratifiedKFold`` (optional bin merging).
    n_trials : int
        Number of Optuna trials.
    n_splits : int
        Number of CV folds.
    random_state : int
        Seed for CV splits (and any step inside ``build_estimator`` that uses it).
    use_sample_weights : bool
        If True, balanced sample weights on each train fold.
    optuna_seed : int
        Seed for the TPE sampler.
    min_score : int
        Lower clip for QWK rounding in ``evaluate_fold``.
    max_score : int
        Upper clip for QWK rounding in ``evaluate_fold``.
    use_pruner : bool
        If True, ``MedianPruner`` with per-fold reporting of **QWK** (matches ``direction="maximize"``:
        higher fold QWK is better; do **not** negate—negated values invert pruning vs. the study).
    verbose_cv : bool
        If True, print Optuna trial headers and per-fold CV lines (see ``run_stratified_cv``).
    limit_blas_threads : bool
        If True, set single-thread BLAS env defaults (if unset) and pass
        ``limit_blas_threads=True`` into ``run_stratified_cv`` so ``threadpoolctl`` can cap BLAS
        during each ``fit`` — improves Jupyter interrupt handling on many-core CPUs.
    show_progress_bar : bool
        Passed to ``study.optimize`` (Optuna tqdm bar advances once per **completed trial**).
    objective_subsample_n : int, optional
        If set to a positive integer **below** the number of rows, the study optimizes QWK on a
        **stratified random subset** of that size (same protocol, cheaper TF-IDF + MLP per trial).
        Use for hyperparameter **screening**; validate the winner on full data separately.

    Returns
    -------
    Tuple[Any, pd.DataFrame]
        ``(study, trials_table)`` with mean QWK/RMSE/MAE per completed trial.
    """
    _require_optuna()
    if limit_blas_threads:
        _apply_blas_thread_env_defaults()

    # Optional subsample: dominates wall time for text pipelines (vectorizer × folds × trials).
    n_full = int(len(raw_text_series))
    if objective_subsample_n is not None and objective_subsample_n > 0:
        if objective_subsample_n < n_full:
            raw_text_series, target_values, stratification_bins = (
                stratified_subsample_for_optuna_objective(
                    raw_text_series=raw_text_series,
                    target_values=target_values,
                    stratification_bins=stratification_bins,
                    n_samples=objective_subsample_n,
                    random_state=random_state,
                )
            )
            if verbose_cv:
                print(
                    f"Optuna objective uses stratified subsample: "
                    f"n={len(raw_text_series)} of {n_full} rows (faster trials).",
                    flush=True,
                )
        elif verbose_cv:
            print(
                f"Optuna objective uses full data (n={n_full}); "
                f"objective_subsample_n={objective_subsample_n} >= n.",
                flush=True,
            )

    def objective(trial: Any) -> float:
        if verbose_cv:
            print(
                f"\nOptuna trial {trial.number + 1}/{n_trials}: "
                f"{n_splits}-fold CV (mean QWK objective)...",
                flush=True,
            )
        model = build_estimator(trial)

        def after_fold(fold_number: int, metrics: Dict[str, float]) -> None:
            # Report raw QWK so pruning sees the same "higher is better" scale as the study
            # (direction="maximize"). Reporting -QWK inverted MedianPruner vs. true QWK.
            if use_pruner and optuna is not None:
                trial.report(metrics["qwk"], step=fold_number - 1)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        fold_df = run_stratified_cv(
            model=model,
            feature_matrix=raw_text_series,
            target_values=target_values,
            stratification_bins=stratification_bins,
            n_splits=n_splits,
            random_state=random_state,
            use_sample_weights=use_sample_weights,
            min_score=min_score,
            max_score=max_score,
            after_each_fold=after_fold if use_pruner else None,
            verbose=verbose_cv,
            limit_blas_threads=limit_blas_threads,
        )

        qwk_mean = float(fold_df["qwk"].mean())
        trial.set_user_attr("rmse_mean", float(fold_df["rmse"].mean()))
        trial.set_user_attr("mae_mean", float(fold_df["mae"].mean()))
        trial.set_user_attr("qwk_mean", qwk_mean)
        return qwk_mean

    sampler = TPESampler(seed=optuna_seed) if TPESampler is not None else None
    pruner = MedianPruner(n_startup_trials=5) if (use_pruner and MedianPruner) else None

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    study.set_user_attr("objective_n_rows", int(len(raw_text_series)))
    study.set_user_attr("objective_subsample_n_requested", objective_subsample_n)
    study.set_user_attr("objective_full_corpus_n_rows", n_full)
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=show_progress_bar,
        gc_after_trial=True,
    )

    return study, _optuna_study_to_trials_dataframe(study)


# Optuna categorical params must be JSON-serializable (no tuple choices); use string labels.
_MLP_HIDDEN_LAYER_SIZE_LABELS: Tuple[str, ...] = ("128", "256", "512", "256x128")
_MLP_HIDDEN_LAYER_SIZE_LABEL_TO_TUPLE: Dict[str, Tuple[int, ...]] = {
    "128": (128,),
    "256": (256,),
    "512": (512,),
    "256x128": (256, 128),
}


def parse_optuna_mlp_hidden_layer_sizes(value: Any) -> Tuple[int, ...]:
    """
    Convert Optuna's stored ``hidden_layer_sizes`` (str label or legacy tuple) to a sklearn tuple.

    One-line purpose
        Rebuild ``MLPRegressor(hidden_layer_sizes=...)`` from ``study.best_trial.params``.

    Parameters
    ----------
    value : Any
        String label from Optuna (e.g. ``\"256x128\"``) or a legacy tuple from old studies.

    Returns
    -------
    Tuple[int, ...]
        Layer widths for ``MLPRegressor``.
    """
    if isinstance(value, tuple):
        return value
    if isinstance(value, str) and value in _MLP_HIDDEN_LAYER_SIZE_LABEL_TO_TUPLE:
        return _MLP_HIDDEN_LAYER_SIZE_LABEL_TO_TUPLE[value]
    raise ValueError(
        f"Unknown hidden_layer_sizes value: {value!r}. "
        f"Expected one of {list(_MLP_HIDDEN_LAYER_SIZE_LABEL_TO_TUPLE)} or a tuple."
    )


def _mlp_params_from_optuna_trial(
    trial: Any,
    *,
    random_state: int,
    max_iter_bounds: Tuple[int, int] = OPTUNA_MLP_MAX_ITER_BOUNDS,
) -> Dict[str, Any]:
    """
    Sample ``MLPRegressor`` kwargs from one Optuna trial (TPE-friendly search space).

    One-line purpose
        Map ``trial.suggest_*`` calls to a flat dict for ``MLPRegressor(**kwargs)``.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Active Optuna trial.
    random_state : int
        Seed for ``MLPRegressor(random_state=...)``.
    max_iter_bounds : Tuple[int, int]
        Inclusive ``(low, high)`` for ``trial.suggest_int`` on ``max_iter`` (default
        ``OPTUNA_MLP_MAX_ITER_BOUNDS``). With ``early_stopping=True``, this bounds wall time
        per fold; increase if training stops before convergence.

    Returns
    -------
    Dict[str, Any]
        Keyword arguments for ``MLPRegressor``.
    """
    low, high = max_iter_bounds
    if low > high or low < 1:
        raise ValueError(f"Invalid max_iter_bounds: {max_iter_bounds!r}")

    # Strings only: Optuna warns if categorical choices are tuples (non-JSON persistent storage).
    hls_label = trial.suggest_categorical(
        "hidden_layer_sizes",
        list(_MLP_HIDDEN_LAYER_SIZE_LABELS),
    )
    return {
        "hidden_layer_sizes": _MLP_HIDDEN_LAYER_SIZE_LABEL_TO_TUPLE[hls_label],
        "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
        "learning_rate_init": trial.suggest_float(
            "learning_rate_init", 1e-4, 5e-2, log=True
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size", [32, 64, 128, 256]
        ),
        "max_iter": trial.suggest_int("max_iter", low, high),
        "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        "random_state": random_state,
    }


def optuna_tune_tfidf_mlp(
    raw_text_series: pd.Series,
    target_values: np.ndarray,
    stratification_bins: np.ndarray | None,
    vectorizer_kwargs: Mapping[str, Any],
    *,
    svd_n_components: int = 300,
    mlp_fixed_kwargs: Mapping[str, Any] | None = None,
    n_trials: int = 30,
    n_splits: int = 5,
    random_state: int = 42,
    use_sample_weights: bool = True,
    optuna_seed: int = 42,
    min_score: int = 0,
    max_score: int = 5,
    use_pruner: bool = True,
    mlp_max_iter_bounds: Tuple[int, int] = OPTUNA_MLP_MAX_ITER_BOUNDS,
    verbose_cv: bool = False,
    limit_blas_threads: bool = True,
    show_progress_bar: bool = True,
    objective_subsample_n: Optional[int] = 4000,
) -> Tuple[Any, pd.DataFrame]:
    """
    Bayesian (TPE) search for ``TfidfVectorizer`` → ``TruncatedSVD`` → ``MLPRegressor``.

    Thin wrapper around ``optuna_optimize_with_stratified_cv``: only defines ``build_estimator``
    (MLP search space + fixed SVD width). Same CV and metrics as Section 4.

    One-line purpose
        Phase A: tune MLP with fixed ``svd_n_components`` using the shared Optuna CV driver.

    Parameters
    ----------
    raw_text_series : pd.Series
        Row-aligned cleaned text (Section 4 ``tfidf`` feature input).
    target_values : np.ndarray
        Ordinal target vector.
    stratification_bins : np.ndarray | None
        Bins for ``StratifiedKFold`` (e.g. ``build_stratification_bins``).
    vectorizer_kwargs : Mapping[str, Any]
        ``TfidfVectorizer`` kwargs — use the **same** ``tfidf`` settings as Section 4 (e.g.
        ``DEFAULT_TFIDF_VECTORIZER_KWARGS``); this function does not alter them.
    svd_n_components : int
        Fixed SVD width for Phase A (MLP-only tuning).
    mlp_fixed_kwargs : Mapping[str, Any] | None
        Extra MLP kwargs merged before trial params (trial overrides on conflict).
    n_trials : int
        Optuna trial budget.
    n_splits : int
        CV folds.
    random_state : int
        CV + MLP/SVD reproducibility seed.
    use_sample_weights : bool
        Balanced weights on train folds (match Section 4).
    optuna_seed : int
        TPE sampler seed.
    min_score : int
        QWK clip low.
    max_score : int
        QWK clip high.
    use_pruner : bool
        If True, ``MedianPruner`` with per-fold QWK (same maximize semantics as the study).
    mlp_max_iter_bounds : Tuple[int, int]
        Inclusive bounds for Optuna ``max_iter`` (default ``OPTUNA_MLP_MAX_ITER_BOUNDS``).
    verbose_cv : bool
        If True, print each CV fold (passed through to the generic Optuna driver).
    limit_blas_threads : bool
        If True, cap BLAS threads during ``fit`` (better Jupyter interrupt behavior).
    show_progress_bar : bool
        Optuna tqdm bar (one tick per completed trial).
    objective_subsample_n : int, optional
        Stratified row count for the **Optuna objective only** (default ``4000``). Set ``None``
        to use the full corpus (slow). Subsample mean QWK ranks configs; it is not comparable
        to full-data Section 4 scores.

    Returns
    -------
    Tuple[Any, pd.DataFrame]
        ``(optuna.Study, trials_table)`` with columns including ``qwk_mean``, ``rmse_mean``,
        ``mae_mean``, and suggested hyperparameters.
    """
    mlp_baseline: Dict[str, Any] = {
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 15,
    }
    mlp_baseline.update(dict(mlp_fixed_kwargs or {}))

    def build_estimator(trial: Any) -> Pipeline:
        suggested = _mlp_params_from_optuna_trial(
            trial,
            random_state=random_state,
            max_iter_bounds=mlp_max_iter_bounds,
        )
        mlp_kwargs = {**mlp_baseline, **suggested}
        return make_tfidf_mlp_pipeline(
            vectorizer_kwargs=vectorizer_kwargs,
            n_svd_components=svd_n_components,
            mlp_estimator=MLPRegressor(**mlp_kwargs),
            random_state=random_state,
        )

    return optuna_optimize_with_stratified_cv(
        build_estimator=build_estimator,
        raw_text_series=raw_text_series,
        target_values=target_values,
        stratification_bins=stratification_bins,
        n_trials=n_trials,
        n_splits=n_splits,
        random_state=random_state,
        use_sample_weights=use_sample_weights,
        optuna_seed=optuna_seed,
        min_score=min_score,
        max_score=max_score,
        use_pruner=use_pruner,
        verbose_cv=verbose_cv,
        limit_blas_threads=limit_blas_threads,
        show_progress_bar=show_progress_bar,
        objective_subsample_n=objective_subsample_n,
    )


def optuna_tune_tfidf_mlp_svd_n_components(
    raw_text_series: pd.Series,
    target_values: np.ndarray,
    stratification_bins: np.ndarray | None,
    vectorizer_kwargs: Mapping[str, Any],
    mlp_estimator: Any,
    *,
    svd_n_components_min: int = 128,
    svd_n_components_max: int = 512,
    n_trials: int = 20,
    n_splits: int = 5,
    random_state: int = 42,
    use_sample_weights: bool = True,
    optuna_seed: int = 42,
    min_score: int = 0,
    max_score: int = 5,
    use_pruner: bool = True,
    verbose_cv: bool = False,
    limit_blas_threads: bool = True,
    show_progress_bar: bool = True,
    objective_subsample_n: Optional[int] = 4000,
) -> Tuple[Any, pd.DataFrame]:
    """
    Bayesian search over ``TruncatedSVD(n_components)`` with a **fixed** ``MLPRegressor``.

    Delegates to ``optuna_optimize_with_stratified_cv``; each trial suggests an integer
    ``svd_n_components`` in ``[svd_n_components_min, svd_n_components_max]`` (inclusive).
    Cap ``svd_n_components_max`` to at most ``max_features`` from the vectorizer and rank
    limits in your data if you hit sklearn errors.

    One-line purpose
        Phase B: TPE on SVD width while the MLP is frozen (e.g. best from Phase A).

    Parameters
    ----------
    raw_text_series : pd.Series
        Cleaned text column for the TF-IDF pipeline.
    target_values : np.ndarray
        Ordinal targets.
    stratification_bins : np.ndarray | None
        CV stratification bins.
    vectorizer_kwargs : Mapping[str, Any]
        ``TfidfVectorizer`` kwargs — same as Section 4; not modified here.
    mlp_estimator : Any
        Unfitted ``MLPRegressor`` (cloned each trial).
    svd_n_components_min : int
        Lower bound for ``trial.suggest_int`` (inclusive).
    svd_n_components_max : int
        Upper bound for ``trial.suggest_int`` (inclusive).
    n_trials : int
        Optuna trial budget (typically smaller than MLP phase — 1-D search).
    n_splits : int
        CV folds.
    random_state : int
        CV / SVD random state.
    use_sample_weights : bool
        Balanced train-fold weights.
    optuna_seed : int
        TPE sampler seed.
    min_score : int
        QWK clip low.
    max_score : int
        QWK clip high.
    use_pruner : bool
        Whether to use the same MedianPruner as the generic driver.
    verbose_cv : bool
        If True, print each CV fold during tuning.
    limit_blas_threads : bool
        If True, cap BLAS threads during ``fit`` (interrupt-friendlier in notebooks).
    show_progress_bar : bool
        Optuna tqdm progress (one update per completed trial).
    objective_subsample_n : int, optional
        Same as ``optuna_tune_tfidf_mlp`` (stratified subsample for speed; ``None`` = full data).

    Returns
    -------
    Tuple[Any, pd.DataFrame]
        ``(study, trials_table)`` like other tuning helpers.
    """

    def build_estimator(trial: Any) -> Pipeline:
        n_svd = trial.suggest_int(
            "svd_n_components",
            svd_n_components_min,
            svd_n_components_max,
        )
        return make_tfidf_mlp_pipeline(
            vectorizer_kwargs=vectorizer_kwargs,
            n_svd_components=int(n_svd),
            mlp_estimator=clone(mlp_estimator),
            random_state=random_state,
        )

    return optuna_optimize_with_stratified_cv(
        build_estimator=build_estimator,
        raw_text_series=raw_text_series,
        target_values=target_values,
        stratification_bins=stratification_bins,
        n_trials=n_trials,
        n_splits=n_splits,
        random_state=random_state,
        use_sample_weights=use_sample_weights,
        optuna_seed=optuna_seed,
        min_score=min_score,
        max_score=max_score,
        use_pruner=use_pruner,
        verbose_cv=verbose_cv,
        limit_blas_threads=limit_blas_threads,
        show_progress_bar=show_progress_bar,
        objective_subsample_n=objective_subsample_n,
    )


def rank_optuna_trials_like_cv_leaderboard(trials_table: pd.DataFrame) -> pd.DataFrame:
    """
    Sort trials like ``build_cv_leaderboard``: highest ``qwk_mean``, then lower ``mae_mean``.

    One-line purpose
        Pick the best trial with the same tie-break as Section 4.

    Parameters
    ----------
    trials_table : pd.DataFrame
        Table from ``optuna_optimize_with_stratified_cv`` or any TF-IDF tuning wrapper.

    Returns
    -------
    pd.DataFrame
        Rows with finite ``qwk_mean``, sorted for winner inspection.
    """
    valid = trials_table[np.isfinite(trials_table["qwk_mean"])].copy()
    if valid.empty:
        return valid
    return valid.sort_values(
        by=["qwk_mean", "mae_mean"],
        ascending=[False, True],
    ).reset_index(drop=True)


def sweep_svd_n_components_for_tfidf_mlp(
    raw_text_series: pd.Series,
    target_values: np.ndarray,
    stratification_bins: np.ndarray | None,
    vectorizer_kwargs: Mapping[str, Any],
    n_components_list: Sequence[int],
    mlp_estimator: Any,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    use_sample_weights: bool = True,
    min_score: int = 0,
    max_score: int = 5,
) -> pd.DataFrame:
    """
    Grid over ``TruncatedSVD(n_components)`` with a fixed MLP (Phase B after Optuna).

    Uses ``make_tfidf_mlp_pipeline`` and ``run_stratified_cv`` only — no changes to ``utils.py``.

    One-line purpose
        Compare mean CV metrics vs SVD width for a chosen ``MLPRegressor``.

    Parameters
    ----------
    raw_text_series : pd.Series
        Cleaned text aligned with targets.
    target_values : np.ndarray
        Ordinal labels.
    stratification_bins : np.ndarray | None
        CV stratification bins.
    vectorizer_kwargs : Mapping[str, Any]
        ``TfidfVectorizer`` kwargs.
    n_components_list : Sequence[int]
        SVD ranks to evaluate.
    mlp_estimator : Any
        Unfitted regressor (cloned internally).
    n_splits : int
        CV folds.
    random_state : int
        CV / SVD seed.
    use_sample_weights : bool
        Train-fold balanced weights.
    min_score : int
        QWK lower clip.
    max_score : int
        QWK upper clip.

    Returns
    -------
    pd.DataFrame
        Per ``svd_n_components``: mean/std QWK, RMSE, MAE across folds.
    """
    results: List[Dict[str, Any]] = []
    for n_comp in n_components_list:
        pipe = make_tfidf_mlp_pipeline(
            vectorizer_kwargs=vectorizer_kwargs,
            n_svd_components=int(n_comp),
            mlp_estimator=clone(mlp_estimator),
            random_state=random_state,
        )
        fold_df = run_stratified_cv(
            model=pipe,
            feature_matrix=raw_text_series,
            target_values=target_values,
            stratification_bins=stratification_bins,
            n_splits=n_splits,
            random_state=random_state,
            use_sample_weights=use_sample_weights,
            min_score=min_score,
            max_score=max_score,
        )
        results.append(
            {
                "svd_n_components": int(n_comp),
                "qwk_mean": float(fold_df["qwk"].mean()),
                "qwk_std": float(fold_df["qwk"].std()),
                "rmse_mean": float(fold_df["rmse"].mean()),
                "rmse_std": float(fold_df["rmse"].std()),
                "mae_mean": float(fold_df["mae"].mean()),
                "mae_std": float(fold_df["mae"].std()),
            }
        )
    return pd.DataFrame(results)
