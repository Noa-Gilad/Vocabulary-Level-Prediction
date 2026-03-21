"""
Modeling utilities for vocabulary-level prediction.

Functions for target creation, rater-agreement filtering,
cross-validation evaluation, sample-weight generation, experiment
orchestration (``run_all_experiments_cv`` for the full registry; ``run_registry_experiments_cv`` for row subsets),
leaderboard aggregation, and CV visualization.
Reuses feature-extraction helpers from utils.py where they exist.
"""

from __future__ import annotations

import warnings
from typing import Any, Collection, Dict, List, Mapping, Optional, Sequence, Tuple

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
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


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
        column_name for column_name in required_columns
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
    y_pred_rounded = round_and_clip_predictions(
        y_pred_continuous, min_score, max_score
    )
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

# Tree / boosting short names: on the embedding track we skip StandardScaler (plan).
DEFAULT_EMBEDDING_TREE_MODEL_SHORT_NAMES: frozenset[str] = frozenset({
    "RF",
    "XGB",
    "LGBM",
})


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
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", final_estimator),
    ])


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
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", final_estimator),
    ])


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
    return Pipeline([
        ("tfidf", TfidfVectorizer(**dict(vectorizer_kwargs))),
        ("model", final_estimator),
    ])


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
    return Pipeline([
        ("tfidf", TfidfVectorizer(**dict(vectorizer_kwargs))),
        ("svd", TruncatedSVD(n_components=n_svd_components, random_state=random_state)),
        ("mlp", mlp_estimator),
    ])


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
            final_estimator, vectorizer_kwargs=tfidf_vectorizer_kwargs,
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
            experiments.append({
                "name": f"{name_prefix}_{model_short_name}",
                "feature_source": feature_source,
                "pipeline": pipeline,
            })
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
            experiments.append({
                "name": f"{name_prefix}_{model_short_name}",
                "feature_source": feature_source,
                "pipeline": pipeline,
            })

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

    if series.dtype.kind in ("i", "u", "f", "c") or np.issubdtype(series.dtype, np.number):
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
        raise ValueError("run_experiment is all False; set at least one True to run CV.")

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

    leaderboard = pd.DataFrame({
        "qwk_mean": grouped["qwk"].mean(),
        "qwk_std": grouped["qwk"].std(),
        "rmse_mean": grouped["rmse"].mean(),
        "rmse_std": grouped["rmse"].std(),
        "mae_mean": grouped["mae"].mean(),
        "mae_std": grouped["mae"].std(),
    })

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
        metric_mean_col, ascending=not higher_is_better,
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
        all_fold_results.loc[
            all_fold_results["experiment"] == exp, metric
        ].values
        for exp in experiment_order
    ]
    ax.boxplot(box_data, labels=experiment_order, vert=True)
    ax.set_xticklabels(experiment_order, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig
