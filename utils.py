"""
Vocabulary Level Prediction — shared utilities.
Preprocessing, data loading, and helper functions.
"""

# -----------------------------------------------------------------------------
# Environment setting
# -----------------------------------------------------------------------------
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # Install with: pip install matplotlib

try:
    from sklearn.metrics import cohen_kappa_score
except ImportError:
    cohen_kappa_score = None  # Install with: pip install scikit-learn

try:
    import nltk
    from nltk import pos_tag
except ImportError:
    nltk = None  # Install with: pip install nltk
    pos_tag = None

try:
    import textstat
except ImportError:
    textstat = None  # Install with: pip install textstat

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # Install with: pip install sentence-transformers

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
except ImportError:
    PCA = None
    KMeans = None
    StandardScaler = None

# Default model for sentence embeddings (EDA step 8).
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# Lighter, faster on CPU; slightly lower quality (use when no GPU).
DEFAULT_EMBEDDING_MODEL_FAST = "sentence-transformers/paraphrase-MiniLM-L3-v2"

# Common English stopwords (minimal set) for word-frequency and n-gram analysis when none provided.
DEFAULT_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will",
    "with", "this", "but", "they", "have", "had", "what", "when", "where", "who",
    "which", "why", "how", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "just", "if", "or", "because", "until", "while",
})

# Penn Treebank tag groups for POS ratio computation (EDA 2.6).
_POS_NOUN = frozenset({"NN", "NNS", "NNP", "NNPS"})
_POS_VERB = frozenset({"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"})
_POS_ADJ = frozenset({"JJ", "JJR", "JJS"})
_POS_ADV = frozenset({"RB", "RBR", "RBS"})
_POS_OTHER = None  # all remaining tags

# -----------------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------------


def normalize_line_endings_and_whitespace(text: str) -> str:
    """
    Replace line breaks and tabs with space, collapse multiple spaces, strip edges.

    Parameters
    ----------
    text : str
        Raw input string (may contain \\r\\n, \\r, \\n, \\t).

    Returns
    -------
    str
        String with single spaces only, no leading/trailing spaces.
    """
    if not isinstance(text, str):
        return str(text)
    # Order matters: replace \\r\\n first so we don't leave a space per character
    normalized = (
        text.replace("\r\n", " ")
        .replace("\r", " ")
        .replace("\n", " ")
        .replace("\t", " ")
    )
    # Collapse runs of spaces and strip
    return " ".join(normalized.split())


def lowercase_text(text: str) -> str:
    """
    Lowercase the entire string.

    Parameters
    ----------
    text : str
        Input string.

    Returns
    -------
    str
        Lowercased string.
    """
    if not isinstance(text, str):
        return str(text)
    return text.lower()


def strip_punctuation_keep_sentence_endings(text: str) -> str:
    """
    Remove all punctuation except period, question mark, and exclamation.

    Keeps . ? ! so sentence boundaries remain for sentence count and readability.

    Parameters
    ----------
    text : str
        Input string.

    Returns
    -------
    str
        String with only . ? ! as punctuation.
    """
    if not isinstance(text, str):
        return str(text)
    # Keep only letters, digits, whitespace, and . ? !
    kept = re.sub(r"[^\w\s.?!]", "", text, flags=re.UNICODE)
    # Collapse any double spaces left after removal
    return " ".join(kept.split())


def preprocess_text(text: str) -> str:
    """
    Run full initial preprocessing on a single text: normalize whitespace, lowercase, strip punctuation (keep . ? !).

    Parameters
    ----------
    text : str
        Raw essay or text string.

    Returns
    -------
    str
        Preprocessed string.
    """
    t = normalize_line_endings_and_whitespace(text)
    t = lowercase_text(t)
    t = strip_punctuation_keep_sentence_endings(t)
    return t


def load_data(path: str) -> pd.DataFrame:
    """
    Load vocabulary dataset from CSV.

    Parameters
    ----------
    path : str
        Path to CSV file (e.g. vocabulary_data.csv).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns text_id, Text, Vocabulary_1, Vocabulary_2 (or as in file).
    """
    return pd.read_csv(path)


def apply_preprocessing_to_dataframe(
    df: pd.DataFrame,
    text_column: str,
    output_column: str | None = None,
) -> pd.DataFrame:
    """
    Apply preprocess_text to each row of a text column; optionally write to a new column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the text column.
    text_column : str
        Name of column with raw text.
    output_column : str | None
        If None, overwrites text_column. If str, writes preprocessed text to this new column.

    Returns
    -------
    pd.DataFrame
        DataFrame with preprocessed text in text_column or output_column (copy not inplace).
    """
    out = df.copy()
    target = output_column if output_column is not None else text_column
    out[target] = out[text_column].astype(str).map(preprocess_text)
    return out


# -----------------------------------------------------------------------------
# EDA
# -----------------------------------------------------------------------------


def get_missing_counts(df: pd.DataFrame) -> pd.Series:
    """
    Count missing values per column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.Series
        Number of nulls per column (index = column names).
    """
    return df.isnull().sum()


def get_target_value_counts(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Value counts for a target column (e.g. Vocabulary_1, Vocabulary_2).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the column.
    column : str
        Name of the target column.

    Returns
    -------
    pd.Series
        Value counts, index = unique values, values = counts.
    """
    return df[column].value_counts().sort_index()


def compute_rater_agreement(
    df: pd.DataFrame,
    col1: str,
    col2: str,
) -> dict:
    """
    Compute weighted Cohen's kappa and exact agreement % between two rater columns (ordinal 0-5).

    The score is ordinal (ordered 0 < 1 < ... < 5), not just discrete. Weighted kappa
    (quadratic weights) and Spearman correlation respect this order.

    These metrics measure item-level (joint) agreement: for each row, do the two raters
    give the same or similar score? Similar marginal distributions (e.g. side-by-side bar
    chart) do not imply high agreement: raters can assign the same total number of 3s
    and 4s but to different essays.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the two rating columns.
    col1 : str
        First rater column name.
    col2 : str
        Second rater column name.

    Returns
    -------
    dict
        Keys: 'weighted_kappa', 'exact_agreement_pct', 'spearman_corr'.
    """
    v1 = df[col1].dropna().astype(int)
    v2 = df[col2].dropna().astype(int)
    # Align by index so we only use rows with both present
    common = v1.index.intersection(v2.index)
    v1 = v1.loc[common]
    v2 = v2.loc[common]
    n = len(v1)
    if n == 0:
        return {
            "weighted_kappa": None,
            "exact_agreement_pct": None,
            "spearman_corr": None,
        }
    if cohen_kappa_score is None:
        raise ImportError(
            "compute_rater_agreement requires scikit-learn. Install with: pip install scikit-learn"
        )
    exact = (v1 == v2).sum() / n * 100
    kappa = cohen_kappa_score(v1, v2, weights="quadratic")
    corr = v1.corr(v2, method="spearman")
    return {
        "weighted_kappa": kappa,
        "exact_agreement_pct": exact,
        "spearman_corr": corr,
    }


def plot_target_distributions_side_by_side(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    scores: range | None = None,
    ax=None,
) -> "plt.Axes":
    """
    Bar plot of value counts for col1 and col2, side by side for each score (e.g. 0-5).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the two target columns.
    col1 : str
        First column (e.g. Vocabulary_1); one bar group per score.
    col2 : str
        Second column (e.g. Vocabulary_2); one bar group per score.
    scores : range | None
        Score values to show on x-axis. If None, uses range(6) (0-5).
    ax : plt.Axes | None
        If provided, draw on this axes; otherwise create a new figure.

    Returns
    -------
    plt.Axes
        The axes used for the plot.
    """
    if plt is None:
        raise ImportError(
            "plot_target_distributions_side_by_side requires matplotlib. Install with: pip install matplotlib"
        )
    if scores is None:
        scores = range(6)  # 0-5
    scores = list(scores)
    c1 = df[col1].value_counts().reindex(scores, fill_value=0)
    c2 = df[col2].value_counts().reindex(scores, fill_value=0)
    if ax is None:
        _, ax = plt.subplots()
    width = 0.35
    x = range(len(scores))
    ax.bar([i - width / 2 for i in x], c1.values, width, label=col1)
    ax.bar([i + width / 2 for i in x], c2.values, width, label=col2)
    ax.set_xticks(x)
    ax.set_xticklabels(scores)
    ax.set_xlabel("Score")
    ax.set_ylabel("Number of essays")
    ax.set_title(f"{col1} vs {col2} (side by side)")
    ax.legend()
    return ax


def get_absolute_difference_counts(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    max_diff: int = 5,
) -> pd.Series:
    """
    Count of essays per absolute score difference |col1 - col2| (0 = agreement, 1-5 = disagreement).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the two score columns.
    col1 : str
        First column name.
    col2 : str
        Second column name.
    max_diff : int
        Maximum absolute difference to include (default 5). Differences are 0..max_diff.

    Returns
    -------
    pd.Series
        Counts indexed by 0, 1, ..., max_diff (reindexed so missing values are 0).
    """
    diff = (df[col1].astype(float) - df[col2].astype(float)).abs().dropna().astype(int)
    counts = diff.value_counts().reindex(range(max_diff + 1), fill_value=0).sort_index()
    return counts


def plot_absolute_difference_distribution(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    max_diff: int = 5,
    ax=None,
) -> "plt.Axes":
    """
    Bar plot of count of essays per absolute score difference |col1 - col2| (0-5).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the two score columns.
    col1 : str
        First column name.
    col2 : str
        Second column name.
    max_diff : int
        Maximum absolute difference (default 5). X-axis shows 0..max_diff.
    ax : plt.Axes | None
        If provided, draw on this axes; otherwise create a new figure.

    Returns
    -------
    plt.Axes
        The axes used for the plot.
    """
    if plt is None:
        raise ImportError(
            "plot_absolute_difference_distribution requires matplotlib. Install with: pip install matplotlib"
        )
    counts = get_absolute_difference_counts(df, col1, col2, max_diff=max_diff)
    if ax is None:
        _, ax = plt.subplots()
    ax.bar(counts.index, counts.values)
    ax.set_xlabel("Absolute difference |" + col1 + " − " + col2 + "|")
    ax.set_ylabel("Number of essays")
    ax.set_title("Distribution of absolute score difference (0 = agreement)")
    return ax


# -----------------------------------------------------------------------------
# EDA 2.2 Text length analysis
# -----------------------------------------------------------------------------


def get_char_count(text: str) -> int:
    """
    Character count of a string.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    int
        Number of characters.
    """
    return len(str(text)) if text else 0


def get_word_count(text: str) -> int:
    """
    Word count (split on whitespace).

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    int
        Number of tokens after split.
    """
    if not text or not str(text).strip():
        return 0
    return len(str(text).split())


def get_sentence_count(text: str) -> int:
    """
    Sentence count (split on . ? !).

    Parameters
    ----------
    text : str
        Input text (should keep . ? ! for sentence boundaries).

    Returns
    -------
    int
        Number of non-empty segments after splitting on sentence endings.
    """
    if not text or not str(text).strip():
        return 0
    parts = re.split(r"[.?!]+", str(text))
    return max(1, len([p for p in parts if p.strip()])) if any(p.strip() for p in parts) else 1


def get_avg_word_length(text: str) -> float:
    """
    Average word length in characters (chars / words); 0 if no words.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    float
        Mean character count per word.
    """
    words = str(text).split() if text else []
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def get_text_length_features_df(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Compute char_count, word_count, sentence_count, avg_word_length per row.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a text column.
    text_column : str
        Name of the text column.

    Returns
    -------
    pd.DataFrame
        Four columns: char_count, word_count, sentence_count, avg_word_length (index aligned with df).
    """
    series = df[text_column].astype(str)
    out = pd.DataFrame(index=df.index)
    out["char_count"] = series.map(get_char_count)
    out["word_count"] = series.map(get_word_count)
    out["sentence_count"] = series.map(get_sentence_count)
    out["avg_word_length"] = series.map(get_avg_word_length)
    return out


def get_length_features_summary(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary statistics for length features (mean, std, min, quartiles, max). No count row.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame from get_text_length_features_df.

    Returns
    -------
    pd.DataFrame
        describe() without the count row (mean, std, min, 25%, 50%, 75%, max).
    """
    return features_df.describe().drop("count", errors="ignore")


def plot_length_distributions(
    features_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    figsize: tuple[int, int] = (12, 10),
) -> "plt.Figure":
    """
    Plot histograms for each length feature (2x2 grid).

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame from get_text_length_features_df.
    feature_columns : list[str] | None
        Columns to plot. If None, uses char_count, word_count, sentence_count, avg_word_length.
    figsize : tuple[int, int]
        Figure size.

    Returns
    -------
    plt.Figure
        The figure.
    """
    if plt is None:
        raise ImportError(
            "plot_length_distributions requires matplotlib. Install with: pip install matplotlib"
        )
    if feature_columns is None:
        feature_columns = [
            "char_count",
            "word_count",
            "sentence_count",
            "avg_word_length",
        ]
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    for i, col in enumerate(feature_columns):
        if i >= len(axes):
            break
        axes[i].hist(features_df[col].dropna(), bins=50, edgecolor="black", alpha=0.7)
        axes[i].set_title(col)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Number of essays")
    fig.suptitle("Text length feature distributions", y=1.02)
    plt.tight_layout(pad=1.2)
    return fig


def plot_feature_distribution(
    series: pd.Series,
    ax=None,
    title: str | None = None,
    xlabel: str | None = None,
    bins: int = 50,
) -> "plt.Axes":
    """
    Plot a single histogram for one feature (modular: one feature, one plot).

    Parameters
    ----------
    series : pd.Series
        One column of values to plot.
    ax : plt.Axes | None
        Axes to draw on; if None, creates a new figure and axes.
    title : str | None
        Subplot title; if None, uses series.name.
    xlabel : str | None
        X-axis label; if None, uses series.name.
    bins : int
        Number of histogram bins.

    Returns
    -------
    plt.Axes
    """
    if plt is None:
        raise ImportError("plot_feature_distribution requires matplotlib")
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    label = title if title is not None else (series.name if series.name else "value")
    ax.hist(series.dropna(), bins=bins, edgecolor="black", alpha=0.7)
    ax.set_title(label)
    ax.set_xlabel(xlabel if xlabel is not None else label)
    ax.set_ylabel("Number of essays")
    return ax


def get_length_target_correlations(
    features_df: pd.DataFrame,
    target: pd.Series,
    method: str = "spearman",
) -> pd.Series:
    """
    Correlation of each length feature with a single target (ordinal 0-5 score).

    Uses Spearman by default because the vocabulary score is ordinal (ordered), not
    just discrete: it respects 0 < 1 < ... < 5. Pearson would assume interval scale.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame from get_text_length_features_df.
    target : pd.Series
        Target variable, ordinal 0-5 (same index as features_df).
    method : str
        Correlation method; use 'spearman' for ordinal target (default).

    Returns
    -------
    pd.Series
        Correlation of each feature column with target.
    """
    common = features_df.index.intersection(target.index)
    f = features_df.loc[common]
    t = pd.to_numeric(target.loc[common], errors="coerce").dropna()
    common = common.intersection(t.index)
    if len(common) < 2:
        return pd.Series(dtype=float)
    f = f.loc[common]
    t = t.loc[common]
    return f.corrwith(t, method=method)


def plot_length_target_heatmap(
    features_df: pd.DataFrame,
    target_columns: list[str],
    df: pd.DataFrame,
    method: str = "spearman",
    ax=None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> "plt.Axes":
    """
    Heatmap of correlations between length features and target columns (ordinal 0-5).

    Uses Spearman by default so correlation respects the order of the score (0-5).

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame from get_text_length_features_df.
    target_columns : list[str]
        Target column names (must exist in df).
    df : pd.DataFrame
        Full DataFrame containing target_columns.
    method : str
        Correlation method; use 'spearman' for ordinal targets (default).
    ax : plt.Axes | None
        Axes to use; if None, creates new figure.
    title : str | None
        If set, use as plot title; else default "Length features vs targets (...)".
    figsize : tuple[float, float] | None
        Figure (width, height); if None, computed from number of features and targets.
        Use e.g. (6, 10) for n-gram heatmaps to avoid vertically stretched rows.

    Returns
    -------
    plt.Axes
        The axes used.
    """
    if plt is None:
        raise ImportError(
            "plot_length_target_heatmap requires matplotlib. Install with: pip install matplotlib"
        )
    feature_cols = [c for c in features_df.columns if c in features_df]
    corr_matrix = pd.DataFrame(index=feature_cols)
    for tc in target_columns:
        if tc not in df.columns:
            continue
        corr_matrix[tc] = get_length_target_correlations(features_df, df[tc], method=method)
    if ax is None:
        if figsize is None:
            figsize = (max(5, len(target_columns) * 2.5), max(4, len(feature_cols) * 1.0))
        _, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
    ax.set_xlabel("Target variable (score column)")
    ax.set_yticks(range(len(corr_matrix.index)))
    ax.set_yticklabels(corr_matrix.index)
    ax.set_ylabel("Length feature")
    # Annotate each cell with the correlation value so small differences are readable
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            val = corr_matrix.values[i, j]
            text_color = "white" if abs(val) > 0.5 else "black"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center", color=text_color, fontsize=11,
            )
    plt.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_title(title if title is not None else f"Length features vs targets ({method})")
    plt.tight_layout()
    return ax


# -----------------------------------------------------------------------------
# EDA 2.3 Vocabulary richness (strip . ? ! on demand for word-level metrics)
# -----------------------------------------------------------------------------


def _normalize_text_for_richness(text: str) -> list[str]:
    """
    Strip sentence-ending punctuation (. ? !) and return list of words for richness metrics.

    Used on demand so "word" and "word." count as the same type. Does not modify
    stored text; call this when computing TTR, hapax, unique words.

    Parameters
    ----------
    text : str
        Input text (e.g. Text_cleaned).

    Returns
    -------
    list[str]
        Words after stripping . ? ! and splitting on whitespace.
    """
    if not text or not str(text).strip():
        return []
    s = re.sub(r"[.?!]+", "", str(text))
    return s.split()


def get_unique_word_count(text: str) -> int:
    """
    Number of unique word types (strip . ? ! on demand so word and word. are one type).

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    int
        Number of unique words.
    """
    words = _normalize_text_for_richness(text)
    return len(set(words))


def get_ttr(text: str) -> float:
    """
    Type-token ratio (unique words / total words). 0.0 if no words.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    float
        TTR in [0, 1].
    """
    words = _normalize_text_for_richness(text)
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def get_hapax_legomena_ratio(text: str) -> float:
    """
    Ratio of hapax legomena (words that appear exactly once) to total word types. 0.0 if no words.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    float
        Hapax count / type count in [0, 1].
    """
    words = _normalize_text_for_richness(text)
    if not words:
        return 0.0
    counts = Counter(words)
    types = len(counts)
    hapax = sum(1 for c in counts.values() if c == 1)
    return hapax / types if types else 0.0


def get_vocabulary_richness_df(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Per-essay vocabulary richness: unique_words, ttr, hapax_ratio (strip . ? ! on demand).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a text column.
    text_column : str
        Name of the text column (e.g. Text_cleaned).

    Returns
    -------
    pd.DataFrame
        Columns: unique_words, ttr, hapax_ratio (index aligned with df).
    """
    series = df[text_column].astype(str)
    out = pd.DataFrame(index=df.index)
    out["unique_words"] = series.map(get_unique_word_count)
    out["ttr"] = series.map(get_ttr)
    out["hapax_ratio"] = series.map(get_hapax_legomena_ratio)
    return out


def get_richness_features_summary(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary statistics for richness features (no count row).

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame from get_vocabulary_richness_df.

    Returns
    -------
    pd.DataFrame
        describe() without the count row.
    """
    return features_df.describe().drop("count", errors="ignore")


def plot_richness_distributions(
    features_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    figsize: tuple[int, int] = (12, 4),
) -> "plt.Figure":
    """
    Histograms for each vocabulary richness feature (1 row of 3 subplots by default).

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame from get_vocabulary_richness_df.
    feature_columns : list[str] | None
        Columns to plot. If None, uses unique_words, ttr, hapax_ratio.
    figsize : tuple[int, int]
        Figure size.

    Returns
    -------
    plt.Figure
        The figure.
    """
    if plt is None:
        raise ImportError(
            "plot_richness_distributions requires matplotlib. Install with: pip install matplotlib"
        )
    if feature_columns is None:
        feature_columns = ["unique_words", "ttr", "hapax_ratio"]
    n = len(feature_columns)
    nrows = (n + 2) // 3
    ncols = min(3, n)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for i, col in enumerate(feature_columns):
        if i >= len(axes):
            break
        axes[i].hist(features_df[col].dropna(), bins=50, edgecolor="black", alpha=0.7)
        axes[i].set_title(col)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Number of essays")
    for j in range(len(feature_columns), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Vocabulary richness feature distributions", y=1.02)
    plt.tight_layout(pad=1.2)
    return fig


# -----------------------------------------------------------------------------
# EDA 2.4 Word frequency
# -----------------------------------------------------------------------------


def get_top_n_words_corpus(
    text_series: pd.Series,
    n: int = 30,
    stopwords: frozenset[str] | set[str] | None = None,
) -> pd.Series:
    """
    Top N words (by frequency) across all texts, stopwords removed. Uses normalized words (strip . ? !).

    Parameters
    ----------
    text_series : pd.Series
        One text per row (e.g. df["Text_cleaned"]).
    n : int
        Number of top words to return.
    stopwords : frozenset | set | None
        Words to exclude; if None, uses DEFAULT_STOPWORDS.

    Returns
    -------
    pd.Series
        Word counts, index = word, sorted descending (top n).
    """
    stop = stopwords if stopwords is not None else DEFAULT_STOPWORDS
    counter = Counter()
    for text in text_series.astype(str):
        words = _normalize_text_for_richness(text)
        for w in words:
            if w and w.lower() not in stop:
                counter[w.lower()] += 1
    if not counter:
        return pd.Series(dtype=int)
    return pd.Series(counter).nlargest(n)


def get_top_n_words_per_score_group(
    df: pd.DataFrame,
    text_column: str,
    target_column: str,
    n: int = 30,
    stopwords: frozenset[str] | set[str] | None = None,
    score_values: list[int] | None = None,
) -> dict[int, pd.Series]:
    """
    Top N words per score level (e.g. per Vocabulary 0-5). Stopwords removed.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with text and target columns.
    text_column : str
        Name of the text column.
    target_column : str
        Name of the ordinal score column (e.g. Vocabulary_1).
    n : int
        Number of top words per group.
    stopwords : frozenset | set | None
        Words to exclude; if None, uses DEFAULT_STOPWORDS.
    score_values : list[int] | None
        Score levels to group by; if None, uses sorted unique values in target_column.

    Returns
    -------
    dict[int, pd.Series]
        Map score value -> Series of (word, count) for top n words in that group.
    """
    stop = stopwords if stopwords is not None else DEFAULT_STOPWORDS
    if score_values is None:
        score_values = sorted(df[target_column].dropna().unique().astype(int))
    out = {}
    for score in score_values:
        subset = df[df[target_column] == score][text_column].astype(str)
        counter = Counter()
        for text in subset:
            for w in _normalize_text_for_richness(text):
                if w and w.lower() not in stop:
                    counter[w.lower()] += 1
        out[int(score)] = pd.Series(counter).nlargest(n) if counter else pd.Series(dtype=int)
    return out


# -----------------------------------------------------------------------------
# EDA 2.5 N-gram analysis (sentence-bound only)
# -----------------------------------------------------------------------------


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences on . ? ! so n-grams do not cross sentence boundaries.

    Parameters
    ----------
    text : str
        Input text (e.g. Text_cleaned, which keeps . ? !).

    Returns
    -------
    list[str]
        Non-empty sentence strings.
    """
    if not text or not str(text).strip():
        return []
    parts = re.split(r"[.?!]+\s*", str(text))
    return [s.strip() for s in parts if s.strip()]


def _get_sentence_bound_ngrams_for_text(
    text: str,
    ngram_size: int,
    stopwords: frozenset[str] | set[str] | None = None,
) -> list[str]:
    """
    Extract sentence-bound n-grams (bigrams or trigrams) from one text. No cross-sentence n-grams.

    Parameters
    ----------
    text : str
        One essay/text.
    ngram_size : int
        2 for bigrams, 3 for trigrams.
    stopwords : frozenset | set | None
        Words to exclude from n-grams; if None, uses DEFAULT_STOPWORDS.

    Returns
    -------
    list[str]
        N-gram strings (e.g. "word1 word2") collected from all sentences.
    """
    stop = stopwords if stopwords is not None else DEFAULT_STOPWORDS
    sentences = _split_sentences(text)
    out = []
    for sent in sentences:
        words = _normalize_text_for_richness(sent)
        # Filter stopwords and lowercase for consistency
        words = [w.lower() for w in words if w and w.lower() not in stop]
        if len(words) < ngram_size:
            continue
        if ngram_size == 2:
            ngrams = [" ".join(pair) for pair in zip(words, words[1:])]
        elif ngram_size == 3:
            ngrams = [" ".join(tri) for tri in zip(words, words[1:], words[2:])]
        else:
            ngrams = []
        out.extend(ngrams)
    return out


def get_top_n_ngrams_corpus(
    text_series: pd.Series,
    n: int = 10,
    ngram_size: int = 2,
    stopwords: frozenset[str] | set[str] | None = None,
) -> pd.Series:
    """
    Top N sentence-bound n-grams (bigrams or trigrams) over the corpus. Stopwords removed.

    Parameters
    ----------
    text_series : pd.Series
        One text per row (e.g. df["Text_cleaned"]).
    n : int
        Number of top n-grams to return.
    ngram_size : int
        2 for bigrams, 3 for trigrams.
    stopwords : frozenset | set | None
        Words to exclude; if None, uses DEFAULT_STOPWORDS.

    Returns
    -------
    pd.Series
        N-gram counts, index = n-gram string, sorted descending (top n).
    """
    counter = Counter()
    for text in text_series.astype(str):
        for ng in _get_sentence_bound_ngrams_for_text(text, ngram_size, stopwords):
            counter[ng] += 1
    if not counter:
        return pd.Series(dtype=int)
    return pd.Series(counter).nlargest(n)


def _get_ngram_to_doc_indices(
    df: pd.DataFrame,
    text_column: str,
    ngram_size: int,
    min_doc_count: int,
    stopwords: frozenset[str] | set[str] | None = None,
) -> dict[str, set]:
    """
    One pass over the corpus: for each n-gram, the set of row indices (df.index) where it appears.
    Returns only n-grams that appear in at least min_doc_count documents (no frequency cap).
    """
    ngram_to_docs: dict[str, set] = defaultdict(set)
    for idx in df.index:
        text = df.loc[idx, text_column]
        for ng in _get_sentence_bound_ngrams_for_text(
            str(text) if pd.notna(text) else "", ngram_size, stopwords
        ):
            ngram_to_docs[ng].add(idx)
    return {
        ng: doc_set
        for ng, doc_set in ngram_to_docs.items()
        if len(doc_set) >= min_doc_count
    }


def _spearman_corr_binary_matrix_with_target(
    presence_matrix: np.ndarray,
    target_ranked: np.ndarray,
) -> np.ndarray:
    """
    Vectorized Spearman correlation of each binary column with one ranked target.
    For 0/1 columns, Spearman(binary, y) = point-biserial(rank(y), binary). One rank,
    then one pass over columns. Returns 1D array of length presence_matrix.shape[1].
    """
    n = target_ranked.size
    if n != presence_matrix.shape[0]:
        raise ValueError("target_ranked length must match presence_matrix rows")
    n1 = presence_matrix.sum(axis=0)
    n0 = n - n1
    sum_rank_1 = (target_ranked[:, np.newaxis] * presence_matrix).sum(axis=0)
    sum_rank_0 = target_ranked.sum() - sum_rank_1
    mean_1 = np.where(n1 > 0, sum_rank_1 / n1, 0.0)
    mean_0 = np.where(n0 > 0, sum_rank_0 / n0, 0.0)
    std_rank = np.std(target_ranked, ddof=0)
    if std_rank <= 0:
        return np.zeros(presence_matrix.shape[1])
    r = (mean_1 - mean_0) * np.sqrt(np.where((n1 > 0) & (n0 > 0), n1 * n0 / (n * n), 0.0)) / std_rank
    return np.where((n1 > 0) & (n0 > 0), r, 0.0)


def _build_presence_chunk(
    common_index: pd.Index,
    ngram_to_docs: dict[str, set],
    ngram_list: list[str],
) -> pd.DataFrame:
    """
    Build a presence matrix (0/1) for a chunk of n-grams over the common row index.

    Parameters
    ----------
    common_index : pd.Index
        Row index (subset of df.index) to use.
    ngram_to_docs : dict[str, set]
        Maps each n-gram to the set of row indices where it appears.
    ngram_list : list[str]
        N-grams to include as columns (chunk subset).

    Returns
    -------
    pd.DataFrame
        Index = common_index, columns = ngram_list, values 0.0 or 1.0.
    """
    common_set = set(common_index)
    common_list = list(common_index)
    idx_to_row = {idx: i for i, idx in enumerate(common_list)}
    n_rows = len(common_list)
    n_cols = len(ngram_list)
    data = np.zeros((n_rows, n_cols), dtype=np.float64)
    for j, ng in enumerate(ngram_list):
        doc_set = ngram_to_docs.get(ng)
        if not doc_set:
            continue
        for idx in doc_set & common_set:
            data[idx_to_row[idx], j] = 1.0
    return pd.DataFrame(data, index=common_index, columns=ngram_list)


def get_top_ngrams_by_absolute_correlation(
    df: pd.DataFrame,
    text_column: str,
    target_columns: list[str],
    ngram_size: int,
    top_k: int = 25,
    min_doc_count: int = 2,
    chunk_size: int = 5000,
    stopwords: frozenset[str] | set[str] | None = None,
    method: str = "spearman",
) -> list[str]:
    """
    Top K n-grams by maximum absolute correlation with any target (most discriminative).

    Correlates every n-gram that appears in at least min_doc_count documents (no frequency
    cap). Uses chunked presence matrices and vectorized corrwith for efficiency. Returns
    top_k by |correlation|. Spearman is used by default for ordinal targets.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with text_column and target columns.
    text_column : str
        Name of the text column.
    target_columns : list[str]
        Target column names (e.g. ["Vocabulary_1", "Vocabulary_2"]).
    ngram_size : int
        2 for bigrams, 3 for trigrams.
    top_k : int
        Number of n-grams to return.
    min_doc_count : int
        Only consider n-grams that appear in at least this many documents (avoids noise).
    chunk_size : int
        Number of n-grams per chunk for presence matrix (limits memory).
    stopwords : frozenset | set | None
        Words to exclude; if None, uses DEFAULT_STOPWORDS.
    method : str
        "spearman" (default for ordinal) or "pearson".

    Returns
    -------
    list[str]
        Top top_k n-gram strings (by max absolute correlation with any target).
    """
    ngram_to_docs = _get_ngram_to_doc_indices(
        df, text_column, ngram_size, min_doc_count, stopwords
    )
    if not ngram_to_docs:
        return []
    index = df.index
    ngram_list = list(ngram_to_docs.keys())
    max_abs_corr: dict[str, float] = {ng: 0.0 for ng in ngram_list}
    for tc in target_columns:
        if tc not in df.columns:
            continue
        target = pd.to_numeric(df[tc], errors="coerce").reindex(index).dropna()
        if len(target) < 2:
            continue
        common = index.intersection(target.index)
        if len(common) < 2:
            continue
        t = target.loc[common].astype(float)
        if method == "spearman":
            target_vec = t.rank().to_numpy(dtype=np.float64)
        else:
            target_vec = t.to_numpy(dtype=np.float64)
        for start in range(0, len(ngram_list), chunk_size):
            chunk_ngrams = ngram_list[start : start + chunk_size]
            presence_chunk = _build_presence_chunk(common, ngram_to_docs, chunk_ngrams)
            corr_arr = _spearman_corr_binary_matrix_with_target(
                presence_chunk.to_numpy(dtype=np.float64), target_vec
            )
            for i, ng in enumerate(chunk_ngrams):
                val = float(corr_arr[i]) if pd.notna(corr_arr[i]) else 0.0
                if abs(val) > abs(max_abs_corr[ng]):
                    max_abs_corr[ng] = val
    if not max_abs_corr:
        return []
    series = pd.Series(max_abs_corr)
    return series.abs().nlargest(top_k).index.tolist()


def get_ngram_presence_matrix(
    df: pd.DataFrame,
    text_column: str,
    ngram_size: int,
    top_k_ngrams: int = 25,
    ngram_list: list[str] | None = None,
    stopwords: frozenset[str] | set[str] | None = None,
) -> pd.DataFrame:
    """
    Binary presence (0/1) of n-grams per essay. For use with plot_length_target_heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with text_column.
    text_column : str
        Name of the text column.
    ngram_size : int
        2 for bigrams, 3 for trigrams.
    top_k_ngrams : int
        Used only if ngram_list is None: number of top corpus n-grams by frequency.
    ngram_list : list[str] | None
        If provided, use these n-grams as columns (e.g. from get_top_ngrams_by_absolute_correlation).
    stopwords : frozenset | set | None
        Words to exclude; if None, uses DEFAULT_STOPWORDS.

    Returns
    -------
    pd.DataFrame
        Index = df.index, columns = n-gram strings, values = 0 or 1.
    """
    if ngram_list is not None:
        top_ngrams = ngram_list
    else:
        top_ngrams = get_top_n_ngrams_corpus(
            df[text_column], n=top_k_ngrams, ngram_size=ngram_size, stopwords=stopwords
        ).index.tolist()
    if not top_ngrams:
        return pd.DataFrame(index=df.index)
    presence = pd.DataFrame(0, index=df.index, columns=top_ngrams)
    for idx in df.index:
        ngrams_in_text = set(
            _get_sentence_bound_ngrams_for_text(
                df.loc[idx, text_column], ngram_size, stopwords
            )
        )
        for ng in top_ngrams:
            if ng in ngrams_in_text:
                presence.loc[idx, ng] = 1
    return presence


def get_ngram_presence_correlation(
    df: pd.DataFrame,
    text_column: str,
    target_series: pd.Series,
    ngram_size: int = 2,
    top_k_ngrams: int = 100,
    stopwords: frozenset[str] | set[str] | None = None,
    method: str = "spearman",
) -> pd.Series:
    """
    Spearman (or Pearson) correlation of binary n-gram presence (0/1 per essay) with target.

    First gets top_k_ngrams from the corpus, then for each essay computes 0/1 per n-gram,
    then correlates each n-gram's vector with target_series. Aligns on df.index.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with text_column.
    text_column : str
        Name of the text column.
    target_series : pd.Series
        Target variable (e.g. df["Vocabulary_1"]).
    ngram_size : int
        2 for bigrams, 3 for trigrams.
    top_k_ngrams : int
        Number of top corpus n-grams to use for presence features.
    stopwords : frozenset | set | None
        Words to exclude; if None, uses DEFAULT_STOPWORDS.
    method : str
        "spearman" or "pearson".

    Returns
    -------
    pd.Series
        Index = n-gram string, value = correlation with target (sorted by absolute value).
    """
    # Get top-k n-grams from corpus
    top_ngrams = get_top_n_ngrams_corpus(
        df[text_column], n=top_k_ngrams, ngram_size=ngram_size, stopwords=stopwords
    ).index.tolist()
    if not top_ngrams:
        return pd.Series(dtype=float)
    # Build presence matrix (index = df.index, columns = n-grams)
    common_index = df.index.intersection(target_series.dropna().index)
    target_aligned = target_series.loc[common_index].astype(float)
    presence = pd.DataFrame(0, index=common_index, columns=top_ngrams)
    for idx in common_index:
        if idx not in df.index:
            continue
        ngrams_in_text = set(
            _get_sentence_bound_ngrams_for_text(
                df.loc[idx, text_column], ngram_size, stopwords
            )
        )
        for ng in top_ngrams:
            if ng in ngrams_in_text:
                presence.loc[idx, ng] = 1
    # Correlate each n-gram with target using pandas (no scipy required)
    corrs = {}
    for ng in top_ngrams:
        r = presence[ng].loc[common_index].corr(target_aligned, method=method)
        corrs[ng] = float(r) if pd.notna(r) else 0.0
    return pd.Series(corrs).sort_values(ascending=False)


def plot_top_ngrams_bar(
    ngram_counts: pd.Series,
    title: str,
    top_n: int = 10,
    xlabel: str = "Count",
    figsize: tuple[float, float] = (10, 5),
):
    """
    Horizontal bar chart of top N n-grams so that labels are clearly visible.

    Parameters
    ----------
    ngram_counts : pd.Series
        Index = n-gram string, value = count (e.g. from get_top_n_ngrams_corpus).
    title : str
        Figure title.
    top_n : int
        Number of bars to show.
    xlabel : str
        Label for horizontal axis.
    figsize : tuple
        Figure size (width, height).

    Returns
    -------
    plt.Figure
        The figure.
    """
    if plt is None:
        raise ImportError("plot_top_ngrams_bar requires matplotlib")
    plot_series = ngram_counts.head(top_n)
    if plot_series.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        return fig
    fig, ax = plt.subplots(figsize=figsize)
    y_pos = range(len(plot_series))
    ax.barh(y_pos, plot_series.values, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_series.index.tolist(), fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("N-gram")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_ngram_correlation_bars(
    corr_series: pd.Series,
    target_name: str,
    top_n: int = 10,
    figsize: tuple[float, float] = (10, 8),
):
    """
    Two horizontal bar charts: top N positive and top N negative correlations of n-gram presence with target.

    Parameters
    ----------
    corr_series : pd.Series
        N-gram -> correlation (e.g. from get_ngram_presence_correlation).
    target_name : str
        Name of target (e.g. "Vocabulary_1") for title and axis clarity.
    top_n : int
        Number of bars per subplot.
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
        Figure with two subplots: positive correlations, negative correlations.
    """
    if plt is None:
        raise ImportError("plot_ngram_correlation_bars requires matplotlib")
    fig, (ax_pos, ax_neg) = plt.subplots(2, 1, figsize=figsize)
    # Top positive
    pos = corr_series[corr_series > 0].head(top_n)
    if not pos.empty:
        y_pos = range(len(pos))
        ax_pos.barh(y_pos, pos.values, align="center", color="steelblue")
        ax_pos.set_yticks(y_pos)
        ax_pos.set_yticklabels(pos.index.tolist(), fontsize=10)
        ax_pos.set_xlabel(f"Correlation with {target_name} (Spearman)")
        ax_pos.set_ylabel("N-gram")
        ax_pos.set_title(f"Top {top_n} n-grams positively correlated with {target_name}")
    else:
        ax_pos.set_title(f"No positive correlations with {target_name}")
    # Top negative (most negative first)
    neg = corr_series[corr_series < 0].tail(top_n).iloc[::-1]
    if not neg.empty:
        y_neg = range(len(neg))
        ax_neg.barh(y_neg, neg.values, align="center", color="coral")
        ax_neg.set_yticks(y_neg)
        ax_neg.set_yticklabels(neg.index.tolist(), fontsize=10)
        ax_neg.set_xlabel(f"Correlation with {target_name} (Spearman)")
        ax_neg.set_ylabel("N-gram")
        ax_neg.set_title(f"Top {top_n} n-grams negatively correlated with {target_name}")
    else:
        ax_neg.set_title(f"No negative correlations with {target_name}")
    plt.tight_layout(pad=1.2)
    return fig


# -----------------------------------------------------------------------------
# EDA 2.6 POS tagging
# -----------------------------------------------------------------------------


def get_pos_ratios_df(
    df: pd.DataFrame,
    text_column: str,
) -> pd.DataFrame:
    """
    Part-of-speech ratios per essay (noun, verb, adj, adv, other). Uses NLTK Penn Treebank.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with text_column.
    text_column : str
        Name of the text column (e.g. "Text_cleaned").

    Returns
    -------
    pd.DataFrame
        Index = df.index, columns = noun_ratio, verb_ratio, adj_ratio, adv_ratio, other_ratio.
    """
    # NOTE: We intentionally tokenize using the same "strip . ? !" normalization used
    # elsewhere in EDA (n-grams / richness). This ensures tokens like "word" and "word."
    # are treated consistently, and punctuation does not leak into the POS statistics.
    if nltk is None or pos_tag is None:
        raise ImportError("get_pos_ratios_df requires nltk. Install with: pip install nltk")
    # Ensure tagger data exists (newer NLTK: averaged_perceptron_tagger_eng; older: averaged_perceptron_tagger).
    for resource in ("averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"):
        try:
            nltk.data.find(f"taggers/{resource}")
            break
        except LookupError:
            nltk.download(resource, quiet=True)
    out = pd.DataFrame(
        index=df.index,
        columns=["noun_ratio", "verb_ratio", "adj_ratio", "adv_ratio", "other_ratio"],
        dtype=float,
    )
    for idx in df.index:
        text = df.loc[idx, text_column]
        if pd.isna(text) or not str(text).strip():
            out.loc[idx] = np.nan
            continue

        # Sentence split first, then strip `. ? !` on demand within each sentence.
        # This matches the "sentence-bound" approach used in the n-gram analysis and
        # avoids POS tokens that cross sentence boundaries due to delimiter artifacts.
        sentences = _split_sentences(str(text))
        tokens: list[str] = []
        for sent in sentences:
            tokens.extend(_normalize_text_for_richness(sent))

        if not tokens:
            out.loc[idx] = np.nan
            continue

        tags = pos_tag(tokens)
        n = len(tags)
        counts = {"noun": 0, "verb": 0, "adj": 0, "adv": 0, "other": 0}
        for _w, tag in tags:
            # Each tag is a Penn Treebank string like "NN", "VBD", etc.
            t = tag if isinstance(tag, str) else str(tag)
            if t in _POS_NOUN:
                counts["noun"] += 1
            elif t in _POS_VERB:
                counts["verb"] += 1
            elif t in _POS_ADJ:
                counts["adj"] += 1
            elif t in _POS_ADV:
                counts["adv"] += 1
            else:
                counts["other"] += 1
        out.loc[idx, "noun_ratio"] = counts["noun"] / n
        out.loc[idx, "verb_ratio"] = counts["verb"] / n
        out.loc[idx, "adj_ratio"] = counts["adj"] / n
        out.loc[idx, "adv_ratio"] = counts["adv"] / n
        out.loc[idx, "other_ratio"] = counts["other"] / n
    return out


# -----------------------------------------------------------------------------
# EDA 2.7 Readability
# -----------------------------------------------------------------------------


def get_readability_features_df(
    df: pd.DataFrame,
    text_column: str,
) -> pd.DataFrame:
    """
    Readability features per essay: Flesch–Kincaid grade level and Flesch Reading Ease.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with text_column.
    text_column : str
        Name of the text column (e.g. "Text_cleaned").

    Returns
    -------
    pd.DataFrame
        Index = df.index, columns = flesch_kincaid_grade, flesch_reading_ease.
    """
    if textstat is None:
        raise ImportError(
            "get_readability_features_df requires textstat. Install with: pip install textstat"
        )
    out = pd.DataFrame(
        index=df.index,
        columns=["flesch_kincaid_grade", "flesch_reading_ease"],
        dtype=float,
    )
    for idx in df.index:
        text = df.loc[idx, text_column]
        if pd.isna(text) or not str(text).strip():
            out.loc[idx] = np.nan
            continue
        s = str(text)
        try:
            out.loc[idx, "flesch_kincaid_grade"] = textstat.flesch_kincaid_grade(s)
            out.loc[idx, "flesch_reading_ease"] = textstat.flesch_reading_ease(s)
        except Exception:
            out.loc[idx] = np.nan
    return out


# -----------------------------------------------------------------------------
# EDA 2.8 Sentence embeddings + PCA
# -----------------------------------------------------------------------------


def get_essay_embeddings(
    df: pd.DataFrame,
    text_column: str,
    model: "SentenceTransformer | None" = None,
    model_name: str | None = None,
    use_fast_model: bool = False,
    batch_size: int = 256,
) -> np.ndarray:
    """
    One embedding vector per essay: sentence-split, embed all sentences in batches, mean-pool per essay, L2-normalize.

    Uses _split_sentences so embeddings respect sentence boundaries. All sentences across
    essays are encoded in one batched call for speed; then each essay's vectors are
    mean-pooled and L2-normalized. Empty essays get a zero vector of the same dimension.

    For CPU-only: pass use_fast_model=True for a smaller model, or pass a pre-loaded model
    to avoid reloading on repeated runs.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with text_column.
    text_column : str
        Name of the text column (e.g. "Text_cleaned").
    model : SentenceTransformer | None
        Pre-loaded model; if set, model_name and use_fast_model are ignored (avoids reload).
    model_name : str | None
        Model name to load; if None and model is None, uses DEFAULT_EMBEDDING_MODEL or FAST.
    use_fast_model : bool
        If True and model is None, use DEFAULT_EMBEDDING_MODEL_FAST (faster on CPU, slightly lower quality).
    batch_size : int
        Batch size for model.encode (default 256 for CPU throughput; reduce if OOM).

    Returns
    -------
    np.ndarray
        Shape (len(df), embedding_dim), same row order as df.index.
    """
    if SentenceTransformer is None:
        raise ImportError(
            "get_essay_embeddings requires sentence-transformers. Install with: pip install sentence-transformers"
        )
    if model is None:
        if use_fast_model:
            name = DEFAULT_EMBEDDING_MODEL_FAST
        else:
            name = model_name if model_name is not None else DEFAULT_EMBEDDING_MODEL
        model = SentenceTransformer(name)
    dim = model.get_sentence_embedding_dimension()
    out = np.zeros((len(df), dim), dtype=np.float32)

    # Collect all sentences and which essay index each belongs to (one encode over full dataset).
    sentences_flat = []
    essay_indices = []
    for i, idx in enumerate(df.index):
        text = df.loc[idx, text_column]
        if pd.isna(text) or not str(text).strip():
            continue
        sentences = _split_sentences(str(text))
        if not sentences:
            continue
        sentences_flat.extend(sentences)
        essay_indices.extend([i] * len(sentences))

    if not sentences_flat:
        return out

    # Single batched encode (much faster than one encode per essay).
    vecs = model.encode(
        sentences_flat,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # Mean-pool per essay and L2-normalize (single pass over sentences).
    essay_indices = np.array(essay_indices)
    sums = np.zeros((len(df), dim), dtype=np.float64)
    counts = np.zeros(len(df), dtype=np.int64)
    for j in range(len(essay_indices)):
        i = essay_indices[j]
        sums[i] += vecs[j]
        counts[i] += 1
    for i in range(len(df)):
        if counts[i] == 0:
            continue
        doc_vec = sums[i] / counts[i]
        n = np.linalg.norm(doc_vec)
        if n > 0:
            doc_vec = doc_vec / n
        out[i] = doc_vec.astype(np.float32)

    return out


def get_embedding_pca_2d(
    embeddings: np.ndarray,
    df: pd.DataFrame | None = None,
    target_columns: list[str] | None = None,
    top_k: int = 500,
    random_state: int = 42,
):
    """
    Fit PCA and return 2D coordinates and optionally correlation table.

    When df and target_columns are provided: regular PCA with top_k components by variance
    (PC1, PC2, ...); 2D = (PC1, PC2); pca_corr and pca_scores_top for all top_k components.
    Otherwise: PCA(2) on full embeddings only.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (n_samples, n_features).
    df : pd.DataFrame | None
        If provided with target_columns, returns (pca, X_2d, pca_corr, pca_scores_top).
    target_columns : list[str] | None
        Target column names for correlation table (e.g. ["Vocabulary_1", "Vocabulary_2"]).
    top_k : int
        Number of top components by variance when df/target_columns provided (default 500).
    random_state : int
        For reproducibility.

    Returns
    -------
    tuple
        If df and target_columns provided: (pca, X_2d, pca_corr, pca_scores_top).
        Otherwise: (pca, X_2d).
    """
    if PCA is None or StandardScaler is None:
        raise ImportError("get_embedding_pca_2d requires sklearn. Install with: pip install scikit-learn")
    if df is not None and target_columns is not None:
        n_comp = min(top_k, embeddings.shape[0], embeddings.shape[1])
        if n_comp < 1:
            n_comp = min(2, embeddings.shape[0], embeddings.shape[1])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(embeddings)
        pca = PCA(n_components=n_comp, random_state=random_state)
        scores = pca.fit_transform(X_scaled)
        idx = df.index[: len(scores)]
        pca_scores_top = pd.DataFrame(
            scores,
            index=idx,
            columns=[f"pc{i}" for i in range(n_comp)],
        )
        X_2d = scores[:, :2]
        out_corr = pd.DataFrame(index=pca_scores_top.columns)
        for tc in target_columns:
            if tc not in df.columns:
                continue
            t = pd.to_numeric(df.loc[idx, tc], errors="coerce").dropna()
            common = pca_scores_top.index.intersection(t.index)
            if len(common) < 2:
                continue
            out_corr[tc] = pca_scores_top.loc[common].corrwith(t.loc[common], method="spearman")
        return pca, X_2d, out_corr, pca_scores_top
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings)
    pca = PCA(n_components=2, random_state=random_state)
    X_2d = pca.fit_transform(X_scaled)
    return pca, X_2d


def plot_embedding_scatter_2d(
    X_2d: np.ndarray,
    df: pd.DataFrame,
    target_column: str,
    title: str | None = None,
    ax=None,
    figsize: tuple[float, float] = (7, 5),
):
    """
    Scatter plot of 2D embedding space, points colored by target (Vocabulary score).

    Parameters
    ----------
    X_2d : np.ndarray
        Shape (n_samples, 2).
    df : pd.DataFrame
        Must contain target_column; rows aligned with X_2d by position (same order as when embeddings were built).
    target_column : str
        Column name for color (e.g. "Vocabulary_1").
    title : str | None
        Plot title.
    ax : plt.Axes | None
        Axes to use; if None, creates new figure.
    figsize : tuple
        Figure size when ax is None.

    Returns
    -------
    plt.Axes
    """
    if plt is None:
        raise ImportError("plot_embedding_scatter_2d requires matplotlib")
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    t = pd.to_numeric(df[target_column].iloc[: len(X_2d)], errors="coerce")
    valid = ~t.isna()
    sc = ax.scatter(
        X_2d[valid, 0],
        X_2d[valid, 1],
        c=t[valid].astype(float),
        cmap="viridis",
        alpha=0.7,
        s=20,
    )
    plt.colorbar(sc, ax=ax, label=target_column)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title if title else f"Embedding PCA 2D colored by {target_column}")
    return ax


def plot_embedding_scatter_2d_by_cluster(
    X_2d: np.ndarray,
    cluster_labels: np.ndarray,
    title: str | None = None,
    ax=None,
    figsize: tuple[float, float] = (7, 5),
):
    """
    Scatter plot of 2D embedding space, points colored by KMeans cluster id.

    Parameters
    ----------
    X_2d : np.ndarray
        Shape (n_samples, 2); same order as cluster_labels.
    cluster_labels : np.ndarray
        Integer cluster id per sample (e.g. from get_embedding_kmeans_labels).
    title : str | None
        Plot title.
    ax : plt.Axes | None
        Axes to use; if None, creates new figure.
    figsize : tuple
        Figure size when ax is None.

    Returns
    -------
    plt.Axes
    """
    if plt is None:
        raise ImportError("plot_embedding_scatter_2d_by_cluster requires matplotlib")
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    n = min(len(X_2d), len(cluster_labels))
    sc = ax.scatter(
        X_2d[:n, 0],
        X_2d[:n, 1],
        c=cluster_labels[:n],
        cmap="tab10",
        alpha=0.6,
        s=15,
    )
    plt.colorbar(sc, ax=ax, label="Cluster")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title if title else "Embedding PCA 2D colored by cluster")
    return ax


def get_embedding_pca_components_df(
    embeddings: np.ndarray,
    n_components: int,
    index: pd.Index,
) -> pd.DataFrame:
    """
    Fit PCA and return DataFrame of component scores (one row per essay).

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (n_samples, n_features).
    n_components : int
        Number of components.
    index : pd.Index
        Index for the returned DataFrame (e.g. df.index).

    Returns
    -------
    pd.DataFrame
        Index = index[:len(embeddings)], columns = pc0, pc1, ...
    """
    if PCA is None or StandardScaler is None:
        raise ImportError("get_embedding_pca_components_df requires sklearn")
    n_components = min(n_components, embeddings.shape[0], embeddings.shape[1])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings)
    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(X_scaled)
    idx = index[: len(scores)]
    return pd.DataFrame(
        scores,
        index=idx,
        columns=[f"pc{i}" for i in range(n_components)],
    )


def get_pca_component_correlations(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    target_columns: list[str],
    n_components: int | None = None,
    top_k: int = 50,
    method: str = "spearman",
    return_scores: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit PCA on embeddings, correlate each component with each target, return top_k by |correlation|.

    Components are ranked by mean absolute correlation with the target columns (so we keep
    the directions most predictive of vocabulary level), not by variance order.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (n_samples, n_features); row order must match df.index.
    df : pd.DataFrame
        Contains target_columns.
    target_columns : list[str]
        Target column names (e.g. ["Vocabulary_1", "Vocabulary_2"]).
    n_components : int | None
        Number of PCA components to fit; if None, uses min(500, n_features, n_samples-1).
    top_k : int
        Number of components to return, ranked by mean absolute correlation with targets (default 50).
    method : str
        Correlation method (e.g. "spearman" for ordinal targets).
    return_scores : bool
        If True, return (corr_df, score_df) so score_df can be passed to plot_length_target_heatmap.

    Returns
    -------
    pd.DataFrame or tuple
        Correlation table (rows = top_k components). If return_scores True, (corr_df, score_df).
    """
    if PCA is None or StandardScaler is None:
        raise ImportError("get_pca_component_correlations requires sklearn")
    max_comp = min(embeddings.shape[0], embeddings.shape[1])
    if n_components is None:
        n_components = min(500, max_comp)
    else:
        n_components = min(n_components, max_comp)
    if n_components < 1:
        return pd.DataFrame()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings)
    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(X_scaled)
    idx = df.index[: len(components)]
    comp_df = pd.DataFrame(
        components,
        index=idx,
        columns=[f"pc{i}" for i in range(n_components)],
    )
    out = pd.DataFrame(index=comp_df.columns)
    for tc in target_columns:
        if tc not in df.columns:
            continue
        t = pd.to_numeric(df.loc[idx, tc], errors="coerce").dropna()
        common_t = comp_df.index.intersection(t.index)
        if len(common_t) < 2:
            continue
        out[tc] = comp_df.loc[common_t].corrwith(t.loc[common_t], method=method)
    # Rank by mean absolute correlation with targets; keep top_k.
    target_cols_present = [c for c in target_columns if c in out.columns]
    if not target_cols_present:
        return out.head(top_k)
    out["_rank"] = out[target_cols_present].abs().mean(axis=1)
    out = out.sort_values("_rank", ascending=False).drop(columns=["_rank"])
    top_names = out.head(top_k).index.tolist()
    corr_result = out.head(top_k)
    if return_scores:
        score_df = comp_df[top_names].copy()
        return corr_result, score_df
    return corr_result


# -----------------------------------------------------------------------------
# EDA 2.9 Clustering
# -----------------------------------------------------------------------------


def get_embedding_kmeans_labels(
    embeddings: np.ndarray,
    n_clusters: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """
    KMeans cluster labels for embedding rows.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (n_samples, n_features).
    n_clusters : int
        Number of clusters (e.g. 5 to mirror 0–4 or 1–5 score levels).
    random_state : int
        For reproducibility.

    Returns
    -------
    np.ndarray
        Integer labels, shape (n_samples,).
    """
    if KMeans is None:
        raise ImportError("get_embedding_kmeans_labels requires sklearn")
    # L2-normalize so Euclidean distance reflects cosine similarity (recommended for text embeddings).
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    X = embeddings / norms
    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    return km.fit_predict(X)


def get_cluster_space_2d(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    random_state: int = 42,
) -> np.ndarray:
    """
    Compute 2D coordinates in "cluster space": distance to each centroid, then PCA to 2D.

    Centroids are computed as mean of embeddings per cluster. Each point is then
    represented by its distance to each centroid (n_clusters dims); PCA(2) on
    that gives a 2D view that is about cluster structure, not raw variance (PCA).

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (n_samples, n_features); same order as cluster_labels.
    cluster_labels : np.ndarray
        Integer cluster id per sample (e.g. from get_embedding_kmeans_labels).
    random_state : int
        For PCA reproducibility.

    Returns
    -------
    np.ndarray
        Shape (n_samples, 2); 2D coordinates for plotting (cluster space, not PCA).
    """
    if PCA is None:
        raise ImportError("get_cluster_space_2d requires sklearn")
    n = len(embeddings)
    assert n == len(cluster_labels), "embeddings and cluster_labels length must match"
    k = int(np.max(cluster_labels) + 1)
    # Centroids: mean of embeddings per cluster (same as KMeans would give)
    centroids = np.zeros((k, embeddings.shape[1]), dtype=embeddings.dtype)
    for c in range(k):
        mask = cluster_labels == c
        if np.any(mask):
            centroids[c] = np.mean(embeddings[mask], axis=0)
    # Distance of each point to each centroid -> (n, k)
    dists = np.zeros((n, k))
    for c in range(k):
        dists[:, c] = np.linalg.norm(embeddings - centroids[c], axis=1)
    # Standardize distances before PCA so axes are centered and evenly scaled
    n_comp = min(2, n, k)
    if n_comp < 2:
        return np.zeros((n, 2))
    scaler = StandardScaler()
    dists_scaled = scaler.fit_transform(dists)
    pca = PCA(n_components=n_comp, random_state=random_state)
    out = pca.fit_transform(dists_scaled)
    if out.shape[1] < 2:
        out = np.column_stack([out[:, 0], np.zeros(n)])
    return out


def plot_cluster_space_score_labels(
    X_2d_cluster: np.ndarray,
    cluster_labels: np.ndarray,
    scores: np.ndarray | pd.Series,
    title: str | None = None,
    ax=None,
    figsize: tuple[float, float] = (7, 5),
    fontsize: float = 9,
    alpha: float = 0.95,
    subsample: int | None = 500,
) -> "plt.Axes":
    """
    Plot 2D cluster space with score digit at each point; color by cluster.

    No dots: each point is the actual score (0-5) as text, colored by its cluster.
    Axes are the 2D from cluster-distance space (not PCA on raw embeddings).

    Parameters
    ----------
    X_2d_cluster : np.ndarray
        Shape (n_samples, 2) from get_cluster_space_2d.
    cluster_labels : np.ndarray
        Integer cluster id per sample.
    scores : np.ndarray or pd.Series
        Score per sample (e.g. Vocabulary_1 or Vocabulary_2, 0-5); same length as X_2d_cluster.
    title : str | None
        Plot title.
    ax : plt.Axes | None
        Axes to use; if None, creates new figure.
    figsize : tuple
        Figure size when ax is None.
    fontsize : float
        Text size for score digits (small for many points).
    alpha : float
        Text transparency.
    subsample : int | None
        If set, plot only this many random points for readability (default 2000 when n is large).

    Returns
    -------
    plt.Axes
    """
    if plt is None:
        raise ImportError("plot_cluster_space_score_labels requires matplotlib")
    scores = np.asarray(scores).ravel()
    n = min(len(X_2d_cluster), len(cluster_labels), len(scores))
    X_2d_cluster = X_2d_cluster[:n]
    cluster_labels = cluster_labels[:n]
    scores = scores[:n]
    # Subsample so score digits are readable (default 500; full data is too dense)
    if subsample is not None and n > subsample:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=subsample, replace=False)
        X_2d_cluster = X_2d_cluster[idx]
        cluster_labels = cluster_labels[idx]
        scores = np.asarray(scores)[idx]
        n = subsample
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    # Set axis limits FIRST so all data is visible (ax.text does not auto-scale axes)
    pad = 0.05
    x_min, x_max = X_2d_cluster[:, 0].min(), X_2d_cluster[:, 0].max()
    y_min, y_max = X_2d_cluster[:, 1].min(), X_2d_cluster[:, 1].max()
    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)
    ax.set_xlim(x_min - pad * x_range, x_max + pad * x_range)
    ax.set_ylim(y_min - pad * y_range, y_max + pad * y_range)
    k = int(np.max(cluster_labels) + 1)
    cmap = plt.cm.get_cmap("tab10", max(k, 10))
    for i in range(n):
        c = cluster_labels[i]
        color = cmap(c)
        try:
            s = int(round(float(scores[i])))
        except (ValueError, TypeError):
            s = "?"
        ax.text(
            X_2d_cluster[i, 0],
            X_2d_cluster[i, 1],
            str(s),
            fontsize=fontsize,
            color=color,
            alpha=alpha,
            ha="center",
            va="center",
        )
    ax.set_xlabel("Cluster space dim 1")
    ax.set_ylabel("Cluster space dim 2")
    ax.set_title(title if title else "K-means cluster space (text = score)")
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=cmap(c), label=f"Cluster {c}") for c in range(k)]
    ax.legend(handles=patches, loc="best", fontsize=8)
    return ax


def get_cluster_target_summary(
    labels: np.ndarray,
    df: pd.DataFrame,
    target_columns: list[str],
) -> pd.DataFrame:
    """
    Per-cluster mean of target columns and sample count.

    Parameters
    ----------
    labels : np.ndarray
        Cluster label per row (same length as df or aligned by position).
    df : pd.DataFrame
        Contains target_columns.
    target_columns : list[str]
        Target column names (e.g. ["Vocabulary_1", "Vocabulary_2"]).

    Returns
    -------
    pd.DataFrame
        Index = cluster id (0 .. n_clusters-1); columns = count and mean per target.
    """
    n = min(len(labels), len(df))
    labels = labels[:n]
    df_slice = df.iloc[:n]
    clusters = np.unique(labels)
    out = pd.DataFrame(index=pd.Index(clusters, name="cluster"))
    out["count"] = pd.Series(labels).value_counts().sort_index()
    for tc in target_columns:
        if tc not in df_slice.columns:
            continue
        t = pd.to_numeric(df_slice[tc], errors="coerce").values
        out[f"{tc}_mean"] = [np.nanmean(t[labels == c]) for c in clusters]
    return out


# -----------------------------------------------------------------------------
# EDA 2.10 Summary
# -----------------------------------------------------------------------------


def get_handcrafted_feature_ranked_correlations(
    length_df: pd.DataFrame,
    richness_df: pd.DataFrame,
    pos_df: pd.DataFrame,
    readability_df: pd.DataFrame,
    df: pd.DataFrame,
    target_columns: list[str],
    method: str = "spearman",
) -> pd.DataFrame:
    """
    Concatenate handcrafted feature DataFrames, correlate each feature with each target,
    return table with features ranked by mean absolute correlation.

    Parameters
    ----------
    length_df, richness_df, pos_df, readability_df : pd.DataFrame
        Per-essay feature tables (index = essay index).
    df : pd.DataFrame
        Contains target_columns.
    target_columns : list[str]
        Target column names (e.g. ["Vocabulary_1", "Vocabulary_2"]).
    method : str
        Correlation method.

    Returns
    -------
    pd.DataFrame
        Rows = feature names, columns = target_columns + "|corr|_mean"; sorted by |corr|_mean descending.
    """
    combined = pd.concat(
        [length_df, richness_df, pos_df, readability_df],
        axis=1,
        join="inner",
    )
    combined = combined.dropna(how="all")
    common = combined.index.intersection(df.index)
    combined = combined.loc[common]
    corr_df = pd.DataFrame(index=combined.columns)
    for tc in target_columns:
        if tc not in df.columns:
            continue
        t = pd.to_numeric(df.loc[common, tc], errors="coerce").dropna()
        common_t = combined.index.intersection(t.index)
        if len(common_t) < 2:
            continue
        corr_df[tc] = combined.loc[common_t].corrwith(t.loc[common_t], method=method)
    corr_df["|corr|_mean"] = corr_df[[c for c in target_columns if c in corr_df.columns]].abs().mean(axis=1)
    return corr_df.sort_values("|corr|_mean", ascending=False)
