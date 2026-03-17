"""
Vocabulary Level Prediction — shared utilities.
Preprocessing, data loading, and helper functions.
"""

# -----------------------------------------------------------------------------
# Environment setting
# -----------------------------------------------------------------------------
import re
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # Install with: pip install matplotlib

try:
    from sklearn.metrics import cohen_kappa_score
except ImportError:
    cohen_kappa_score = None  # Install with: pip install scikit-learn

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
    t = target.loc[common]
    return f.corrwith(t, method=method)


def plot_length_target_heatmap(
    features_df: pd.DataFrame,
    target_columns: list[str],
    df: pd.DataFrame,
    method: str = "spearman",
    ax=None,
    title: str | None = None,
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
        _, ax = plt.subplots(
            figsize=(max(5, len(target_columns) * 2.5), max(4, len(feature_cols) * 1.0))
        )
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
    from collections import Counter
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

# Common English stopwords (minimal set) for word-frequency analysis when none provided.
DEFAULT_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will",
    "with", "this", "but", "they", "have", "had", "what", "when", "where", "who",
    "which", "why", "how", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "just", "if", "or", "because", "until", "while",
})


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
    from collections import Counter
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
    from collections import Counter
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
