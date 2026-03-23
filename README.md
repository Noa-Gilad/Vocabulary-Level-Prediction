# Vocabulary Level Prediction

## Purpose

The task is to predict essay-level vocabulary proficiency on an ordinal scale (`0-5`) from text.  
Two modeling tracks were intentionally separated:

1. **Classical NLP + ML** (`Model_selection.ipynb`) 
2. **RoBERTa fine-tuning** (`BERT_comparison.ipynb`) 

This README is written as the main review document. The methodological choices, non-default parameters, and resulting conclusions are recorded explicitly.

---

## Project Layout and Execution

- `EDA.ipynb`: exploratory data analysis and feature diagnostics.
- `Model_selection.ipynb`: classical model registry, cross-validation, Optuna tuning, final test evaluation.
- `BERT_comparison.ipynb`: transformer training pipeline.
- `modeling_utils.py`: preprocessing helpers, CV runner, metrics, weighting, Optuna wrappers, plotting utilities.
- `utils.py`: text preprocessing, handcrafted feature extraction, readability and embedding helpers.
- `transformers_utils.py`: weighted trainer and transformer tuning utilities.

Recommended execution order:
1. `EDA.ipynb`
2. `Model_selection.ipynb`
3. `BERT_comparison.ipynb`

Environment:

```bash
pip install -r requirements.txt
```

---

## 1. Problem Formulation

Each essay has two independent ordinal ratings:
- `Vocabulary_1`
- `Vocabulary_2`

### 1.1 Final target definition

The supervised label is built with `create_consensus_target`:

1. arithmetic mean of `Vocabulary_1` and `Vocabulary_2`,
2. `np.rint` rounding,
3. integer cast and clipping to `[0, 5]`.

Therefore, the modeling target (`target_vocab`) is an **integer consensus score**.

Reason for this choice:
- direct alignment with ordinal evaluation;
- single consistent target representation across CV, tuning, and final test reporting.

### 1.2 Evaluation metrics

Primary metric: **Quadratic Weighted Kappa (QWK)**.  
Secondary metrics: **RMSE**, **MAE**.

Why QWK is used here:
- Cohen's kappa measures agreement beyond random chance.
- Quadratic weighting penalizes distant ordinal errors more strongly than near-miss errors.
- This matches rubric-based scoring: predicting `4` instead of `5` is less severe than predicting `1` instead of `5`.

Implementation detail in this project:
- predictions are rounded and clipped before QWK;
- fixed labels `0..5` are always supplied to `cohen_kappa_score`, so fold-to-fold comparisons remain on the same label frame.

---

## 2. Data Handling Protocol

### 2.1 Training-set agreement filtering

`filter_by_rater_agreement` is applied on training data with:
- `max_allowed_gap = 2`
- retained condition: `abs(Vocabulary_1 - Vocabulary_2) < 2` (equivalent to `|delta| <= 1`)

Observed count in run logs:
- `7112 -> 6804` rows (308 removed).

Rationale:
- EDA indicated that high disagreement is uncommon and comparatively noisier;
- filtering improves label consistency during fitting.

### 2.2 Test-set policy

On `Data/test.csv`, no agreement filtering is applied.  
Consensus targets are created only for evaluation labels, while all rows are retained.

Rationale:
- no label-based row exclusion at evaluation time;
- closer approximation to deployment behavior.

---

## 3. EDA: Findings and Decisions

### 3.1 Rater agreement and imbalance

Documented statistics:
- weighted kappa: ~`0.408`
- exact agreement: ~`50.3%`
- Spearman rank correlation: ~`0.380`

Interpretation:
- agreement is moderate, not high;
- disagreement is mostly local (`|delta|` concentrated around `0` and `1`);
- class frequencies are imbalanced.

Decision impact:
- stratified folds were required;
- balanced sample weighting was justified;
- agreement filtering was applied to training only.

### 3.2 Lexical and structural handcrafted features

Feature groups examined included:
- length-related: `char_count`, `word_count`, `sentence_count`, `avg_word_length`;
- richness-related: `unique_words`, `ttr`, `hapax_ratio`.

In practical terms, `ttr` (Type-Token Ratio) reflects how diverse an essay's vocabulary is: higher values indicate a wider variety of words relative to essay length, while lower values indicate more repetition; `hapax_ratio` complements this by measuring how much of the vocabulary appears only once.

Correlation summary from the notebook ranking:
- four features showed intermediate positive association with both raters:
  - `unique_words` (`|corr|_mean ~ 0.357`)
  - `sentence_count` (`~0.307`)
  - `char_count` (`~0.299`)
  - `word_count` (`~0.279`)
- most remaining handcrafted/readability/POS features were weak to near-zero.

Interpretation and decision:
- the top four features are informative but not strong enough for full predictive modeling on their own;
- they were therefore retained as an interpretable **baseline feature family**, not as the final representation.

### 3.3 N-gram inspection

Unigram/bigram/trigram presence correlations were reviewed for both raters.  
Specific terms showed directional association, but no compact term list provided robust class separation.

Decision:
- use TF-IDF representations in supervised models;
- do not rely on manually selected term rules as primary predictors.

### 3.4 POS and readability

POS-ratio analysis and readability measures (including Flesch Reading Ease and Flesch-Kincaid Grade Level) were evaluated.  
Readability correlations were near zero in the notebook summaries (roughly between `-0.09` and `0.03`).

Decision:
- POS/readability remained diagnostic features;
- they were not treated as dominant standalone predictors.

### 3.5 Embeddings, PCA, and clustering

Essay embeddings were visualized via PCA and analyzed with k-means summaries.  
Score distributions overlapped substantially in low-dimensional projections and cluster summaries.

Decision:
- unsupervised geometric separation was insufficient;
- predictive signal had to be extracted through supervised training.

---

## 4. Classical Modeling Protocol (`Model_selection.ipynb`)

### 4.1 Reproducibility controls

- `CV_RANDOM_STATE = 42`
- splitter: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- stratification bins: `build_stratification_bins(..., merge_below=2)`

Why `merge_below=2`:
- low classes are merged for fold-assignment stability;
- prevents fragile folds with missing rare labels;
- only fold assignment is affected; original labels remain unchanged for training/evaluation.

### 4.2 Feature tracks compared

1. `random_uniform` sanity baseline (`UniformRandomScoreRegressor`).
2. `handcrafted` features (baseline, interpretable).
3. `tfidf` features (raw text passed to fold-fitted TF-IDF pipelines).
4. `embeddings` features (precomputed essay vectors + regressors).

Leakage control:
- TF-IDF vectorization remains inside each CV pipeline and is fitted on train folds only.

### 4.3 TF-IDF configuration

From `DEFAULT_TFIDF_VECTORIZER_KWARGS`:
- `max_features = 20000`
- `ngram_range = (1, 2)`
- `sublinear_tf = True`
- `min_df = 1`
- `max_df = 0.95`

Rationale:
- unigram+bigram coverage captures local lexical context;
- feature cap controls sparsity/compute;
- very frequent terms are down-weighted by cutoff;
- rare vocabulary remains available.

### 4.4 Non-default model choices documented in notebook

Notable explicit settings:

- `Ridge(alpha=0.1, solver='lsqr', random_state=42)`  
  (lighter regularization than default `alpha=1.0` for this setting).

- `Lasso(alpha=0.001, max_iter=5000, tol=1e-3, random_state=42)`  
  (raised iteration budget for sparse/high-dimensional convergence behavior).

- `ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=50000, tol=1e-3, random_state=42)`  
  (high `max_iter` used where lower values were insufficient).

- `LinearSVR(max_iter=50000, C=0.5, dual=True)`  
  (increased iterations and explicit `C` selection).

- `MLPRegressor(hidden_layer_sizes=(256,), max_iter=500, early_stopping=True, validation_fraction=0.1, n_iter_no_change=15, random_state=42)`  
  (early stopping and validation split used to control overfit/runtime).

- `XGBRegressor(tree_method='hist', random_state=42, n_jobs=-1)`  
  (hist method for practical training speed).

- `LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1)`  
  (explicit deterministic seed and reduced logging noise).

Pipeline-specific design:
- embedding track scales linear/MLP models but not tree-based models (`RF`, `XGB`, `LGBM`);
- TF-IDF + MLP path uses `TruncatedSVD` before MLP to work in a dense reduced space.

### 4.5 CV outcomes

Representative means from notebook outputs:
- `random_baseline_U01`: QWK ~`0.0095`
- `handcrafted_Ridge`: QWK ~`0.3451`
- `tfidf_Ridge`: QWK ~`0.4661`
- `tfidf_LinearSVR`: QWK ~`0.4990`
- `tfidf_MLP`: QWK ~`0.5113` (strongest among shown lines)

Conclusion:
- handcrafted features provide an interpretable baseline;
- TF-IDF-based models provide substantially stronger predictive performance;
- TF-IDF + MLP emerged as the leading classical candidate in the logged run.

CV leaderboard figures (mean fold metrics) are exported as:
- `Figures/cv_mean_qwk.png`
- `Figures/cv_mean_rmse.png`
- `Figures/cv_mean_mae.png`

---

## 5. Classical Hyperparameter Tuning (Optuna)

Tuning was intentionally staged to balance search quality and compute cost.

### 5.1 Phase A: MLP tuning at fixed SVD width

Configuration:
- fixed `svd_n_components = 300`
- `N_OPTUNA_TRIALS = 15`
- 5-fold CV objective
- stratified objective subsample: `6000` rows from `6804`
- seeded sampling/pruning with `42`
- objective metric: mean fold QWK

Searched MLP parameters:
- `hidden_layer_sizes` in `{128, 256, 512, 256x128}`
- `alpha`
- `learning_rate_init`
- `batch_size`
- `max_iter`
- `activation`

### 5.2 Phase B: SVD width tuning with frozen MLP

Configuration:
- `N_SVD_OPTUNA_TRIALS = 15`
- search range `SVD_N_MIN=128`, `SVD_N_MAX=512`
- MLP fixed to Phase-A best settings
- same CV and weighting protocol

Recorded notebook interpretation:
- the top mean-QWK SVD trial occurred at lower dimensionality;
- `n_components = 494` was selected for improved fold stability with only marginal QWK trade-off.

Decision principle:
- stability across folds was preferred over a very small gain in mean objective.

---

## 6. Final Classical Model and Test Evaluation

Final pipeline:
- `TF-IDF -> TruncatedSVD(494) -> MLPRegressor(Phase-A best params)`
- fit on all filtered training data;
- evaluate on full test split (consensus labels, no test filtering).

Reported test metrics:
- **QWK**: `0.532867`
- **RMSE**: `0.708233`
- **MAE**: `0.569546`

Diagnostics generated in notebook:
- predicted vs true scatter;
- continuous prediction distribution by true class (boxplot);
- row-normalized confusion matrix on rounded predictions.

Interpretation from those diagnostics:
- predictions track the ordinal trend but with visible dispersion around the diagonal;
- neighboring score levels are frequently mixed (most confusion is between adjacent classes), which is consistent with the QWK objective;
- lower levels (`0-1`) are harder to predict reliably, consistent with their underrepresentation in the label distribution.

**Complementary modeling hypothesis (transformer track).**  
Classical pipelines (handcrafted, TF-IDF, frozen embeddings) mainly capture explicit lexical frequency and engineered signals. A transformer was evaluated to test whether contextual sequence representations from pretrained language modeling improve ordinal vocabulary prediction beyond those classical feature spaces.

---

## 7. Why Transformers Are Separated from Classical Models

The separation is methodological and intentional.

1. **Different compute regime**  
   Classical models support broad 5-fold registry sweeps; transformer full k-fold sweeps are significantly more expensive.

2. **Different optimization structure**  
   Transformer fine-tuning introduces sequence length, batch scheduling, GPU memory constraints, and trainer-specific pruning/storage behavior.

---

## 8. RoBERTa fine-tuning (`BERT_comparison.ipynb`)

### 8.1 Setup and objective

Pretrained **`roberta-base`** was fine-tuned **end-to-end** for scalar regression on the same consensus integer target as the classical track (`target_vocab` from both raters). This means encoder and regression head were optimized jointly (not head-only freezing). Training used **weighted MSE** with **balanced sample weights** on the training fold so rare score levels were not ignored, matching the spirit of the classical CV weighting. Inputs were tokenized with **`MAX_LENGTH = 512`**. The development protocol used a **15% stratified hold-out** (`VAL_FRACTION = 0.15`, merged bins as elsewhere) on the agreement-filtered training file; the **test CSV was never used** for model selection.

### 8.2 Baseline fine-tune vs Optuna search

A **baseline** run was defined as an initial end-to-end fine-tuning recipe using conservative, standard transformer defaults (**learning rate `2e-5`**, **`3` epochs**, **per-device train/eval batch size `8`**, **weight decay `0.01`**), with the best checkpoint selected by **validation QWK** (`load_best_model_at_end`, `metric_for_best_model="qwk"`). That run yielded **validation QWK ≈ 0.5963**.

**Full tuning** kept the same architecture and data split, but replaced fixed baseline settings with a **15-trial Optuna** search over learning rate, epochs, train batch size, weight decay, and warmup ratio; study state was persisted in **`optuna_roberta.db`** and pruning used **MedianPruner** on epoch-level objectives. The **best trial** reached **validation QWK ≈ 0.5681** (hyperparameters logged in notebook output).

Because tuned validation QWK **did not exceed** baseline, **baseline hyperparameters were retained** for final refit (selection by validation QWK, not Optuna trial rank). **Phase B** then retrained **from scratch on the entire agreement-filtered training corpus**, producing the single checkpoint used for test prediction.

### 8.3 Test metrics (held-out test)

| | QWK | RMSE | MAE |
|---|-----|------|-----|
| **RoBERTa (final refit)** | **0.564896** | **0.689487** | **0.529520** |

For comparison, the classical TF-IDF + SVD + MLP test row in **Section 6** reported **QWK 0.532867**, **RMSE 0.708233**, **MAE 0.569546**. On this split, the RoBERTa pipeline improved all three metrics.

**Note:** The summary table in the notebook is indexed as `Test set (RoBERTa, tuned)`; given the selection rule above, the evaluated checkpoint corresponds to the **baseline training recipe** after Phase B full-train refit, not the Optuna-best trial.

Qualitatively, predictions follow the ordinal trend and errors remain mostly local (adjacent levels). Relative to TF-IDF + MLP, mid-range behavior is broadly similar, while the transformer shows clearer improvement at the upper end of the scale: **true levels 4 and 5 align more strongly with the confusion-matrix diagonal**. Low true levels (especially 0-2) remain less separable and are often shifted upward toward middle scores, consistent with class sparsity and rater ambiguity near neighboring bands.

Figure exports for direct inspection of those test diagnostics:
- **Classical (TF-IDF + SVD + MLP):** `Figures/tfidf_mlp_test_scatter.png`, `Figures/tfidf_mlp_test_boxplot.png`, `Figures/tfidf_mlp_test_confusion.png`
- **RoBERTa (final refit):** `Figures/roberta_test_scatter.png`, `Figures/roberta_test_boxplot.png`, `Figures/roberta_test_confusion.png`

---

## 9. Reproducibility Notes

- Seed usage is fixed where practical (`CV_RANDOM_STATE=42` in the classical notebook; `RANDOM_STATE=42` in `BERT_comparison.ipynb`, aligned Optuna sampler where applicable).
- CV metric logic, weighting, and tuning wrappers are centralized in `modeling_utils.py`; transformer training helpers live in `transformers_utils.py`.
- RoBERTa Optuna studies can be resumed from `optuna_roberta.db` when the notebook storage settings are unchanged.
- Trial tables preserve parameter-value and fold-mean metric traces for audit.
- Hardware/library differences (especially the deep-learning stack) may affect bit-level reproducibility.

