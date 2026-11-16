# Four-page executive summary — suggested content plan

## Pages 1–2: Main text
1. **Title and research question** — concise statement of the fraud-detection problem.
2. **Motivation** — why rare fraud events make evaluation difficult; why validation design matters.
3. **Dataset** — source, number of rows, target definition, fraud prevalence, features, missingness and duplicates as observed in your run.
4. **Preprocessing** — duplicate handling, feature selection, imputation/scaling within pipelines, and why `Time` is reserved for ordering rather than used as a predictor.
5. **Models** — justify Logistic Regression as an interpretable linear baseline and Random Forest as a nonlinear tree ensemble. Explain class weighting.
6. **Methodological experiment** — hypothesis: a random split may give a different (potentially more optimistic) estimate than a chronological split because random splitting mixes earlier and later observations. State the reason before presenting results.
7. **Evaluation** — prioritise average precision / PR-AUC, precision, recall and F1; include ROC-AUC as a secondary metric. Explain why accuracy is insufficient.
8. **Results and interpretation** — insert only actual results from `outputs/model_metrics.csv`; compare the two split strategies and discuss test fraud counts.
9. **Limitations and conclusion** — two-day period, anonymised PCA variables, age of data, generalisability, temporal split limitations and implications.

## Page 3: Figures
Suggested figures:
- Class distribution (log scale).
- Fraud prevalence across chronological time bins.
- Precision-recall or confusion matrix for the main temporal test.
- Split-strategy PR-AUC comparison.

## Page 4
- References actually accessed and read.
- Brief Generative AI use statement naming tools and describing their material contribution.
- Presentation URL if prerecorded.

Do not write conclusions or numerical findings until you have run the analysis.
