# Model Card: Telecom Customer Churn Classifier

## Model identity

- **Model version:** 2.0.0
- **Feature schema version:** 1.0.0
- **Algorithm:** XGBoost classifier in an sklearn preprocessing pipeline
- **Training date:** 2026-09-27
- **Artifact metadata:** `models/model_metadata.json`
- **Dataset fingerprint:** SHA-256 is stored in model metadata
- **Model selection:** highest mean five-fold training PR-AUC from a bounded, seeded randomized search

## Purpose and intended use

Demonstration and portfolio use: rank customers for human review or retention-outreach planning in the IBM Telco churn dataset. Outputs are model estimates and should be reviewed with current data and business policy before any operational use.

## Uses that are not appropriate

- Automatically cancel, restrict, price, or deny service to a customer
- Treat a risk score as a causal explanation or guaranteed churn event
- Claim that a retention offer will reduce churn based on this model alone
- Present estimated revenue exposure as certain loss or savings
- Use as a production service without security, load, fairness, and prospective validation

## Data and target

Training uses the supplied public IBM Telco Customer Churn CSV (7,043 rows, 21 source columns). `Churn` is the binary target (Yes = 1, No = 0); `customerID` is excluded. The 19 raw account/service/charge predictors and three row-engineered features are documented in [`data_dictionary.md`](data_dictionary.md). The dataset reflects a historical collection and may not represent current customers, products, or prices.

## Training and imbalance strategy

The stratified 20% test partition is held untouched until final evaluation. Hyperparameter search uses the remaining training partition and five-fold stratified CV. Numeric imputation/scaling and categorical imputation/one-hot encoding are fitted within each fold. Candidates use class weights or XGBoost `scale_pos_weight`. Vanilla SMOTE on one-hot values is not used because it can generate fractional category indicators.

## Performance

Metrics below are from one stratified test split, at the configured threshold 0.50:

| Metric | Test result |
| --- | ---: |
| Accuracy | 0.7559 |
| Precision | 0.5275 |
| Recall | 0.7701 |
| F1 | 0.6261 |
| ROC-AUC | 0.8428 |
| PR-AUC / average precision | 0.6513 |
| Brier score (lower is better) | 0.1595 |

Selected model mean five-fold training PR-AUC was 0.6658 (fold standard deviation 0.0227); mean ROC-AUC was 0.8459. The test PR-AUC is compared with the churn prevalence baseline and other candidate metrics in `reports/model_results.csv`. Calibration curve and Brier score evaluate probability quality but do not establish reliable calibration for future populations.

## Threshold and business metrics

The default threshold is 0.50 and is configurable with `CHURN_THRESHOLD`. It was not optimized because outreach costs, offer costs, and customer value were not provided. The OOF threshold report is `reports/threshold_analysis.csv`; test-set ranking results are `reports/top_k_lift_analysis.csv`.

At Top 10% on this test split, 141 customers were targeted, 105 actual churners were captured, precision was 74.5%, recall was 28.1%, and lift was 2.81×. `MonthlyCharges × predicted churn probability` is reported as **model-based estimated revenue exposure**, not causal loss or achievable savings.

For this held-out split, the score-weighted monthly exposure estimate was `$40,090.03`. It is an aggregate descriptive estimate under the formula above, not the amount expected to be lost.

## Explainability

Global SHAP summary and mean absolute contributions are saved under `reports/figures/shap_summary.png` and `reports/figures/shap_global_importance.csv`. The dashboard computes local positive/negative SHAP contributions for the selected input. SHAP explains fitted model behavior, not causal drivers.

## Risks and limitations

- Public, historical, single-source data can be biased or out of date.
- Group-specific errors, fairness, and subgroup calibration require review before use.
- The chosen score ranking is close to other CV candidates; CV uncertainty should be considered.
- No prospective outcomes, intervention data, costs, or causal experiment are available.
- Prediction logging tracks scores and selected input fields, not ground-truth outcomes.
- PSI is a heuristic drift flag; it does not measure model-performance degradation.

## Monitoring and maintenance

Versioned metadata records dataset hash, feature list, training configuration, best parameters, cross-validation summary, threshold source, and held-out metrics. Review input drift summaries and retrain only with refreshed, validated data and a new held-out evaluation. Add outcome labels before reporting live precision, recall, calibration, or other true performance metrics.
