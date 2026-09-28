# Telco Churn Data Dictionary

Source: public IBM Telco Customer Churn CSV placed locally at `data/telco_churn.csv`. The source has 7,043 rows and 21 columns. `customerID` is an identifier, not a predictive feature. The model accepts the remaining 19 raw predictor fields and derives three row-level features. The target is `Churn`.

Numeric columns use median imputation and standard scaling inside the fitted training pipeline. Categorical columns use most-frequent imputation and one-hot encoding with unknown categories ignored by the transformer; inference validation still rejects unsupported source categories.

| Field | Meaning | Type / examples | Processing | Model use |
| --- | --- | --- | --- | --- |
| `customerID` | Source account key | string, e.g. `0001-ABCD` | Excluded from predictors; optional output key only | No |
| `gender` | Reported gender category | categorical: Female, Male | Most-frequent imputation; one-hot | Yes |
| `SeniorCitizen` | Senior-citizen indicator | binary integer: 0, 1 | Median imputation; standard scaling | Yes |
| `Partner` | Has a partner | Yes / No | Most-frequent imputation; one-hot | Yes |
| `Dependents` | Has dependents | Yes / No | Most-frequent imputation; one-hot | Yes |
| `tenure` | Months as a customer | integer, 0–72 | Median imputation; standard scaling | Yes |
| `PhoneService` | Subscribes to phone service | Yes / No | Most-frequent imputation; one-hot | Yes |
| `MultipleLines` | Multiple-lines service | Yes / No / No phone service | Most-frequent imputation; one-hot | Yes |
| `InternetService` | Internet connection type | DSL / Fiber optic / No | Most-frequent imputation; one-hot | Yes |
| `OnlineSecurity` | Online security subscription | Yes / No / No internet service | Most-frequent imputation; one-hot | Yes |
| `OnlineBackup` | Online backup subscription | Yes / No / No internet service | Most-frequent imputation; one-hot | Yes |
| `DeviceProtection` | Device protection subscription | Yes / No / No internet service | Most-frequent imputation; one-hot | Yes |
| `TechSupport` | Technical support subscription | Yes / No / No internet service | Most-frequent imputation; one-hot | Yes |
| `StreamingTV` | Streaming TV subscription | Yes / No / No internet service | Most-frequent imputation; one-hot | Yes |
| `StreamingMovies` | Streaming movie subscription | Yes / No / No internet service | Most-frequent imputation; one-hot | Yes |
| `Contract` | Contract duration | Month-to-month / One year / Two year | Most-frequent imputation; one-hot | Yes |
| `PaperlessBilling` | Paperless billing choice | Yes / No | Most-frequent imputation; one-hot | Yes |
| `PaymentMethod` | Payment channel | Electronic check, mailed check, automatic transfer/card | Most-frequent imputation; one-hot | Yes |
| `MonthlyCharges` | Current monthly bill | numeric currency amount | Median imputation; standard scaling | Yes |
| `TotalCharges` | Accumulated bill amount | numeric currency amount; blanks may occur | Blank strings become missing; median imputation; standard scaling | Yes |
| `Churn` | Whether the customer left | Yes / No, encoded 1 / 0 | Target only; excluded from predictors | Target |
| `tenure_group` | Coarse tenure segment | 0–12 / 13–24 / 25–48 / 49–72 | Derived per row from `tenure`; one-hot | Yes |
| `service_count` | Number of affirmative optional services | integer, 0–6 | Derived by counting `Yes` across security, backup, protection, support, TV, and movie fields | Yes |
| `average_monthly_spend` | Average accumulated charges per active month | numeric currency amount | `TotalCharges / tenure`; missing for zero tenure or missing total; median-imputed and scaled | Yes |

The dataset does not establish causal effects. Category frequencies and charge distributions reflect this source and collection period; they should not be treated as current telecom population benchmarks.
