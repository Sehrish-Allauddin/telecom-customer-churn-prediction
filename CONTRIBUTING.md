# Contributing

## Branches

Use short descriptive branches such as `feature/api-health` or `fix/input-validation`. Start from the default branch and keep each branch focused.

## Code standards

Use Python 3.10+, type hints for public functions, clear docstrings, and the existing preprocessing/model contracts. The current development policy is a stratified train/test split, five-fold training CV, class-weighted imbalance handling, and PR-AUC model selection. Document and test changes to these choices. Do not fit preprocessing on held-out rows or use the final test set to tune.

## Tests and checks

Install `requirements-dev.txt`, then run:

```bash
python -m pytest -q
python -m pytest -q --cov=src --cov=api --cov-report=term-missing
ruff check .
black --check .
isort --check-only .
```

For formatting, run `black .` and `isort .`. Tests should use synthetic data and must not require Kaggle, credentials, or internet access.

## Pull requests

Open a pull request with a concise summary, motivation, verification results, and any model-behavior impact. Include updated documentation and screenshots for dashboard changes when useful. Do not include `.env`, private data, customer identifiers, or unreviewed model artifacts.

## Commits

Use imperative, specific commit subjects such as `Add input schema validation`. Keep generated reports and unrelated formatting changes out of focused commits.
