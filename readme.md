# Telco Customer Churn Prediction

This project predicts customer churn probability and explains the factors that
drive risk. It uses a leakage-safe scikit-learn pipeline, stratified
cross-validation, recall-focused threshold tuning, and model explainability.

## Quick setup and run

Open a terminal, clone the repository, enter its folder, create a virtual
environment, install the dependencies, and run the training script:

```powershell
git clone <repository-url>
cd Customer-Churn-Prediction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python .\train_churn.py
```

To open and run the notebook:

```powershell
jupyter notebook .\notebooks\churn.ipynb
```

If PowerShell blocks virtual-environment activation, run the following once
for the current user, then activate the environment again:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

The script downloads the dataset automatically. It first tries
`load_dataset("scikit-learn/churn-prediction")` and falls back to KaggleHub.
No local dataset or machine-specific path is required.

## Deliverables

- `notebooks/churn.ipynb`: end-to-end notebook with loading, cleaning, EDA,
  three classifiers, evaluation, threshold tuning, coefficients/importances,
  and a SHAP summary.
- `train_churn.py`: executable training workflow.
- `requirements.txt`: dependencies.
- `models/best_model.joblib`: generated after training.

The fallback Kaggle source is:

```python
import kagglehub
path = kagglehub.dataset_download("blastchar/telco-customer-churn")
```

The saved joblib contains the fitted pipeline, selected decision threshold,
and feature column order. Use `predict_proba` for a churn probability.

## Retention recommendations

The notebook measures these recommendations rather than assuming them:

1. Offer annual or two-year upgrades to month-to-month customers.
2. Provide onboarding check-ins and early-tenure service credits.
3. Bundle technical support and online security with fiber plans.
4. Offer simpler payment choices and proactive bill explanations.