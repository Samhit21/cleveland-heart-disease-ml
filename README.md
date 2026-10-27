# Cleveland Heart Disease Risk Modeling

A classical machine-learning project that compares Logistic Regression, Random Forest, and Gradient Boosting on the Cleveland Heart Disease dataset.

## What this project demonstrates

- Exploratory data analysis with pandas, matplotlib, and seaborn
- Train/test splitting with stratification
- Numeric and categorical preprocessing with scikit-learn pipelines
- Logistic Regression, Random Forest, and Gradient Boosting
- Evaluation with accuracy, precision, recall, F1 score, ROC-AUC, confusion matrices, and classification reports
- Side-by-side model comparison
- Example probability-based prediction

## Project structure

```text
.
├── cleveland_heart_disease_ml.ipynb
├── Heart_disease_cleveland_new.csv
├── requirements.txt
└── README.md
```

The CSV is not included in this repository package. Place your copy of `Heart_disease_cleveland_new.csv` in the project root before running the notebook.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Then open `cleveland_heart_disease_ml.ipynb` and run the cells from top to bottom.

## Models

### Logistic Regression
Used as an interpretable linear baseline.

### Random Forest
Uses an ensemble of decision trees to model nonlinear relationships and interactions.

### Gradient Boosting
Builds trees sequentially so later trees focus on errors made by earlier ones.

## Evaluation

The notebook evaluates each model using:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- Confusion matrix
- Classification report

For this type of risk-classification problem, recall is particularly useful because false negatives are important to examine. The project still compares multiple metrics rather than optimizing for a single score.

## Notes on the original experiment

In the original notebook run, Logistic Regression achieved about 0.89 accuracy and 0.97 ROC-AUC on the held-out test set. Random Forest and Gradient Boosting produced similar classification performance. The cleaned notebook recalculates every model's metrics independently to avoid variable reuse between models.

## Limitations

This is an educational machine-learning project, not a clinical diagnostic system. The dataset is small, and a single train/test split is not enough to establish real-world medical performance. Cross-validation, hyperparameter tuning, calibration, and external validation would be appropriate next steps.

## Tech stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Jupyter
