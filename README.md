# Small-scale Farmer Resilience Capacity: Explainable Boosting Machine

An MSc in AI/ML project that predicts whether Sri Lankan smallholder farmers have **High or Low resilience capacity** using socio-economic, farming challenge, and risk-management survey data. The project uses an Explainable Boosting Machine (EBM) to provide interpretable predictions.

## Project Structure

- `notebooks/ML_Assignment_EBM.ipynb` – Data preprocessing, model training, evaluation, and explainability.
- `app.py` – Streamlit application for predictions and individual explanations.
- `model/` – Trained EBM model and input schema.
- `results/` – Evaluation results, tables, and figures.

## Dataset

**Source:** [Mendeley Data – Challenges and Resilience Capacities of Smallholder Farmers in Sri Lanka](https://data.mendeley.com/)

- 300 farmers
- 72 input features
- Target: High or Low resilience capacity
- Public, anonymized survey data

Download the dataset in Excel format from Mendeley Data before running the notebook.

## Models Evaluated

- Logistic Regression
- Random Forest
- Gradient Boosting
- Explainable Boosting Machine (EBM)

EBM was selected for its interpretability, including feature contributions and individual prediction explanations.

## Test Results

| Model | Accuracy | F1 (Macro) | AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.600 | 0.598 | 0.592 |
| Random Forest | 0.667 | 0.666 | 0.742 |
| Gradient Boosting | 0.667 | 0.663 | 0.659 |
| EBM | 0.633 | 0.627 | 0.659 |

## Run the Notebook

Open `notebooks/ML_Assignment_EBM.ipynb` in Google Colab, run the cells, and upload the dataset when prompted.

## Run the Application

```bash
pip install -r requirements.txt
streamlit run app.py
```

