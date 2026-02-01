# Medication Adherence Prediction Using Machine Learning

A machine learning framework for predicting medication non-adherence among chronic disease patients in Ghana using structured electronic medical record (EMR) data.

## Overview

This project develops an interpretable, context-specific ML model to identify patients at risk of medication non-adherence. The work addresses a critical gap in healthcare AI: most adherence prediction models are developed in high-income settings and fail to generalize to low- and middle-income countries (LMICs) where healthcare financing, access patterns, and socioeconomic factors differ substantially.

### Key Results

| Metric | Score |
|--------|-------|
| Accuracy | 89.5% |
| AUC-ROC | 0.934 |
| F1-Score | 0.93 |
| Recall (Non-adherent) | 77% |
| Recall (Adherent) | 94% |

### Top Predictive Features (SHAP Analysis)

1. **Insurance Status (NHIS)** — Strongest predictor of adherence
2. **Age × Medication Complexity** — Interaction effect capturing polypharmacy burden in older patients
3. **Chronic Comorbidity Status** — Presence of conditions like diabetes, heart failure
4. **Age** — Independent effect on adherence behavior
5. **Total Medication Count** — Regimen complexity
6. **Estimated Medication Cost** — Financial burden proxy
7. **Cost Burden × Insurance** — Interaction capturing uninsured cost sensitivity

## Methodology

### Data Source
- Structured EMR data from Presbyterian Hospital, Agogo, Ghana
- 1,367 adult patients with chronic conditions (hypertension, diabetes, cardiovascular disease)
- 6-month observation window for adherence measurement

### Adherence Measurement
- **Proportion of Days Covered (PDC)** calculated from prescription refill patterns
- Adherent: PDC ≥ 80% | Non-adherent: PDC < 80%
- Class distribution: 74.4% adherent, 25.6% non-adherent

### Models Evaluated

**Baseline Models:**
- Logistic Regression
- Random Forest
- Support Vector Machine (RBF kernel)
- XGBoost
- Multi-Layer Perceptron (MLP)

**Ensemble Methods:**
- Voting Classifier (soft voting)
- Stacking Classifier (final model)

### Domain-Informed Feature Engineering

Custom interaction features designed to capture Ghana-specific healthcare realities:

```python
# Interaction features
age_meds_interaction = age_scaled × total_medication_count
price_burden_uninsured = estimated_price_scaled × (1 - insurance_status)
multi_morbidity_cost = has_chronic_comorbidity × estimated_price_scaled
gender_price_interaction = gender_encoded × estimated_price_scaled
age_chronic_combo = age_scaled × has_chronic_comorbidity
insurance_meds = insurance_status × total_medication_count
```

### Model Interpretability

SHAP (Shapley Additive Explanations) applied for:
- **Global interpretability**: Feature importance rankings across the dataset
- **Local interpretability**: Patient-level prediction explanations

### Ablation Studies

Systematic feature reduction to identify minimum viable feature sets:

| Features | Accuracy | AUC-ROC | Non-Adherent Recall |
|----------|----------|---------|---------------------|
| 12 (full) | 87.3% | 0.933 | 68% |
| 7 | 87.6% | 0.935 | 69% |
| 5 | 86.2% | 0.912 | 63% |
| 3 | 84.5% | 0.878 | 58% |

The 7-feature model achieves optimal performance, suggesting the bottom 5 features introduce noise.

## Repository Structure

```
├── README.md
├── requirements.txt
├── notebooks/
│   └── medication_adherence_prediction.ipynb    # Main analysis notebook
├── src/
│   ├── preprocessing.py      # Data cleaning and feature engineering
│   ├── models.py             # Model training and evaluation
│   └── interpretability.py   # SHAP analysis utilities
├── data/
│   └── README.md             # Data access information (data not included)
└── app/
    └── streamlit_app.py      # Clinical decision support prototype
```

## Installation

```bash
# Clone repository
git clone https://github.com/[username]/medication-adherence-prediction.git
cd medication-adherence-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
shap>=0.41.0
matplotlib>=3.6.0
seaborn>=0.12.0
streamlit>=1.20.0
```

## Usage

### Running the Analysis

```python
# Load and preprocess data
import pandas as pd
from src.preprocessing import preprocess_data, add_interactions

df = pd.read_csv('data/patient_records.csv')
X, y = preprocess_data(df)
X = add_interactions(X)

# Train stacked ensemble
from src.models import train_stacked_ensemble

model = train_stacked_ensemble(X_train, y_train)
predictions = model.predict(X_test)
```

### SHAP Interpretation

```python
import shap

explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# Global feature importance
shap.summary_plot(shap_values, X_test)

# Individual patient explanation
shap.waterfall_plot(shap_values[0])
```

## Clinical Application

A Streamlit-based prototype demonstrates real-world applicability:

- Input patient demographics, clinical data, and medication information
- Receive real-time adherence risk prediction with probability score
- View SHAP-based explanation of risk factors
- Adjustable decision threshold for sensitivity/specificity trade-offs

```bash
streamlit run app/streamlit_app.py
```

## Limitations

- Single-site data (Presbyterian Hospital, Agogo) — external validation needed
- Structured EMR data only — does not capture behavioral, psychosocial, or environmental factors
- Retrospective design — prospective validation recommended before clinical deployment

## Citation

This codebase accompanies the following publication:

```
Danso, S., Nyantakyi, I., Bempah, A.K., Boateng, E.K., Arku, F., Bonsu-Duah, S., 
& Danso, D. (2025). Predicting Medication Adherence Among Chronic Disease Patients 
in Ghana Using Machine Learning: A Data-Driven Approach. International Conference 
on Image Processing and Artificial Intelligence (ICIPAI-26), San Francisco, USA.
```

The methodology and codebase were developed as part of Afriyie Karikari Bempah's Master's thesis at Ghana Communication Technology University.

## Author

**Afriyie Karikari Bempah**  
PharmD | MSc Computer Science | MSc Finance  
Email: geniuskarikari@gmail.com  
LinkedIn: [linkedin.com/in/afriyiekarikaribempah](https://linkedin.com/in/afriyiekarikaribempah)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Presbyterian Hospital, Agogo for data access
- Dr. Emmanuel Freeman and Dr. Samuel Danso (thesis supervisors)
- Ghana Communication Technology University
