# ❤️ Heart Disease Classification

A beginner-friendly **Machine Learning** project that predicts whether a patient is likely to have
**heart disease (Yes / No)** from medical test results, with a web app built using **Streamlit**.

> ⚠️ **Educational project.** It uses synthetic data and is NOT a medical tool or a diagnosis.

## 1. Problem Statement
Heart disease is a leading cause of death, and early detection saves lives. Doctors use tests such as
blood pressure, cholesterol and ECG. We build a model that learns patterns from past patients and
flags new patients who may be at higher risk.

- **Type of ML:** Supervised learning -> **Binary classification** (two classes: 0 = No Disease, 1 = Disease)
- **Input (X):** 11 features  |  **Output (y):** `heart_disease`

## 2. Dataset
`data/heart_data.csv` - 1000 patients, 12 columns.
The data is **synthetic** (created by `generate_dataset.py`) because real patient records are private.
Columns follow the well-known UCI "Heart Disease" dataset, and the chance of disease rises with medically
sensible risk factors (age, male sex, blood pressure, cholesterol, exercise angina, ST depression, ...)
plus random noise. You may replace the CSV with the real UCI / Kaggle dataset (keep the column names).

| Column | Meaning |
|---|---|
| age | age in years |
| sex | Male / Female |
| chest_pain_type | Typical Angina / Atypical Angina / Non-anginal Pain / No Symptoms |
| resting_bp | resting blood pressure (mm Hg) |
| cholesterol | serum cholesterol (mg/dl) |
| fasting_blood_sugar_high | fasting sugar above 120 mg/dl: Yes / No |
| resting_ecg | Normal / ST-T Abnormality / LV Hypertrophy |
| max_heart_rate | highest heart rate reached in exercise test |
| exercise_induced_angina | chest pain during exercise: Yes / No |
| oldpeak_st_depression | ST depression on ECG after exercise |
| st_slope | Upsloping / Flat / Downsloping |
| **heart_disease** | **Target:** 1 = disease, 0 = no disease |

## 3. Methodology
1. **Load data** with pandas (the code regenerates the data automatically if the CSV is missing).
2. **Pre-processing:** StandardScaler for numbers, OneHotEncoder for text columns, inside a scikit-learn `Pipeline`.
3. **Train/Test split:** 80% / 20% (stratified).
4. **Algorithms compared:** Logistic Regression, Decision Tree (depth 4), Random Forest (200 trees), K-Nearest Neighbors (k = 15).
5. **Evaluation:** Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix, 5-fold Cross-Validation.
6. **Best model** = highest F1-score; it is used in the web app.

## 4. Results (on the included dataset)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.715** | 0.736 | 0.743 | **0.740** | **0.783** |
| Decision Tree | 0.650 | 0.642 | 0.807 | 0.715 | 0.735 |
| Random Forest | 0.695 | 0.718 | 0.725 | 0.721 | 0.753 |
| K-Nearest Neighbors | 0.675 | 0.690 | 0.734 | 0.711 | 0.742 |

**Notes for discussion**
- Accuracy is around 70% because the synthetic data deliberately contains random noise, like real
  patients who do not follow neat rules. With the real UCI dataset the numbers will differ.
- **Recall matters most in medicine:** a missed sick patient (false negative) is worse than a false alarm.
  The Decision Tree has the highest recall (0.807) but makes more false alarms, so Logistic Regression
  is chosen as the best overall balance.
- Run `python train_model.py` to reproduce these numbers and save charts into `outputs/`.

## 5. Project Structure
```
├── app.py                # Streamlit web app (4 pages)
├── ml_pipeline.py        # All ML code: load, preprocess, train, evaluate
├── train_model.py        # Terminal run: prints results + saves charts to outputs/
├── generate_dataset.py   # Creates the synthetic dataset
├── data/heart_data.csv   # Dataset
├── outputs/              # Charts for the report
├── requirements.txt      # Libraries needed
└── README.md
```

## 6. Run on your computer
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## 7. Deploy FREE on Streamlit Community Cloud
1. Create a **new public repository** on GitHub (https://github.com/new), e.g. `heartdiseaseclassification`.
2. Click **Add file -> Upload files**, drag in all files and folders, choose **Commit directly to the main branch**, click **Commit changes**.
   `app.py` and `requirements.txt` must be in the main (root) folder of the repo.
3. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**.
4. Select the repository, branch `main`, main file path `app.py`, then **Deploy**.

## 8. Limitations and Future Work
- Synthetic data: try the real UCI Heart Disease dataset.
- Try hyper-parameter tuning, more algorithms (XGBoost), and class-weight tuning to raise recall.
- Predictions are a classroom demonstration, never a medical diagnosis.
