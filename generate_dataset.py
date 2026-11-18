"""
generate_dataset.py
-------------------
Creates a SYNTHETIC (artificial) patient dataset of 1000 people and saves it to
data/heart_data.csv.

Why synthetic?  Real medical records are private. The columns here are inspired by the
well-known UCI "Heart Disease" dataset, and the chance of heart disease is calculated from
medically sensible risk factors (older age, male sex, high blood pressure, high cholesterol,
chest pain on exercise, low maximum heart rate, ST depression ...) plus randomness.

You can replace data/heart_data.csv with the real UCI / Kaggle dataset
(keep the same column names) and the whole project still works.
Run:  python generate_dataset.py
"""
import numpy as np
import pandas as pd


def make_dataframe(n=1000, seed=42):
    """Builds the synthetic dataset and returns it as a DataFrame."""
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(54, 9, n), 29, 77).round(0)
    sex = rng.choice(["Male", "Female"], n, p=[0.65, 0.35])
    chest_pain = rng.choice(
        ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "No Symptoms"],
        n, p=[0.12, 0.20, 0.28, 0.40])
    resting_bp = np.clip(rng.normal(130, 17, n), 90, 200).round(0)          # mm Hg
    cholesterol = np.clip(rng.normal(245, 50, n), 120, 450).round(0)        # mg/dl
    fasting_bs = rng.choice(["No", "Yes"], n, p=[0.85, 0.15])               # sugar > 120 mg/dl
    rest_ecg = rng.choice(["Normal", "ST-T Abnormality", "LV Hypertrophy"], n, p=[0.55, 0.30, 0.15])
    max_hr = np.clip(205 - 0.9 * age + rng.normal(0, 17, n), 70, 202).round(0)
    exercise_angina = rng.choice(["No", "Yes"], n, p=[0.68, 0.32])
    oldpeak = np.clip(rng.exponential(1.0, n), 0, 6).round(1)               # ST depression
    st_slope = rng.choice(["Upsloping", "Flat", "Downsloping"], n, p=[0.45, 0.40, 0.15])

    # ---- hidden risk formula (log-odds) ----
    z = (
        -9.0
        + 0.045 * age
        + 0.9 * (sex == "Male")
        + np.select([chest_pain == "No Symptoms", chest_pain == "Typical Angina"], [1.6, -0.3], 0.3)
        + 0.015 * resting_bp
        + 0.006 * cholesterol
        + 0.5 * (fasting_bs == "Yes")
        + 0.3 * (rest_ecg != "Normal")
        - 0.020 * max_hr
        + 1.2 * (exercise_angina == "Yes")
        + 0.6 * oldpeak
        + np.select([st_slope == "Flat", st_slope == "Downsloping"], [0.8, 1.2], 0.0)
        + rng.normal(0, 0.8, n)                                             # randomness
    )
    prob = 1 / (1 + np.exp(-(z - np.median(z) + 0.2)))                      # centre -> balanced classes
    heart_disease = (rng.random(n) < prob).astype(int)

    return pd.DataFrame({
        "age": age, "sex": sex, "chest_pain_type": chest_pain, "resting_bp": resting_bp,
        "cholesterol": cholesterol, "fasting_blood_sugar_high": fasting_bs,
        "resting_ecg": rest_ecg, "max_heart_rate": max_hr,
        "exercise_induced_angina": exercise_angina, "oldpeak_st_depression": oldpeak,
        "st_slope": st_slope, "heart_disease": heart_disease,
    })


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    df = make_dataframe()
    df.to_csv("data/heart_data.csv", index=False)
    print("Saved data/heart_data.csv ->", df.shape)
    print(df["heart_disease"].value_counts())
