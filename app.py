"""
app.py  -  Streamlit web app for "Heart Disease Classification"
Run locally:  streamlit run app.py
EDUCATIONAL PROJECT ONLY - not a medical tool.
"""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import ml_pipeline as ml

st.set_page_config(page_title="Heart Disease Classification", page_icon="❤️", layout="wide")


@st.cache_data
def get_data():
    return ml.load_data()


@st.cache_resource
def get_models():
    """Trains once when the app starts (a few seconds), then reuses the result."""
    return ml.train_and_evaluate(get_data())


df = get_data()
fitted, results, best, _ = get_models()

st.sidebar.title("❤️ Navigation")
page = st.sidebar.radio("Go to", ["1. Project Overview", "2. Dataset Explorer",
                                  "3. Model Results", "4. Check a Patient"])
st.sidebar.warning("Educational project using synthetic data. Not for real medical decisions.")

# ------------------------------------------------------------------ PAGE 1
if page.startswith("1"):
    st.title("❤️ Heart Disease Classification")
    st.write("""
**Goal:** predict whether a patient is likely to have **heart disease (Yes / No)** from
medical test results such as age, blood pressure, cholesterol and ECG findings.

**Type of problem:** *Supervised binary classification* - there are only two possible answers.
The computer learns from past patients whose diagnosis is known, then predicts for a new patient.

**Workflow used in this project**
1. Create / load the dataset
2. Pre-process the data (scale numbers, convert text to numbers)
3. Split into 80 % training and 20 % testing data
4. Train four algorithms: Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors
5. Compare them using accuracy, precision, recall, F1-score and ROC-AUC
6. Use the best model to check new patients

**Why recall matters here:** missing a sick patient (a *false negative*) is more dangerous than
a false alarm, so in medical projects we watch **recall** closely, not just accuracy.
""")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients in dataset", len(df))
    c2.metric("Input features", len(ml.NUMERIC) + len(ml.CATEGORICAL))
    c3.metric("Best model", best)
    c4.metric("F1 of best model", f"{results[best]['f1']:.3f}")
    st.info("The dataset is synthetic (artificially generated) because real patient records are "
            "private. See the README for details.")

# ------------------------------------------------------------------ PAGE 2
elif page.startswith("2"):
    st.title("📊 Dataset Explorer")
    st.subheader("First 10 rows")
    st.dataframe(df.head(10), use_container_width=True)
    st.subheader("Summary statistics")
    st.dataframe(df.describe().round(2), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Class distribution")
        counts = df[ml.TARGET].map({0: "No Disease", 1: "Disease"}).value_counts()
        st.bar_chart(counts)
    with col2:
        st.subheader("Average of a feature per class")
        feat = st.selectbox("Choose a feature", ml.NUMERIC)
        st.bar_chart(df.groupby(df[ml.TARGET].map({0: "No Disease", 1: "Disease"}))[feat].mean())

    st.subheader("Disease rate by a category")
    cat = st.selectbox("Choose a category", ml.CATEGORICAL)
    st.bar_chart(df.groupby(cat)[ml.TARGET].mean().rename("Share with disease"))

    st.subheader("Correlation between numeric columns")
    cols = ml.NUMERIC + [ml.TARGET]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=8)
    fig.colorbar(im); fig.tight_layout()
    st.pyplot(fig)

# ------------------------------------------------------------------ PAGE 3
elif page.startswith("3"):
    st.title("🏆 Model Results")
    table = pd.DataFrame({
        n: {"Accuracy": r["accuracy"], "Precision": r["precision"], "Recall": r["recall"],
            "F1-score": r["f1"], "ROC-AUC": r["roc_auc"], "5-fold CV accuracy": r["cv_accuracy"]}
        for n, r in results.items()
    }).T.round(3)
    st.dataframe(table, use_container_width=True)
    st.success(f"Best model (highest F1-score): **{best}**")

    with st.expander("What do these metrics mean?"):
        st.write("""
- **Accuracy** - out of all test patients, how many were classified correctly.
- **Precision** - when the model says "disease", how often is it right.
- **Recall** - out of all patients who really have disease, how many did the model find. *Very important in medicine.*
- **F1-score** - balance of precision and recall.
- **ROC-AUC** - how well the model separates sick from healthy patients (0.5 = guessing, 1.0 = perfect).
- **Cross-validation** - training/testing repeated 5 times on different parts of the data to check the result is stable.
""")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Confusion matrix ({best})")
        cm = results[best]["confusion"]
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.imshow(cm, cmap="Reds")
        ax.set_xticks([0, 1]); ax.set_xticklabels(ml.LABELS)
        ax.set_yticks([0, 1]); ax.set_yticklabels(ml.LABELS)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
        fig.tight_layout()
        st.pyplot(fig)
        st.caption("Top-left and bottom-right = correct. Bottom-left = missed sick patients (false negatives).")
    with col2:
        st.subheader("Feature importance (Random Forest)")
        st.bar_chart(ml.feature_importance(fitted["Random Forest"]))
        st.caption("Longer bar = that input influences the prediction more.")

# ------------------------------------------------------------------ PAGE 4
else:
    st.title("🩺 Check a Patient")
    st.write(f"Enter the details. Prediction is made by the best model: **{best}**.")
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age", 25, 80, 54)
            sex = st.selectbox("Sex", ["Male", "Female"])
            cp = st.selectbox("Chest pain type",
                              ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "No Symptoms"])
            bp = st.slider("Resting blood pressure (mm Hg)", 90, 200, 130)
        with c2:
            chol = st.slider("Cholesterol (mg/dl)", 120, 450, 240)
            fbs = st.selectbox("Fasting blood sugar above 120 mg/dl", ["No", "Yes"])
            ecg = st.selectbox("Resting ECG", ["Normal", "ST-T Abnormality", "LV Hypertrophy"])
        with c3:
            hr = st.slider("Maximum heart rate achieved", 70, 202, 150)
            ang = st.selectbox("Exercise-induced angina", ["No", "Yes"])
            old = st.slider("ST depression (oldpeak)", 0.0, 6.0, 1.0, 0.1)
            slope = st.selectbox("ST slope", ["Upsloping", "Flat", "Downsloping"])
        go = st.form_submit_button("Check")

    if go:
        row = pd.DataFrame([{
            "age": age, "sex": sex, "chest_pain_type": cp, "resting_bp": bp,
            "cholesterol": chol, "fasting_blood_sugar_high": fbs, "resting_ecg": ecg,
            "max_heart_rate": hr, "exercise_induced_angina": ang,
            "oldpeak_st_depression": old, "st_slope": slope}])
        p = float(fitted[best].predict_proba(row)[0][1])
        if p >= 0.5:
            st.error(f"Model result: **higher risk of heart disease** (probability {p:.0%})")
        else:
            st.success(f"Model result: **lower risk of heart disease** (probability {p:.0%})")
        st.progress(p)
        st.caption("This is a classroom demonstration on synthetic data. It is NOT a diagnosis; "
                   "anyone worried about their heart should see a doctor.")
