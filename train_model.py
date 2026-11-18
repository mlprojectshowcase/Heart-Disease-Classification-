"""
train_model.py
--------------
Run in the terminal:  python train_model.py
Prints the model comparison table and saves charts into 'outputs/' (use them in your report).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ml_pipeline as ml

os.makedirs("outputs", exist_ok=True)
df = ml.load_data()
print("Dataset shape:", df.shape)
print(df[ml.TARGET].value_counts(), "\n")

fitted, results, best, _ = ml.train_and_evaluate(df)

print(f"{'Model':22}{'Acc':>7}{'Prec':>7}{'Recall':>8}{'F1':>7}{'AUC':>7}{'CV Acc':>8}")
for n, r in results.items():
    print(f"{n:22}{r['accuracy']:7.3f}{r['precision']:7.3f}{r['recall']:8.3f}"
          f"{r['f1']:7.3f}{r['roc_auc']:7.3f}{r['cv_accuracy']:8.3f}")
print("\nBest model:", best)

# Chart 1: confusion matrix
cm = results[best]["confusion"]
fig, ax = plt.subplots(figsize=(4.5, 4))
ax.imshow(cm, cmap="Reds")
ax.set_xticks([0, 1]); ax.set_xticklabels(ml.LABELS)
ax.set_yticks([0, 1]); ax.set_yticklabels(ml.LABELS)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion Matrix - {best}")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
fig.tight_layout(); fig.savefig("outputs/confusion_matrix.png", dpi=150)

# Chart 2: model comparison (F1 and recall)
fig, ax = plt.subplots(figsize=(7, 3.8))
names = list(results)
x = range(len(names))
ax.bar([i - 0.2 for i in x], [results[n]["recall"] for n in names], 0.4, label="Recall", color="#E45756")
ax.bar([i + 0.2 for i in x], [results[n]["f1"] for n in names], 0.4, label="F1-score", color="#4C78A8")
ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=10)
ax.set_ylim(0, 1); ax.set_title("Model Comparison"); ax.legend()
fig.tight_layout(); fig.savefig("outputs/model_comparison.png", dpi=150)

# Chart 3: feature importance (Random Forest)
imp = ml.feature_importance(fitted["Random Forest"])
fig, ax = plt.subplots(figsize=(6, 5))
imp.sort_values().plot.barh(ax=ax, color="#F58518")
ax.set_title("Feature Importance (Random Forest)")
fig.tight_layout(); fig.savefig("outputs/feature_importance.png", dpi=150)
print("Charts saved in outputs/")
