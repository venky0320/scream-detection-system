"""Step 2: train, evaluate and save the model (Python 3.14 compatible, no TensorFlow)."""
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix,ConfusionMatrixDisplay, roc_auc_score)

SEED = 42
rng = np.random.default_rng(SEED)

df = pd.read_csv("features.csv")
feature_cols = [c for c in df.columns if c.startswith("f")]

# ---- split BY SOURCE FILE so augmented copies never leak into the test set ----
file_labels = df.groupby("group")["label"].first()
train_files, test_files = train_test_split(file_labels.index, test_size=0.2, random_state=SEED, stratify=file_labels.values)

train_df = df[df.group.isin(train_files)]
test_df = df[df.group.isin(test_files) & (df.augmented == 0)]   # real clips only

X_train, y_train = train_df[feature_cols].values, train_df["label"].values
X_test, y_test = test_df[feature_cols].values, test_df["label"].values

# ---- balance classes by oversampling the smaller class (training set only) ----
counts = np.bincount(y_train)
print("Train class counts before balancing:", dict(enumerate(counts)))
minority = int(np.argmin(counts))
extra = rng.choice(np.where(y_train == minority)[0], size=counts.max() - counts.min(), replace=True)
X_train = np.vstack([X_train, X_train[extra]])
y_train = np.concatenate([y_train, y_train[extra]])
print(f"Train: {len(X_train)}  Test: {len(X_test)}")

# ---- scaling (fit on train only) ----
scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
joblib.dump(scaler, "scaler.joblib")

# ---- model: multi-layer perceptron (neural network) ----
model = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                      solver="adam", alpha=1e-3,          # L2 regularisation
                      batch_size=32, learning_rate_init=1e-3,
                      max_iter=300, early_stopping=True,
                      validation_fraction=0.15, n_iter_no_change=15,
                      random_state=SEED, verbose=False)
model.fit(X_train, y_train)
print(f"Stopped after {model.n_iter_} epochs")
joblib.dump(model, "scream_detector_model.joblib")

# ---- evaluation on the untouched test set ----
probs = model.predict_proba(X_test)[:, 1]
preds = (probs > 0.5).astype(int)
report = classification_report(y_test, preds, target_names=["NotScreaming", "Screaming"])
cm = confusion_matrix(y_test, preds)
auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else float("nan")
print("\n=== TEST RESULTS ===")
print(report)
print("Confusion matrix:\n", cm)
print(f"ROC-AUC: {auc:.3f}")

with open("evaluation_report.txt", "w") as f:
    f.write(f"Train/Test samples: {len(X_train)}/{len(X_test)}\n\n")
    f.write(report + f"\nConfusion matrix:\n{cm}\nROC-AUC: {auc:.3f}\n")

ConfusionMatrixDisplay(cm, display_labels=["NotScreaming", "Screaming"]).plot(cmap="Blues")
plt.title("Confusion matrix (test set)")
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(model.loss_curve_); ax[0].set_title("Training loss"); ax[0].set_xlabel("epoch")
ax[1].plot(model.validation_scores_); ax[1].set_title("Validation accuracy"); ax[1].set_xlabel("epoch")
plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved: scream_detector_model.joblib, scaler.joblib, evaluation_report.txt, "
      "confusion_matrix.png, training_curves.png")