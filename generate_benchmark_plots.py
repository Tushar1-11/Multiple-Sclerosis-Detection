"""
Generate High-Resolution Benchmark & Performance Charts for MSNet using Matplotlib
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup style
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 17
})

output_dir = Path("outputs/charts")
output_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# Chart 1: Classification Multi-Algorithm Comparison
# -------------------------------------------------------------
algorithms = ["Logistic Reg", "SVM (RBF)", "Random Forest", "LightGBM/XGB", "Neural Net (MLP)", "CNNAttention (Ours)"]
accuracy    = [94.94, 94.76, 93.19, 94.85, 94.15, 95.66]
macro_f1    = [0.9427, 0.9407, 0.9209, 0.9412, 0.9338, 0.9495]
roc_auc     = [0.9888, 0.9862, 0.9871, 0.9875, 0.9863, 0.9885]

x = np.arange(len(algorithms))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width, accuracy, width, label='Accuracy (%)', color='#1f77b4')
rects2 = ax.bar(x, [f * 100 for f in macro_f1], width, label='Macro F1 (%)', color='#ff7f0e')
rects3 = ax.bar(x + width, [a * 100 for a in roc_auc], width, label='ROC-AUC (%)', color='#2ca02c')

ax.set_ylabel('Performance (%)')
ax.set_title('Phenotype Subtyping Performance Across Algorithms (Held-Out Test Set)')
ax.set_xticks(x)
ax.set_xticklabels(algorithms, rotation=15)
ax.set_ylim(85, 100)
ax.legend(loc='lower right')
ax.grid(True, linestyle='--', alpha=0.5)

for rect in rects1 + rects2 + rects3:
    height = rect.get_height()
    ax.annotate(f'{height:.1f}%',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
chart1_path = output_dir / "classification_benchmark.png"
plt.savefig(chart1_path, dpi=300)
plt.close()

# -------------------------------------------------------------
# Chart 2: EDSS Regression Error Comparison (MAE & RMSE)
# -------------------------------------------------------------
reg_algos = ["Ridge Regression", "SVR (RBF)", "Random Forest", "LightGBM", "MLP Regressor", "CNNAttention (Ours)"]
mae  = [0.1533, 0.3095, 0.0801, 0.0754, 0.1189, 0.0636]
rmse = [0.2391, 0.4919, 0.1772, 0.1695, 0.1701, 0.1626]

fig, ax = plt.subplots(figsize=(11, 5.5))
x = np.arange(len(reg_algos))
width = 0.35

rects1 = ax.bar(x - width/2, mae, width, label='MAE (Lower is Better)', color='#d95f02')
rects2 = ax.bar(x + width/2, rmse, width, label='RMSE (Lower is Better)', color='#7570b3')

ax.set_ylabel('EDSS Score Error (Points)')
ax.set_title('EDSS Disability Score Regression Error Comparison')
ax.set_xticks(x)
ax.set_xticklabels(reg_algos, rotation=15)
ax.set_ylim(0, 0.55)
ax.legend(loc='upper right')
ax.grid(True, linestyle='--', alpha=0.5)

for rect in rects1 + rects2:
    height = rect.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
chart2_path = output_dir / "edss_regression_benchmark.png"
plt.savefig(chart2_path, dpi=300)
plt.close()

# -------------------------------------------------------------
# Chart 3: Confusion Matrix Heatmap
# -------------------------------------------------------------
cm = np.array([
    [104, 0, 0],
    [0, 222, 11],
    [0, 7, 71]
])
class_names = ["Healthy Control", "RRMS", "Progressive MS"]

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues")

ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))
ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)

for i in range(len(class_names)):
    for j in range(len(class_names)):
        color = "white" if cm[i, j] > 100 else "black"
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=14, weight="bold")

ax.set_title("CNNAttention Hybrid -- Confusion Matrix (N = 415 Test Cases)")
ax.set_xlabel("Predicted Phenotype Label")
ax.set_ylabel("True Ground-Truth Label")
fig.colorbar(im)

plt.tight_layout()
chart3_path = output_dir / "confusion_matrix.png"
plt.savefig(chart3_path, dpi=300)
plt.close()

# -------------------------------------------------------------
# Chart 4: Learning Curves & Overfitting Gap
# -------------------------------------------------------------
sizes = [132, 431, 729, 1028, 1327]
train_f1 = [96.95, 96.38, 96.52, 96.24, 96.10]
val_f1   = [85.82, 93.81, 93.18, 93.86, 95.25]

fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(sizes, train_f1, 'o-', color='#e41a1c', linewidth=2.5, label='Training Macro F1 (%)')
ax.plot(sizes, val_f1, 's-', color='#377eb8', linewidth=2.5, label='Validation Macro F1 (%)')
ax.fill_between(sizes, train_f1, val_f1, color='gray', alpha=0.15, label='Overfitting Gap (Narrows to 0.85%)')

ax.set_xlabel('Training Sample Size')
ax.set_ylabel('Macro F1 Score (%)')
ax.set_title('Learning Curve: Model Generalization vs Sample Size')
ax.set_ylim(80, 100)
ax.legend(loc='lower right')
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
chart4_path = output_dir / "learning_curves.png"
plt.savefig(chart4_path, dpi=300)
plt.close()

print(f"Generated 4 benchmark charts in {output_dir}")
