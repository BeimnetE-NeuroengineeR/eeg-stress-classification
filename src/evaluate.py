import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import load_sam40_dataset
from features import extract_band_powers

def run_full_evaluation():
    os.makedirs('reports', exist_ok=True)
    
    print("1. Loading SAM-40 dataset & extracting features...")
    X_raw, y_multi, meta, channels = load_sam40_dataset()
    X, _ = extract_band_powers(X_raw, meta, fs=128.0)
    
    subjects = np.array([m['subject_id'] for m in meta])
    logo = LeaveOneGroupOut()
    
    # -------------------------------------------------------------
    # Multi-Class Evaluation (4-Class)
    # -------------------------------------------------------------
    print("\n2. Evaluating Multi-Class (4-Class) Model [HistGradientBoosting]...")
    clf_multi = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_depth=4, random_state=42)
    
    y_true_multi, y_pred_multi = [], []
    for train_idx, test_idx in logo.split(X, y_multi, groups=subjects):
        clf_multi.fit(X[train_idx], y_multi[train_idx])
        y_true_multi.extend(y_multi[test_idx])
        y_pred_multi.extend(clf_multi.predict(X[test_idx]))
        
    class_names_4 = ['Relax', 'Arithmetic', 'Mirror', 'Stroop']
    print("\n--- Multi-Class Classification Report ---")
    print(classification_report(y_true_multi, y_pred_multi, target_names=class_names_4))
    
    # Save Multi-Class Confusion Matrix
    cm_multi = confusion_matrix(y_true_multi, y_pred_multi)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names_4, yticklabels=class_names_4)
    plt.title('Multi-Class LOSO Confusion Matrix (HistGradientBoosting)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('reports/confusion_matrix_multiclass.png', dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # Binary Evaluation (Relax vs. Stress)
    # -------------------------------------------------------------
    print("\n3. Evaluating Binary (Relax vs. Stress) Model [SVM RBF]...")
    y_binary = np.where(y_multi == 0, 0, 1)
    
    clf_binary = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', C=1.0, gamma='scale'))
    ])
    
    y_true_bin, y_pred_bin = [], []
    for train_idx, test_idx in logo.split(X, y_binary, groups=subjects):
        clf_binary.fit(X[train_idx], y_binary[train_idx])
        y_true_bin.extend(y_binary[test_idx])
        y_pred_bin.extend(clf_binary.predict(X[test_idx]))
        
    class_names_2 = ['Relax (0)', 'Stress (1)']
    print("\n--- Binary Classification Report ---")
    print(classification_report(y_true_bin, y_pred_bin, target_names=class_names_2))
    
    # Save Binary Confusion Matrix
    cm_bin = confusion_matrix(y_true_bin, y_pred_bin)
    plt.figure(figsize=(6, 4.5))
    sns.heatmap(cm_bin, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names_2, yticklabels=class_names_2)
    plt.title('Binary LOSO Confusion Matrix (SVM RBF)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('reports/confusion_matrix_binary.png', dpi=300)
    plt.close()
    
    print("\nEvaluation complete! Artifacts saved to 'reports/' directory.")

if __name__ == '__main__':
    run_full_evaluation()
