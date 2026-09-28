import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from data_loader import load_sam40_dataset
from features import extract_band_powers

def evaluate_binary_models():
    print("1. Loading raw SAM-40 dataset...")
    X_raw, y_multi, meta, channels = load_sam40_dataset()
    
    # Map multi-class (0: Relax, 1: Arithmetic, 2: Mirror, 3: Stroop)
    # to Binary: 0 = Relax (Low Stress), 1 = Stress (Active Task)
    y_binary = np.where(y_multi == 0, 0, 1)
    
    print("2. Extracting subject-wise normalized features...")
    X, _ = extract_band_powers(X_raw, meta, fs=128.0)
    
    subjects = np.array([m['subject_id'] for m in meta])
    
    models = {
        "SVM (RBF)": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True))
        ]),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_depth=4, random_state=42)
    }
    
    logo = LeaveOneGroupOut()
    
    print("\n3. Starting Binary LOSO-CV (Relax vs. Stress)...")
    print(f"Total Subjects: {len(np.unique(subjects))}")
    print(f"Class Distribution -> Relax (0): {np.sum(y_binary == 0)}, Stress (1): {np.sum(y_binary == 1)}\n")
    
    for name, model in models.items():
        y_true_all = []
        y_pred_all = []
        
        for train_idx, test_idx in logo.split(X, y_binary, groups=subjects):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_binary[train_idx], y_binary[test_idx]
            
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            y_true_all.extend(y_test)
            y_pred_all.extend(preds)
            
        acc = accuracy_score(y_true_all, y_pred_all)
        f1 = f1_score(y_true_all, y_pred_all)
        
        print(f"=== {name} ===")
        print(f"Binary LOSO Accuracy: {acc * 100:.2f}%")
        print(f"Binary F1-Score     : {f1:.4f}\n")

if __name__ == '__main__':
    evaluate_binary_models()
