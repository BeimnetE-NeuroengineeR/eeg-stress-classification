import os
import glob
import re
import numpy as np
from scipy.io import loadmat

# Mapping task prefix names to numeric targets
TASK_MAPPING = {
    'Relax': 0,        # Baseline / Low Stress
    'Arithmetic': 1,   # Cognitive Stress
    'Mirror': 2,       # Visuomotor Stress
    'Stroop': 3        # Cognitive Interference Stress
}

CHANNEL_NAMES = [
    'Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 
    'CP5', 'CP1', 'P3', 'P7', 'PO9', 'O1', 'Pz', 'Oz', 'O2', 'PO10', 
    'P8', 'P4', 'CP2', 'CP6', 'T8', 'FT10', 'FC6', 'C4', 'FC2', 'F4', 'F8', 'Fp2'
]

def load_sam40_dataset(data_dir='data/raw/sam40/EEG_SAM40_Project/Data/filtered_data'):
    """
    Loads all SAM-40 .mat files into structured arrays.
    Returns:
        X: numpy array of shape (480, 32, 3200) -> (trials, channels, timepoints)
        y: numpy array of shape (480,) -> task labels (0-3)
        metadata: list of dicts containing subject_id, task, trial_num
    """
    filepaths = sorted(glob.glob(os.path.join(data_dir, '*.mat')))
    
    X_list = []
    y_list = []
    metadata = []

    # Regex matches: Arithmetic, Relax, Stroop, OR Mirror_image
    pattern = re.compile(r'([A-Za-z]+)(?:_image)?_sub_(\d+)_trial(\d+)\.mat')

    for filepath in filepaths:
        filename = os.path.basename(filepath)
        match = pattern.match(filename)
        
        if not match:
            continue
            
        task, sub_id, trial_num = match.groups()
        
        # Ensure task matches canonical name
        if task not in TASK_MAPPING:
            continue

        # Load .mat file
        mat_data = loadmat(filepath)
        clean_data = mat_data['Clean_data']  # Shape: (32, 3200)
        
        X_list.append(clean_data)
        y_list.append(TASK_MAPPING[task])
        metadata.append({
            'subject_id': int(sub_id),
            'task': task,
            'trial': int(trial_num)
        })

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int64)

    return X, y, metadata, CHANNEL_NAMES

if __name__ == '__main__':
    X, y, meta, channels = load_sam40_dataset()
    print("--- SAM-40 Dataset Loaded Successfully ---")
    print(f"X shape (trials, channels, samples): {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Total subjects: {len(set(m['subject_id'] for m in meta))}")
    
    # Verify exact counts per class
    unique_labels, counts = np.unique(y, return_counts=True)
    label_names = {v: k for k, v in TASK_MAPPING.items()}
    for label, count in zip(unique_labels, counts):
        print(f"  Class {label} ({label_names[label]}): {count} trials")
