import numpy as np
from scipy.signal import welch
from scipy.integrate import trapezoid

EEG_BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta': (13.0, 30.0),
    'gamma': (30.0, 45.0)
}

FRONTAL_PAIRS = [(2, 31), (3, 30), (4, 29), (5, 28)]

def extract_band_powers(data, metadata, fs=128.0):
    """
    Computes absolute band powers, relative ratios, and frontal asymmetry,
    followed by subject-wise normalization to eliminate inter-subject amplitude variance.
    """
    n_trials, n_channels, n_samples = data.shape
    band_names = list(EEG_BANDS.keys())
    
    features_list = []
    
    for i in range(n_trials):
        trial_feats = []
        band_power_dict = {b: np.zeros(n_channels) for b in band_names}
        
        for ch in range(n_channels):
            signal = data[i, ch, :]
            freqs, psd = welch(signal, fs=fs, nperseg=min(256, n_samples))
            
            total_power = trapezoid(psd, freqs) + 1e-8
            for band_name, (fmin, fmax) in EEG_BANDS.items():
                idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
                bp = trapezoid(psd[idx_band], freqs[idx_band])
                
                # Relative band power (band power / total power)
                rel_bp = bp / total_power
                band_power_dict[band_name][ch] = rel_bp
                trial_feats.append(rel_bp)
                
        # Ratios
        theta = band_power_dict['theta']
        alpha = band_power_dict['alpha']
        beta = band_power_dict['beta'] + 1e-8
        
        tbr = theta / beta
        abr = alpha / beta
        trial_feats.extend(tbr)
        trial_feats.extend(abr)
        
        # Frontal Alpha Asymmetry
        for left_ch, right_ch in FRONTAL_PAIRS:
            l_alpha = np.log(band_power_dict['alpha'][left_ch] + 1e-8)
            r_alpha = np.log(band_power_dict['alpha'][right_ch] + 1e-8)
            faa = r_alpha - l_alpha
            trial_feats.append(faa)
            
        features_list.append(trial_feats)

    features = np.array(features_list, dtype=np.float64)
    
    # Perform Subject-Wise Z-score Normalization
    subjects = np.array([m['subject_id'] for m in metadata])
    for sub in np.unique(subjects):
        mask = (subjects == sub)
        sub_mean = np.mean(features[mask], axis=0)
        sub_std = np.std(features[mask], axis=0) + 1e-8
        features[mask] = (features[mask] - sub_mean) / sub_std

    return features, None

if __name__ == '__main__':
    from data_loader import load_sam40_dataset
    print("Extracting subject-normalized relative features...")
    X_raw, y, meta, channels = load_sam40_dataset()
    X_feats, _ = extract_band_powers(X_raw, meta, fs=128.0)
    print(f"Normalized Feature Matrix Shape: {X_feats.shape}")
