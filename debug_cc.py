"""Debug crater_cracks distribution shift."""
import pandas as pd, numpy as np
from scipy.spatial.distance import cdist

_train_raw = pd.read_csv('outputs/audio_features_train.csv')
_labels = pd.read_csv('labels.csv')
train = _train_raw.merge(_labels[['sample_id','defect_type']], on='sample_id', how='inner').copy()
_test_raw = pd.read_csv('outputs/audio_features_test.csv')
test = _test_raw.copy()

CODE_MAP = {'good':'00','excessive_penetration':'01','burnthrough':'02',
            'overlap':'06','lack_of_fusion':'07','excessive_convexity':'08','crater_cracks':'11'}
train['code'] = train['defect_type'].map(CODE_MAP)
test['true_code'] = test['weld_id'].apply(lambda x: x.split('-')[-1])

feat_cols = [c for c in train.columns if c.startswith('aud__') and c != 'aud__file_ok']

train_cc = train[train['code'] == '11']
train_good = train[train['code'] == '00']
test_cc = test[test['true_code'] == '11']
test_good = test[test['true_code'] == '00']

print(f"Train CC: {len(train_cc)}, Test CC: {len(test_cc)}")
print(f"Train Good: {len(train_good)}, Test Good: {len(test_good)}")

# Key features
key_feats = ['aud__duration_s', 'aud__mfcc00_mean', 'aud__mfcc01_mean', 
             'aud__spectral_centroid_mean', 'aud__zcr_mean', 'aud__rms_mean']

print("\nFeature comparison (mean values):")
header = f"{'Feature':>30s} | {'Train_CC':>10s} | {'Train_Good':>10s} | {'Test_CC':>10s} | {'Test_Good':>10s}"
print(header)
print("-" * len(header))
for f in key_feats:
    if f in feat_cols:
        tc = train_cc[f].mean()
        tg = train_good[f].mean()
        xc = test_cc[f].mean()
        xg = test_good[f].mean()
        print(f"{f:>30s} | {tc:10.3f} | {tg:10.3f} | {xc:10.3f} | {xg:10.3f}")

# Centroid distances
cc_centroid_train = np.array(train_cc[feat_cols].fillna(0).mean().values).reshape(1,-1)
good_centroid_train = np.array(train_good[feat_cols].fillna(0).mean().values).reshape(1,-1)
cc_centroid_test = np.array(test_cc[feat_cols].fillna(0).mean().values).reshape(1,-1)

d_cc_to_train_cc = cdist(cc_centroid_test, cc_centroid_train, metric='cosine')[0,0]
d_cc_to_train_good = cdist(cc_centroid_test, good_centroid_train, metric='cosine')[0,0]
print(f"\nTest CC centroid -> Train CC centroid (cosine): {d_cc_to_train_cc:.6f}")
print(f"Test CC centroid -> Train Good centroid (cosine): {d_cc_to_train_good:.6f}")
print(f"Test CC is {'closer to Train CC' if d_cc_to_train_cc < d_cc_to_train_good else 'closer to Train Good'}")

# Also check all classes
print("\nCentroid distances from test CC to each training class centroid:")
for code in sorted(train['code'].unique()):
    centroid = np.array(train[train['code'] == code][feat_cols].fillna(0).mean().values).reshape(1,-1)
    dist = cdist(cc_centroid_test, centroid, metric='cosine')[0,0]
    n = len(train[train['code'] == code])
    print(f"  {code} (n={n:4d}): cosine dist = {dist:.6f}")

# Check per-sample: for each test CC sample, what's its nearest training class?
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_all = scaler.fit_transform(train[feat_cols].fillna(0).values)
X_test_cc = scaler.transform(test_cc[feat_cols].fillna(0).values)

print("\nPer test CC sample - nearest training sample's class:")
for i, (_, row) in enumerate(test_cc.iterrows()):
    x = X_test_cc[i:i+1]
    dists = cdist(x, X_train_all, metric='euclidean')[0]
    nearest_5 = np.argsort(dists)[:5]
    nearest_codes = train.iloc[nearest_5]['code'].values
    print(f"  Test CC sample {row['sample_id']}: nearest 5 = {list(nearest_codes)}, dist={dists[nearest_5[0]]:.2f}")
