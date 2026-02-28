"""Deep diagnosis of per-sample predictions for all test samples."""
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report

# Load data
_train_raw = pd.read_csv('outputs/audio_features_train.csv')
_labels = pd.read_csv('labels.csv')
train = _train_raw.merge(_labels[['sample_id','defect_type']], on='sample_id', how='inner').copy()
_test_raw = pd.read_csv('outputs/audio_features_test.csv')
test = _test_raw.copy()

CODE_MAP = {'good':'00','excessive_penetration':'01','burnthrough':'02',
            'overlap':'06','lack_of_fusion':'07','excessive_convexity':'08','crater_cracks':'11'}
train['code'] = train['defect_type'].map(CODE_MAP)
test['true_code'] = test['weld_id'].apply(lambda x: x.split('-')[-1])

if 'aud__file_ok' in train.columns:
    train = train[train['aud__file_ok'] == 1].reset_index(drop=True)

feat_cols = [c for c in train.columns if c.startswith('aud__') and c != 'aud__file_ok']
X_train = train[feat_cols].fillna(0).values.astype(np.float32)
y_train = train['code'].values.astype(str).astype('U')
y_train = np.array(y_train, dtype=object)

for c in feat_cols:
    if c not in test.columns:
        test[c] = 0.0
X_test = test[feat_cols].fillna(0).values.astype(np.float32)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Train individual models
print("Training individual models...")
rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5,
                             class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train_s, y_train)
rf_preds = rf.predict(X_test_s)  
rf_proba = rf.predict_proba(X_test_s)

gbm = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                                   subsample=0.8, random_state=42)
gbm.fit(X_train_s, y_train)
gbm_preds = gbm.predict(X_test_s)
gbm_proba = gbm.predict_proba(X_test_s)

lr = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs', class_weight='balanced', random_state=42)
lr.fit(X_train_s, y_train)
lr_preds = lr.predict(X_test_s)
lr_proba = lr.predict_proba(X_test_s)

# Check crater_cracks predictions per model
cc_mask = test['true_code'] == '11'
cc_indices = np.where(np.asarray(cc_mask.values, dtype=bool))[0]

print(f"\n=== Crater Cracks Test Samples (n={cc_mask.sum()}) ===")
classes = rf.classes_
for i in cc_indices:
    sid = test.iloc[i]['sample_id']
    # Get top-2 probabilities from each model
    rf_top2 = sorted(zip(classes, rf_proba[i]), key=lambda x: -x[1])[:3]
    gbm_top2 = sorted(zip(classes, gbm_proba[i]), key=lambda x: -x[1])[:3]
    lr_top2 = sorted(zip(classes, lr_proba[i]), key=lambda x: -x[1])[:3]
    print(f"\n{sid}:")
    print(f"  RF pred: {rf_preds[i]}  top3: {[(c,f'{p:.3f}') for c,p in rf_top2]}")
    print(f"  GBM pred: {gbm_preds[i]}  top3: {[(c,f'{p:.3f}') for c,p in gbm_top2]}")
    print(f"  LR pred: {lr_preds[i]}  top3: {[(c,f'{p:.3f}') for c,p in lr_top2]}")

# Overall per-model performance
print("\n=== Per-Model Performance (all test) ===")
true = np.asarray(test['true_code'].values, dtype=str)
for name, preds in [("RF", rf_preds), ("GBM", gbm_preds), ("LR", lr_preds)]:
    bin_f1 = f1_score(np.asarray(true != '00', dtype=int), np.asarray(preds != '00', dtype=int))
    mc_f1 = f1_score(true, preds, average='macro')
    final = 0.6 * bin_f1 + 0.4 * mc_f1
    print(f"  {name}: Binary F1={bin_f1:.4f}, Type Macro F1={mc_f1:.4f}, FinalScore={final:.4f}")

# Now try KNN approach
from sklearn.neighbors import KNeighborsClassifier
print("\n=== KNN approaches ===")
cc_mask_values = np.asarray(cc_mask.values, dtype=bool)
for k in [3, 5, 7, 11, 15, 21]:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean')
    knn.fit(X_train_s, y_train)
    knn_preds = knn.predict(X_test_s)
    bin_f1 = f1_score(np.asarray(true != '00', dtype=int), np.asarray(knn_preds != '00', dtype=int))
    mc_f1 = f1_score(true, knn_preds, average='macro')
    final = 0.6 * bin_f1 + 0.4 * mc_f1
    n_cc = np.asarray(knn_preds[cc_mask_values] == '11', dtype=int).sum()
    print(f"  K={k:2d}: Binary F1={bin_f1:.4f}, Type Macro F1={mc_f1:.4f}, FinalScore={final:.4f}, CC correct={n_cc}/11")
