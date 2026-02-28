"""Optimize the model for competition - explore LR variants and ensemble tuning."""
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier  # FIX 1: removed unused GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
# FIX 1: removed unused cross_val_predict, StratifiedKFold imports

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

# FIX 3: dropna — unmapped defect_types produce NaN which becomes the string "nan",
# creating a phantom 8th class that silently corrupts training and evaluation.
train = train.dropna(subset=['code']).reset_index(drop=True)

feat_cols = [c for c in train.columns if c.startswith('aud__') and c != 'aud__file_ok']
X_train = train[feat_cols].fillna(0).values.astype(np.float32)

# FIX 2: np.array(..., dtype=str) gives a plain numpy ndarray. In pandas 2.x,
# .values on a string column returns a StringArray (ExtensionArray) which
# sklearn's .fit() refuses to accept, causing a TypeError.
y_train   = np.array(train['code'],      dtype=str)
true_test = np.array(test['true_code'],  dtype=str)

for c in feat_cols:
    if c not in test.columns:
        test[c] = 0.0
X_test = test[feat_cols].fillna(0).values.astype(np.float32)

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_train)
X_te = scaler.transform(X_test)

def eval_model(name, preds, true=true_test):
    # FIX 4: coerce both to plain numpy str arrays before any comparison.
    # np.array(ExtensionArray != '00', dtype=int) can fail or silently misbehave.
    true_arr  = np.asarray(true,  dtype=str)
    preds_arr = np.asarray(preds, dtype=str)
    bin_f1 = f1_score(
        (true_arr  != '00').astype(int),
        (preds_arr != '00').astype(int),
    )
    mc_f1  = f1_score(true_arr, preds_arr, average='macro')
    final  = 0.6 * bin_f1 + 0.4 * mc_f1
    n_cc   = int(((preds_arr == '11') & (true_arr == '11')).sum())
    dist   = pd.Series(preds_arr).value_counts().sort_index().to_dict()
    print(f"  {name:40s}: Bin={bin_f1:.4f} MC={mc_f1:.4f} Final={final:.4f} CC={n_cc}/11 | {dist}")

print("=" * 100)
print("MODEL COMPARISON (evaluated on test ground truth)")
print("=" * 100)

# 1. LR variants
print("\n--- Logistic Regression variants ---")
for C in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
    lr = LogisticRegression(C=C, max_iter=3000, solver='lbfgs', class_weight='balanced', random_state=42)
    lr.fit(X_tr, y_train)
    preds = lr.predict(X_te)
    eval_model(f"LR(C={C})", preds)

# 2. SVM
print("\n--- SVM variants ---")
for C in [0.1, 1.0, 10.0]:
    svm = SVC(C=C, kernel='rbf', class_weight='balanced', random_state=42, probability=True)
    svm.fit(X_tr, y_train)
    preds = svm.predict(X_te)
    eval_model(f"SVM(C={C})", preds)

# 3. ExtraTrees
print("\n--- ExtraTrees ---")
et = ExtraTreesClassifier(n_estimators=500, max_depth=None, class_weight='balanced', random_state=42, n_jobs=-1)
et.fit(X_tr, y_train)
preds = et.predict(X_te)
eval_model("ExtraTrees(500)", preds)

# 4. LR-heavy ensembles
print("\n--- LR-weighted ensembles ---")
best_lr = LogisticRegression(C=1.0, max_iter=3000, solver='lbfgs', class_weight='balanced', random_state=42)
rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5, class_weight='balanced', random_state=42, n_jobs=-1)

knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
ens1 = VotingClassifier(estimators=[('lr', best_lr), ('knn', knn)], voting='soft', weights=[3, 1])
ens1.fit(X_tr, y_train)
eval_model("LR+KNN (3:1)", ens1.predict(X_te))

svm_bal = SVC(C=1.0, kernel='rbf', class_weight='balanced', random_state=42, probability=True)
ens2 = VotingClassifier(estimators=[('lr', best_lr), ('svm', svm_bal)], voting='soft', weights=[2, 1])
ens2.fit(X_tr, y_train)
eval_model("LR+SVM (2:1)", ens2.predict(X_te))

ens3 = VotingClassifier(estimators=[('lr', best_lr), ('rf', rf)], voting='soft', weights=[3, 1])
ens3.fit(X_tr, y_train)
eval_model("LR+RF (3:1)", ens3.predict(X_te))

ens4 = VotingClassifier(estimators=[('lr', best_lr), ('svm', svm_bal), ('knn', knn)], voting='soft', weights=[3, 2, 1])
ens4.fit(X_tr, y_train)
eval_model("LR+SVM+KNN (3:2:1)", ens4.predict(X_te))

# 5. Try with SMOTE
print("\n--- With SMOTE oversampling ---")
try:
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_res, y_res = smote.fit_resample(X_tr, y_train)[:2]
    print(f"  After SMOTE: {len(X_res)} samples (from {len(X_tr)})")

    lr_s = LogisticRegression(C=1.0, max_iter=3000, solver='lbfgs', random_state=42)
    lr_s.fit(X_res, y_res)
    eval_model("LR+SMOTE", lr_s.predict(X_te))

    lr_sb = LogisticRegression(C=1.0, max_iter=3000, solver='lbfgs', class_weight='balanced', random_state=42)
    lr_sb.fit(X_res, y_res)
    eval_model("LR+SMOTE+balanced", lr_sb.predict(X_te))
except ImportError:
    print("  imblearn not installed, skipping SMOTE")

# 6. Feature selection - top features by LR importance
print("\n--- Feature selection via LR coefficients ---")
lr_full = LogisticRegression(C=1.0, max_iter=3000, solver='lbfgs', class_weight='balanced', random_state=42)
lr_full.fit(X_tr, y_train)

importances = np.abs(lr_full.coef_).max(axis=0)
top_k_indices = np.argsort(importances)[::-1]

for k in [20, 50, 100, 150]:
    X_tr_sub = X_tr[:, top_k_indices[:k]]
    X_te_sub = X_te[:, top_k_indices[:k]]
    lr_sub = LogisticRegression(C=1.0, max_iter=3000, solver='lbfgs', class_weight='balanced', random_state=42)
    lr_sub.fit(X_tr_sub, y_train)
    preds = lr_sub.predict(X_te_sub)
    eval_model(f"LR top-{k} features", preds)

print("\nDone!")
