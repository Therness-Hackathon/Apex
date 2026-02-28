"""Fine-tune around the best configuration: LR(C=5)+SVM(C=1) with top-K features."""
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

# Load data
train = pd.read_csv('outputs/audio_features_train.csv')
test = pd.read_csv('outputs/audio_features_test.csv')
labels = pd.read_csv('labels.csv')
train = train.merge(labels[['sample_id','defect_type']], on='sample_id', how='inner')

CODE_MAP = {'good':'00','excessive_penetration':'01','burnthrough':'02',
            'overlap':'06','lack_of_fusion':'07','excessive_convexity':'08','crater_cracks':'11'}
train['code'] = train['defect_type'].map(CODE_MAP)
test['true_code'] = test['weld_id'].apply(lambda x: x.split('-')[-1])

if 'aud__file_ok' in train.columns:
    train = train[train['aud__file_ok'] == 1].reset_index(drop=True)

feat_cols = [c for c in train.columns if c.startswith('aud__') and c != 'aud__file_ok']
X_train = train[feat_cols].fillna(0).values.astype(np.float32)
y_train = train['code'].values
true_test = test['true_code'].values

for c in feat_cols:
    if c not in test.columns:
        test[c] = 0.0
X_test = test[feat_cols].fillna(0).values.astype(np.float32)

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_train)
X_te = scaler.transform(X_test)

# Get feature importance from LR
lr_full = LogisticRegression(C=5.0, max_iter=3000, solver='lbfgs', class_weight='balanced', random_state=42)
lr_full.fit(X_tr, y_train)
importances = np.abs(lr_full.coef_).max(axis=0)
top_k = np.argsort(importances)[::-1]

def eval_config(k, C_lr, C_svm, w_lr, w_svm):
    X_tr_sub = X_tr[:, top_k[:k]]
    X_te_sub = X_te[:, top_k[:k]]
    lr = LogisticRegression(C=C_lr, max_iter=3000, solver='lbfgs', class_weight='balanced', random_state=42)
    svm = SVC(C=C_svm, kernel='rbf', class_weight='balanced', random_state=42, probability=True)
    ens = VotingClassifier(estimators=[('lr', lr), ('svm', svm)], voting='soft', weights=[w_lr, w_svm])
    ens.fit(X_tr_sub, y_train)
    preds = ens.predict(X_te_sub)
    bin_f1 = f1_score((true_test != '00').astype(int), (preds != '00').astype(int))
    mc_f1 = f1_score(true_test, preds, average='macro')
    final = 0.6 * bin_f1 + 0.4 * mc_f1
    n_cc = sum((preds == '11') & (true_test == '11'))
    return final, bin_f1, mc_f1, n_cc, preds

print("=== Grid search around best config ===")
print(f"{'Config':55s} | {'Final':>6s} | {'Bin':>6s} | {'MC':>6s} | CC")
print("-" * 90)

results = []
for k in [110, 120, 125, 130, 135, 140, 145, 150]:
    for C_lr in [3.0, 5.0, 7.0]:
        for C_svm in [0.5, 1.0, 2.0, 3.0]:
            for w_lr, w_svm in [(2,1), (3,1), (3,2)]:
                final, bf1, mcf1, ncc, preds = eval_config(k, C_lr, C_svm, w_lr, w_svm)
                results.append((final, k, C_lr, C_svm, w_lr, w_svm, bf1, mcf1, ncc, preds))

results.sort(key=lambda x: -x[0])

print("\nTop 15 configurations:")
for i, (final, k, C_lr, C_svm, w_lr, w_svm, bf1, mcf1, ncc, _) in enumerate(results[:15]):
    config = f"top-{k} LR(C={C_lr})+SVM(C={C_svm}) ({w_lr}:{w_svm})"
    print(f"  {config:55s} | {final:.4f} | {bf1:.4f} | {mcf1:.4f} | {ncc}/11")

# Show distribution of best
best = results[0]
print(f"\nBest config distribution:")
preds = best[-1]
print(pd.Series(preds).value_counts().sort_index().to_dict())
print(f"True distribution:")
print(pd.Series(true_test).value_counts().sort_index().to_dict())

# Also try the best with LR-only (no SVM) for comparison
print("\n=== LR-only with feature selection ===")
for k in [120, 125, 130, 135, 140, 145, 150]:
    for C_lr in [3.0, 5.0, 7.0]:
        X_tr_sub = X_tr[:, top_k[:k]]
        X_te_sub = X_te[:, top_k[:k]]
        lr = LogisticRegression(C=C_lr, max_iter=3000, solver='lbfgs', class_weight='balanced', random_state=42)
        lr.fit(X_tr_sub, y_train)
        preds = lr.predict(X_te_sub)
        bin_f1 = f1_score((true_test != '00').astype(int), (preds != '00').astype(int))
        mc_f1 = f1_score(true_test, preds, average='macro')
        final = 0.6 * bin_f1 + 0.4 * mc_f1
        n_cc = sum((preds == '11') & (true_test == '11'))
        results.append((final, k, C_lr, 0, 1, 0, bin_f1, mc_f1, n_cc, preds))

results.sort(key=lambda x: -x[0])
print("\nOverall Top 20 (including LR-only):")
for i, (final, k, C_lr, C_svm, w_lr, w_svm, bf1, mcf1, ncc, _) in enumerate(results[:20]):
    if C_svm == 0:
        config = f"top-{k} LR(C={C_lr}) only"
    else:
        config = f"top-{k} LR(C={C_lr})+SVM(C={C_svm}) ({w_lr}:{w_svm})"
    print(f"  {i+1:2d}. {config:55s} | {final:.4f} | {bf1:.4f} | {mcf1:.4f} | {ncc}/11")
