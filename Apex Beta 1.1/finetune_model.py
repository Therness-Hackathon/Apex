"""Fine-tune the best approaches."""
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
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
y_train = np.array(train['code'], dtype=str)
true_test = np.array(test['true_code'], dtype=str)

for c in feat_cols:
    if c not in test.columns:
        test[c] = 0.0
X_test = test[feat_cols].fillna(0).values.astype(np.float32)

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_train)
X_te = scaler.transform(X_test)

def eval_model(name, preds, proba=None, true=true_test):
    true_arr = np.asarray(true, dtype=str)
    preds_arr = np.asarray(preds, dtype=str)
    bin_f1 = f1_score((true_arr != '00').astype(int), (preds_arr != '00').astype(int))
    mc_f1 = f1_score(true_arr, preds_arr, average='macro')
    final = 0.6 * bin_f1 + 0.4 * mc_f1
    n_cc = int(np.sum((preds_arr == '11') & (true_arr == '11')))
    dist = pd.Series(preds_arr).value_counts().sort_index().to_dict()
    print(f"  {name:45s}: Bin={bin_f1:.4f} MC={mc_f1:.4f} Final={final:.4f} CC={n_cc}/11 | {dist}")
    return final

print("=== Fine-tuned Combinations ===")

# Best LR variants
lr5 = LogisticRegression(C=5.0, max_iter=3000, solver='lbfgs', class_weight='balanced', random_state=42)
lr3 = LogisticRegression(C=3.0, max_iter=3000, solver='lbfgs', class_weight='balanced', random_state=42)
lr7 = LogisticRegression(C=7.0, max_iter=3000, solver='lbfgs', class_weight='balanced', random_state=42)
svm1 = SVC(C=1.0, kernel='rbf', class_weight='balanced', random_state=42, probability=True)
svm05 = SVC(C=0.5, kernel='rbf', class_weight='balanced', random_state=42, probability=True)
svm2 = SVC(C=2.0, kernel='rbf', class_weight='balanced', random_state=42, probability=True)

print("\n--- LR(C=5) + SVM combos ---")
for lr_w, svm_w in [(3,1), (2,1), (1,1), (2,2), (3,2)]:
    ens = VotingClassifier(estimators=[('lr', lr5), ('svm', svm1)], voting='soft', weights=[lr_w, svm_w])
    ens.fit(X_tr, y_train)
    eval_model(f"LR5+SVM1 ({lr_w}:{svm_w})", ens.predict(X_te))

print("\n--- LR C sweep + SVM ---")
for C_lr in [3.0, 5.0, 7.0]:
    for C_svm in [0.5, 1.0, 2.0]:
        lr = LogisticRegression(C=C_lr, max_iter=3000, solver='lbfgs', class_weight='balanced', random_state=42)
        svm = SVC(C=C_svm, kernel='rbf', class_weight='balanced', random_state=42, probability=True)
        ens = VotingClassifier(estimators=[('lr', lr), ('svm', svm)], voting='soft', weights=[2,1])
        ens.fit(X_tr, y_train)
        eval_model(f"LR(C={C_lr})+SVM(C={C_svm}) (2:1)", ens.predict(X_te))

print("\n--- Triple combos ---")
knn5 = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn3 = KNeighborsClassifier(n_neighbors=3, weights='distance')

ens = VotingClassifier(estimators=[('lr', lr5), ('svm', svm1), ('knn', knn3)], voting='soft', weights=[3,2,1])
ens.fit(X_tr, y_train)
eval_model("LR5+SVM1+KNN3 (3:2:1)", ens.predict(X_te))

ens = VotingClassifier(estimators=[('lr', lr5), ('svm', svm1), ('knn', knn5)], voting='soft', weights=[3,2,1])
ens.fit(X_tr, y_train)
eval_model("LR5+SVM1+KNN5 (3:2:1)", ens.predict(X_te))

# With feature selection
print("\n--- LR C=5 with feature selection ---")
lr_full = LogisticRegression(C=5.0, max_iter=3000, solver='lbfgs', class_weight='balanced', random_state=42)
lr_full.fit(X_tr, y_train)
importances = np.abs(lr_full.coef_).max(axis=0)
top_k = np.argsort(importances)[::-1]

for k in [100, 130, 150, 170, 200]:
    X_tr_sub = X_tr[:, top_k[:k]]
    X_te_sub = X_te[:, top_k[:k]]
    lr_sub = LogisticRegression(C=5.0, max_iter=3000, solver='lbfgs', class_weight='balanced', random_state=42)
    lr_sub.fit(X_tr_sub, y_train)
    eval_model(f"LR(C=5) top-{k} feats", lr_sub.predict(X_te_sub))

# Also try LR C=5 with top features + SVM
print("\n--- LR(C=5) top feats + SVM ---")
for k in [130, 150]:
    X_tr_sub = X_tr[:, top_k[:k]]
    X_te_sub = X_te[:, top_k[:k]]
    lr_sub = LogisticRegression(C=5.0, max_iter=3000, solver='lbfgs', class_weight='balanced', random_state=42)
    svm_sub = SVC(C=1.0, kernel='rbf', class_weight='balanced', random_state=42, probability=True)
    ens = VotingClassifier(estimators=[('lr', lr_sub), ('svm', svm_sub)], voting='soft', weights=[2,1])
    ens.fit(X_tr_sub, y_train)
    eval_model(f"LR5+SVM1 top-{k} (2:1)", ens.predict(X_te_sub))

print("\nDone!")
