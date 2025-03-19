import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')

# load data
ML_data = pd.read_csv('./ML_data_rate.csv')
ML_data = ML_data[ML_data['label']!='Y'].reset_index(drop=True)
print( Counter(ML_data['label']) )
print(ML_data.shape)

enc = LabelEncoder()   # LabelEncoder DN, HCC
ML_data['label'] = enc.fit_transform(ML_data['label'])

# parameters
C = 0.2
solver = 'saga' 
num = 34

for geneId in range(1,11):
    print(f'gene {geneId}--------------')
    # data
    X_inter = ML_data[ML_data['cohort'] == 'inter'].iloc[:,0:num+geneId].reset_index(drop=True)
    y_inter = ML_data.loc[ML_data['cohort'] == 'inter','label'].reset_index(drop=True)
    X_inter = np.array(X_inter)
    # print(X_inter.shape)
    # print( Counter(y_inter) )

    X_exter = ML_data[ML_data['cohort'] == 'exter'].iloc[:,0:num+geneId].reset_index(drop=True)
    y_exter = ML_data.loc[ML_data['cohort'] == 'exter','label'].reset_index(drop=True)
    X_exter = np.array(X_exter)
    # print(X_exter.shape)
    # print( Counter(y_exter) )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_inter)
    X_test = scaler.transform(X_exter)
    y_train = np.array(y_inter)
    y_test = np.array(y_exter)
    
    results = []
    # model
    for C in param_grid['C']:
        for solver in param_grid['solver']:
            logistic_model = LogisticRegression(penalty='l1', solver=solver, random_state=42, max_iter=10000, n_jobs=-1, C=C)
            acc_list, roc_auc_list = [], []
            for repeat in range(10):  # TODO
                kf = KFold(n_splits=5, shuffle=True, random_state=repeat)
                y_true, y_pred, y_prob = [], [], []
                for train_index, val_index in kf.split(X_train):
                    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
                    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
                    logistic_model.fit(X_train_fold, y_train_fold)
                    y_pred_fold = logistic_model.predict(X_val_fold)
                    y_prob_fold = logistic_model.predict_proba(X_val_fold)[:,1]
                    y_pred.extend(y_pred_fold)
                    y_prob.extend(y_prob_fold)
                    y_true.extend(y_val_fold)
                acc = accuracy_score(y_true, y_pred)
                roc_auc = roc_auc_score(y_true, y_prob)
                acc_list.append(acc)
                roc_auc_list.append(roc_auc)
            results.append({'C': C, 'solver': solver,
                        'accuracy_mean': np.mean(acc_list), 'accuracy_std': np.std(acc_list),
                        'roc_auc_mean': np.mean(roc_auc_list), 'roc_auc_std': np.std(roc_auc_list),
                        'accuracy_list': acc_list, 'roc_auc_list': roc_auc_list})

    results_df = pd.DataFrame(results)
    best_params = results_df.loc[results_df['accuracy_mean'].idxmax()].to_dict()
    print(f"The best params are {best_params}")
    print(f"The best score is {best_params['accuracy_mean']}:")
    
    acc_list, roc_auc_list = [], []
    for i in range(10):
        logistic_best = LogisticRegression(penalty='l1', solver=best_params['solver'], random_state=i, max_iter=10000, n_jobs=-1, C=best_params['C'])
        logistic_best.fit(X_train, y_train)
        y_pred_exter = logistic_best.predict(X_test)
        y_prob_exter = logistic_best.predict_proba(X_test)[:,1]
        acc = accuracy_score(y_test, y_pred_exter)
        roc_auc = roc_auc_score(y_test, y_prob_exter)
        acc_list.append(acc)
        roc_auc_list.append(roc_auc)

    results.append({'C': 'exter', 'solver': 'exter',
                'accuracy_mean': np.mean(acc_list), 'accuracy_std': np.std(acc_list),
                'roc_auc_mean': np.mean(roc_auc_list), 'roc_auc_std': np.std(roc_auc_list),
                'accuracy_list': acc_list, 'roc_auc_list': roc_auc_list})

    print(f"The accuracy of exter cohort is {np.mean(acc_list)}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(f'./results/Integration_gene{geneId}.csv', index=False)

# results
results = []
for file_idx in range(1,11):
    df = pd.read_csv(f'./results/Integration_gene{file_idx}.csv')
    df_train = df.loc[df['C'] != 'exter']
    df_test = df.loc[df.index[-1], ['C', 'solver', 'accuracy_mean', 'accuracy_std', 'roc_auc_mean', 'roc_auc_std']].to_dict()
    best_params = df_train.loc[df_train['accuracy_mean'].idxmax(), ['C', 'solver', 'accuracy_mean', 'accuracy_std', 'roc_auc_mean', 'roc_auc_std']].to_dict()
    best_params['features'] = file_idx
    best_params['accuracy_mean_test'] = df_test['accuracy_mean']
    best_params['accuracy_std_test'] = df_test['accuracy_std']
    best_params['roc_auc_mean_test'] = df_test['roc_auc_mean']
    best_params['roc_auc_std_test'] = df_test['roc_auc_std']
    results.append(best_params)
results_df = pd.DataFrame(results)
results_df['accuracy'] = results_df.apply(lambda row: format_mean_std(row['accuracy_mean'], row['accuracy_std']), axis=1)
results_df['roc_auc'] = results_df.apply(lambda row: format_mean_std(row['roc_auc_mean'], row['roc_auc_std']), axis=1)
results_df['accuracy_test'] = results_df.apply(lambda row: format_mean_std(row['accuracy_mean_test'], row['accuracy_std_test']), axis=1)
results_df['roc_auc_test'] = results_df.apply(lambda row: format_mean_std(row['roc_auc_mean_test'], row['roc_auc_std_test']), axis=1)
results_df = results_df.drop(['accuracy_mean', 'accuracy_std', 'roc_auc_mean', 'roc_auc_std',
                            'accuracy_mean_test', 'accuracy_std_test', 'roc_auc_mean_test', 'roc_auc_std_test',
                            'C', 'solver'], axis=1)

