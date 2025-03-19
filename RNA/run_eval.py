import pickle
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsOneClassifier


savePath1 = 'evaluation/evaluation_inter_5fold.csv'

with open(f'data_inter-exter.pkl', 'rb') as f:
    X_inter2, y_inter, X_exterTest2, y_exterTest = pickle.load(f)

param = {'max_depth':1,
        'n_estimators': 30,
        'learning_rate': 0.3}

def calulate_prob(clf, X_test):
    estimator1, estimator2, estimator3 = clf.estimators_ # DN VS HCC

    probDN = estimator1.predict_proba(X_test)[:,0] + estimator2.predict_proba(X_test)[:,0]
    probHCC = estimator1.predict_proba(X_test)[:,1] + estimator3.predict_proba(X_test)[:,0]

    prob = np.concatenate([probDN[:, np.newaxis],probHCC[:, np.newaxis]], axis=1)
    prob = prob / prob.sum(axis=1, keepdims=True) #归一化
    
    return prob[:,1] # HCC prob

y_pred, y_proba, y_true, y_seed = [],[],[],[]

for idx, seed in enumerate(range(10)):  # 10次五折 TODO
    kf = KFold(n_splits=5, shuffle=True, random_state=seed) # TODO
    
    for train_index, test_index in kf.split(X_inter2): # 1/5
        X_train, X_test = X_inter2.iloc[train_index], X_inter2.iloc[test_index]
        y_train, y_test = y_inter.iloc[train_index], y_inter.iloc[test_index]
            
        # 标准化
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # 建模
        clf = OneVsOneClassifier(xgb.XGBClassifier(**param))
        clf.fit(X_train, y_train)
            
        # 进行validation预测
        y_pred.extend(clf.predict(X_test))  # 预测类别
        y_true.extend(y_test)
        y_proba.extend( calulate_prob(clf,X_test))
        y_seed.extend( [idx]*X_test.shape[0] )
        
    
y_pred = np.array(y_pred).flatten()
y_true = np.array(y_true).flatten()
y_proba = np.array(y_proba).flatten()
y_seed = np.array(y_seed).flatten()

df_test = pd.DataFrame({
    'Seed':y_seed,
    'Pred': y_pred,
    'True': y_true,
    'HCCprob':y_proba
})

df_test.to_csv(savePath1, index=False)
