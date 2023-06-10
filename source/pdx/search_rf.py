import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--drug', type=str, default='ib')
parser.add_argument('--outcome', type=str, default='OS')
parser.add_argument('--data_type', type=str, default='comb')

args = parser.parse_args()

drug = args.drug
outcome = args.outcome
data_type = args.data_type

import datetime
today = datetime.date.today()
today_str = today.strftime('%m%d%y')


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from skopt import BayesSearchCV

#pdx data for training
data = pd.read_csv('../data/pdx_act_mut_cpr.csv', index_col=0)
X = data[[col for col in data.columns if 'mut_' in col or 'act_' in col in col]]
y = data['CRorPR']
X_train = X.copy()
y_train = y.copy()

#clinical data for test set
test = pd.read_csv('../data/crc_ib_mut_act.csv', index_col=0)
test = test.dropna(subset=[outcome])
X_test = test[[col for col in test.columns if 'mut_' in col or 'act_' in col in col]]
y_test = test[outcome]


if data_type != 'comb':
    X_train = X_train[[col for col in data.columns if '{}_'.format(data_type) in col]]
    X_test = X_test[[col for col in data.columns if '{}_'.format(data_type) in col]]
else:
    X_train = X_train[[col for col in data.columns if 'mut_' in col or 'act_' in col in col]]
    X_test = X_test[[col for col in data.columns if 'mut_' in col or 'act_' in col in col]]

# Number of trees in random forest
n_estimators = [500, 750, 1000, 1250, 1500]
# Number of features to consider at every split
max_features = ['auto','sqrt']
# Maximum number of levels in tree
max_depth = [20, 40, 60, 80, 100, 120]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 3, 4, 6]
# Minimum number of samples required at each leaf node
min_samples_leaf = [2, 4, 6, 8]
# Method of selecting samples for training each tree
bootstrap = [True]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestClassifier()

rf_random = BayesSearchCV(rf, random_grid, n_iter = 150, cv = 5, verbose=2, scoring='roc_auc', random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)
results = pd.DataFrame(rf_random.cv_results_)
results.sort_values(by='rank_test_score').to_csv('../results/hp_search/pdx/results_rf_{}_{}_{}_{}.csv'.format(drug, outcome, data_type, today_str))
best_rf = rf_random.best_estimator_

from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve, auc
from math import sqrt

#calculate auprc 95% ci for each model
def auroc_ci(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    mean = roc_auc
    std = sqrt(roc_auc * (1.0 - roc_auc) / len(y_true))
    low  = mean - std
    high = mean + std
    return low, mean, high

def auprc_ci(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    mean = pr_auc
    std = sqrt(pr_auc * (1.0 - pr_auc) / len(y_true))
    low  = mean - std
    high = mean + std
    return low, mean, high

y_pred = best_rf.predict_proba(X_test)[:,1]
test_auroc = auroc_ci(y_test, y_pred)
test_auroc_mean = test_auroc[1]
test_auroc_ci = str(test_auroc[0]) + '-' + str(test_auroc[2])
test_auprc = auprc_ci(y_test, y_pred)
test_auprc_mean = test_auprc[1]
test_auprc_ci = str(test_auprc[0]) + '-' + str(test_auprc[2])

val_auroc = rf_random.best_score_

res_df = pd.DataFrame({'drug': [drug], 'outcome': [outcome], 'data_type': [data_type], 'val_auroc': [val_auroc], 'test_auroc_mean': [test_auroc_mean], 'test_auroc_ci': [test_auroc_ci], 'test_auprc_mean': [test_auprc_mean], 'test_auprc_ci': [test_auprc_ci]})

#if results csv file exists, append to it, otherwise create it
try:
    results = pd.read_csv('../results/runs/pdx/res_rf_{}.csv'.format(today_str), index_col=0)
    results = pd.concat([results, res_df])
    results.to_csv('../results/runs/pdx/res_rf_{}.csv'.format(today_str))
except:
    res_df.to_csv('../results/runs/pdx/res_rf_{}.csv'.format(today_str))

test_preds = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred}, index=X_test.index)
test_preds.to_csv('../results/preds/pdx/preds_rf_{}_{}_{}_{}.csv'.format(drug, outcome, data_type, today_str), index=True, index_label='record_id')