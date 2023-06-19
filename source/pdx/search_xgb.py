import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pretrain', type=str, default='crc')
parser.add_argument('--drug', type=str, default='ib')
parser.add_argument('--outcome', type=str, default='OS')
parser.add_argument('--data_type', type=str, default='comb')

args = parser.parse_args()

pretrain = args.pretrain
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
if pretrain == 'crc':
    data = pd.read_csv('../data/pdx_act_mut_cpr_crc.csv', index_col=0)
    X = data[[col for col in data.columns if 'mut_' in col or 'act_' in col in col]]
    y = data['CRorPR']
elif pretrain == 'total':
    data = pd.read_csv('../data/pdx_act_mut_cpr_total.csv', index_col=0)
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

clf_xgb = XGBClassifier(tree_method='gpu_hist', use_label_encoder=False)

param_dist = {'n_estimators': [20, 50, 100, 200, 400],
              'learning_rate': [0.03, 0.05, 0.075, 0.1, 0.3, 0.5],
              'subsample': [0.4, 0.6, 1.0],
              'max_depth': [6, 8, 12, 20],
              'colsample_bytree': [0.6, 0.8, 1.0],
              'min_child_weight': [2, 4, 6]
             }

bayes_xgb = BayesSearchCV(clf_xgb, 
                         param_dist,
                         cv = 5,  
                         n_iter = 150, 
                         scoring = 'roc_auc', 
                         error_score = 0, 
                         verbose = 0, 
                         n_jobs = -1)
print('Fitting model...')
bayes_xgb.fit(X_train, y_train)
results = pd.DataFrame(bayes_xgb.cv_results_)
results.sort_values(by='rank_test_score').to_csv('../results/hp_search/pdx/results_xgb_{}_{}_{}_{}.csv'.format(drug, outcome, data_type, today_str))

best_xgb = bayes_xgb.best_estimator_

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

y_pred = best_xgb.predict_proba(X_test)[:,1]
test_auroc = auroc_ci(y_test, y_pred)
test_auroc_mean = test_auroc[1]
test_auroc_ci = str(test_auroc[0]) + '-' + str(test_auroc[2])
test_auprc = auprc_ci(y_test, y_pred)
test_auprc_mean = test_auprc[1]
test_auprc_ci = str(test_auprc[0]) + '-' + str(test_auprc[2])

val_auroc = bayes_xgb.best_score_

res_df = pd.DataFrame({'pretrain': [pretrain], 'outcome': [outcome], 'data_type': [data_type], 'val_auroc': [val_auroc], 'test_auroc_mean': [test_auroc_mean], 'test_auroc_ci': [test_auroc_ci], 'test_auprc_mean': [test_auprc_mean], 'test_auprc_ci': [test_auprc_ci]})

#if results csv file exists, append to it, otherwise create it
try:
    results = pd.read_csv('../results/runs/pdx/res_xgb_{}.csv'.format(today_str), index_col=0)
    results = pd.concat([results, res_df])
    results.to_csv('../results/runs/pdx/res_xgb_{}.csv'.format(today_str))
except:
    res_df.to_csv('../results/runs/pdx/res_xgb_{}.csv'.format(today_str))

test_preds = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred}, index=X_test.index)
test_preds.to_csv('../results/preds/pdx/preds_xgb_{}_{}_{}_{}.csv'.format(drug, outcome, data_type, today_str), index=True, index_label='record_id')