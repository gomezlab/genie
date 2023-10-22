import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--drug', type=str, default='folfox')
parser.add_argument('--outcome', type=str, default='OS')
parser.add_argument('--data_type', type=str, default='comb')

args = parser.parse_args()

drug = args.drug
outcome = args.outcome
data_type = args.data_type

import datetime
today = datetime.date.today()
today_str = today.strftime('%m%d%y')


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

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import cross_val_score, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from skopt import BayesSearchCV


print('drug: {}, outcome: {}, data_type: {}'.format(drug, outcome, data_type))

data = pd.read_csv('../data/crc_{}_mut_cna_fus_clin.csv'.format(drug))
data.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
data = data.dropna(subset=[outcome])

#test set is VICC
# X_train = data[data['id_institution'].isin(['DFCI', 'MSKCC'])]
# y = train[outcome]
# X_test = data[data['id_institution'] == 'VICC']
# y_test = test[outcome]

#create 5 train, test splits, within each train, create 5 train, valid splits
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1)
groups = data['id']
X = data[[col for col in data.columns if 'mut_' in col or 'cna_' in col or 'clin_' in col or 'fus_' in col]]
y = data[outcome]


res_df = pd.DataFrame(columns=['fold', 'val_auroc_mean', 'val_auroc_ci', 'test_auroc_mean', 'test_auprc_mean'])
fold_count = 0
for train_index, test_index in skf.split(X, y, groups):
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y[train_index]
    y_test = y[test_index]

    #move all the samples in X_test, where clin_stage_dx is 0 to X_train
    mov_test_idxs = X_test[X_test['clin_stage_dx_iv'] == 0].index

    X_train = pd.concat([X_train, X_test.loc[mov_test_idxs]])
    X_test = X_test[X_test['clin_stage_dx_iv'] == 1]
    # use mov_test_idxs to move the y_test values to y_train
    y_train = pd.concat([y_train, y_test.loc[mov_test_idxs]])
    y_test.drop(mov_test_idxs, inplace=True)
    #now get a sample of X_train of the same len as mov_test_idxs, where clin_stage_dx_iv == 1
    X_1_samp = X_train[X_train['clin_stage_dx_iv'] == 1].sample(n=len(mov_test_idxs), random_state=1)
    y_1_samp = y_train.loc[X_1_samp.index]
    X_test = pd.concat([X_test, X_1_samp])
    y_test = pd.concat([y_test, y_1_samp])
    X_train.drop(X_1_samp.index, inplace=True)
    y_train.drop(X_1_samp.index, inplace=True)


    if data_type != 'comb':
        X_train = X_train[[col for col in data.columns if '{}_'.format(data_type) in col]]
        X_test = X_test[[col for col in data.columns if '{}_'.format(data_type) in col]]
    else:
        X_train = X_train[[col for col in data.columns if 'mut_' in col or 'cna_' in col or 'clin_' in col or 'fus_' in col]]
        X_test = X_test[[col for col in data.columns if 'mut_' in col or 'cna_' in col or 'clin_' in col or 'fus_' in col]]
    print('X_train shape: {}, X_test shape: {}'.format(X_train.shape, X_test.shape))
    print('y_train shape: {}, y_test shape: {}'.format(y_train.shape, y_test.shape))
    clf_xgb = XGBClassifier(tree_method='gpu_hist', use_label_encoder=False)

    param_dist = {'n_estimators': [20, 50, 100, 200, 400],
                'learning_rate': [0.03, 0.05, 0.075, 0.1, 0.3, 0.5],
                'subsample': [0.4, 0.6, 1.0],
                'max_depth': [6, 8, 12, 20],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'min_child_weight': [2, 4, 6],
                'reg_alpha': [0, 0.5, 1, 5],
                'reg_lambda': [0, 0.5, 1, 5],
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
    results.sort_values(by='rank_test_score').to_csv('../results/hp_search/results_xgb_{}_{}_{}_{}.csv'.format(drug, outcome, data_type, today_str))

    best_xgb = bayes_xgb.best_estimator_

    best_xgb = bayes_xgb.best_estimator_

    y_pred = best_xgb.predict_proba(X_test)[:,1]
    
    test_auroc_mean = roc_auc_score(y_test, y_pred)
        
    test_auprc_mean = average_precision_score(y_test, y_pred)

    val_auroc = bayes_xgb.best_score_
    val_auroc_mean = val_auroc
    val_auroc_std = str(val_auroc - results['std_test_score'][bayes_xgb.best_index_]) + '-' + str(val_auroc + results['std_test_score'][bayes_xgb.best_index_])
    val_auroc_ci = str(val_auroc - 2*results['std_test_score'][bayes_xgb.best_index_]) + '-' + str(val_auroc + 2*results['std_test_score'][bayes_xgb.best_index_])
    res_df.loc[fold_count] = [fold_count, val_auroc_mean, val_auroc_ci, test_auroc_mean, test_auprc_mean]
    
    del bayes_xgb
    del best_xgb

    fold_count += 1

ave_val_auroc = res_df['val_auroc_mean'].mean()
variance = res_df['val_auroc_mean'].var()


out_folder = '../results/runs/{}'.format(today_str)
#if out_folder does not exist, create it
if not os.path.exists(out_folder):
    os.mkdir(out_folder)
    res_df.to_csv('../results/runs/{}/results_xgb_{}_{}_{}_{}.csv'.format(today_str, drug, outcome, data_type, today_str))
else:
    res_df.to_csv('../results/runs/{}/results_xgb_{}_{}_{}_{}.csv'.format(today_str, drug, outcome, data_type, today_str))
