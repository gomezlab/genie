import pandas as pd
from skopt import BayesSearchCV
# from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedGroupKFold
import os
import numpy as np
from math import sqrt
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve, auc
import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--drug', type=str, default='folfox')
parser.add_argument('--outcome', type=str, default='OS')
parser.add_argument('--data_type', type=str, default='comb')
parser.add_argument('--lasso_c', type=str, default=2)

args = parser.parse_args()

drug = args.drug
outcome = args.outcome
data_type = args.data_type
lasso_c = args.lasso_c
lasso_c = float(lasso_c)

today = datetime.date.today()
today_str = today.strftime('%m%d%y')

# calculate auprc 95% ci for each model


def auroc_ci(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    mean = roc_auc
    std = sqrt(roc_auc * (1.0 - roc_auc) / len(y_true))
    low = mean - std
    high = mean + std
    return low, mean, high


def auprc_ci(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    mean = pr_auc
    std = sqrt(pr_auc * (1.0 - pr_auc) / len(y_true))
    low = mean - std
    high = mean + std
    return low, mean, high


data = pd.read_csv('../data/crc_{}_mut_cna_clin.csv'.format(drug), index_col=0)
data = data.dropna(subset=[outcome])

# create 5 train, test splits, within each train, create 5 train, valid splits
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1)
groups = data.index
X = data[[col for col in data.columns if 'mut_' in col or 'cna_' in col or 'clin_' in col]]
y = data[outcome]
input_shape = [X.shape[1]]


res_df = pd.DataFrame(columns=['fold', 'val_auroc_mean', 'val_auroc_ci', 'test_auroc_mean',
                               'test_auroc_ci', 'test_auprc_mean', 'test_auprc_ci'])

fold_count = 0
for train_index, test_index in skf.split(X, y, groups):
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y[train_index]
    y_test = y[test_index]

    if data_type != 'comb':
        X_train = X_train[[
            col for col in data.columns if '{}_'.format(data_type) in col]]
        X_test = X_test[[
            col for col in data.columns if '{}_'.format(data_type) in col]]
    else:
        X_train = X_train[[
            col for col in data.columns if 'mut_' in col or 'cna_' in col or 'clin_' in col]]
        X_test = X_test[[
            col for col in data.columns if 'mut_' in col or 'cna_' in col or 'clin_' in col]]
        
    #use lasso to select features
    lasso = LogisticRegression(penalty='l1', C=lasso_c, solver='liblinear')
    lasso.fit(X_train, y_train)
    lasso_coef = lasso.coef_
    lasso_coef = lasso_coef[0]
    lasso_coef = pd.DataFrame({'feature': X_train.columns, 'coef': lasso_coef})
    lasso_coef = lasso_coef[lasso_coef['coef'] != 0]
    X_train = X_train[lasso_coef['feature']]
    X_test = X_test[lasso_coef['feature']]
    
    print('X_train shape: {}, X_test shape: {}'.format(X_train.shape, X_test.shape))
    print('y_train shape: {}, y_test shape: {}'.format(y_train.shape, y_test.shape))

     # Elastic-Net mixing parameter. Takes values 0<=l1_ratio<=1. Incremented by 0.1
    l1_ratio = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    # Inverse of regularization strength. Smaller values specify stronger regularization.
    en_C = [0.01, 0.1, 1, 10, 100]
    # Solver. Only saga can be used when using penalty = "elasticnet"
    solver = ['saga']
    # Penalty Type
    penalty = ['elasticnet']

    random_grid = {'en_C': en_C,
                    'l1_ratio': l1_ratio,
                   'solver': solver,
                   'penalty': penalty}

    lr = LogisticRegression()

    lr_random = BayesSearchCV(lr, random_grid, n_iter=150, cv=5,
                              verbose=2, scoring='roc_auc', random_state=42, n_jobs=-1)

    lr_random.fit(X_train, y_train)

    results = pd.DataFrame(lr_random.cv_results_)
    results.sort_values(by='rank_test_score').to_csv(
        '../results/hp_search/results_rf_{}_{}_{}_{}.csv'.format(drug, outcome, data_type, today_str))

    best_lr = lr_random.best_estimator_

    y_pred = best_lr.predict_proba(X_test)[:, 1]
    test_auroc = auroc_ci(y_test, y_pred)
    test_auroc_mean = test_auroc[1]
    test_auroc_ci = str(test_auroc[0]) + '-' + str(test_auroc[2])
    test_auprc = auprc_ci(y_test, y_pred)
    test_auprc_mean = test_auprc[1]
    test_auprc_ci = str(test_auprc[0]) + '-' + str(test_auprc[2])

    val_auroc = lr_random.best_score_
    val_auroc_mean = val_auroc
    val_auroc_std = str(val_auroc - results['std_test_score'][lr_random.best_index_]) + '-' + str(
        val_auroc + results['std_test_score'][lr_random.best_index_])
    val_auroc_ci = str(val_auroc - 2*results['std_test_score'][lr_random.best_index_]) + '-' + str(
        val_auroc + 2*results['std_test_score'][lr_random.best_index_])
    res_df.loc[fold_count] = [fold_count, val_auroc_mean, val_auroc_ci,
                              test_auroc_mean, test_auroc_ci, test_auprc_mean, test_auprc_ci]

    del lr_random
    del best_lr

    fold_count += 1

ave_val_auroc = res_df['val_auroc_mean'].mean()
variance = res_df['val_auroc_mean'].var()


out_folder = '../results/runs/{}'.format(today_str)
# if out_folder does not exist, create it
if not os.path.exists(out_folder):
    os.mkdir(out_folder)
    res_df.to_csv('../results/runs/{}/results_rf_{}_{}_{}_{}.csv'.format(
        today_str, drug, outcome, data_type, today_str))
else:
    res_df.to_csv('../results/runs/{}/results_rf_{}_{}_{}_{}.csv'.format(
        today_str, drug, outcome, data_type, today_str))