import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--drug', type=str, default='egfr')
parser.add_argument('--outcome', type=str, default='OS')
parser.add_argument('--data_type', type=str, default='comb')
parser.add_argument('--lasso_c', type=str, default=2)

args = parser.parse_args()

drug = args.drug
outcome = args.outcome
data_type = args.data_type
lasso_c = args.lasso_c
lasso_c = float(lasso_c)

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
from sklearn.model_selection import cross_val_score, StratifiedGroupKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from skopt import BayesSearchCV


print('drug: {}, outcome: {}, data_type: {}'.format(drug, outcome, data_type))

data = pd.read_csv('../../data/crc_{}_mut_cna_fus_clin.csv'.format(drug))
data.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
data = data.dropna(subset=[outcome])
data.reset_index(inplace=True, drop=True)

#test set is VICC
# X_train = data[data['id_institution'].isin(['DFCI', 'MSKCC'])]
# y = train[outcome]
# X_test = data[data['id_institution'] == 'VICC']
# y_test = test[outcome]

#create 5 train, test splits, within each train, create 5 train, valid splits
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1)
groups = data['id']
X = data[[col for col in data.columns if 'mut_' in col or 'cna_' in col or 'clin_' in col or 'fus' in col]]
y = data[outcome]

input_shape = [X.shape[1]]

def build_model(n_hidden=1, n_neurons=100, dropout=0.4, activation = "relu", learning_rate=3e-3):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Activation(activation))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(loss="binary_crossentropy", metrics=['AUC'], optimizer=optimizer)
    return model

res_df = pd.DataFrame(columns=['fold', 'val_auroc_mean', 'val_auroc_ci', 'test_auroc_mean', 'test_auprc_mean'])
fold_count = 0
for train_index, test_index in skf.split(X, y, groups):
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y[train_index]
    y_test = y[test_index]

    # #move all the samples in X_test, where clin_stage_dx is 0 to X_train
    # mov_test_idxs = X_test[X_test['clin_stage_dx_iv'] == 0].index

    # X_train = pd.concat([X_train, X_test.loc[mov_test_idxs]])
    # X_test = X_test[X_test['clin_stage_dx_iv'] == 1]
    # use mov_test_idxs to move the y_test values to y_train
    # y_train = pd.concat([y_train, y_test.loc[mov_test_idxs]])
    # y_test.drop(mov_test_idxs, inplace=True)
    # #now get a sample of X_train of the same len as mov_test_idxs, where clin_stage_dx_iv == 1
    # X_1_samp = X_train[X_train['clin_stage_dx_iv'] == 1].sample(n=len(mov_test_idxs), random_state=1)
    # y_1_samp = y_train.loc[X_1_samp.index]
    # X_test = pd.concat([X_test, X_1_samp])
    # y_test = pd.concat([y_test, y_1_samp])
    # X_train.drop(X_1_samp.index, inplace=True)
    # y_train.drop(X_1_samp.index, inplace=True)


    if (data_type != 'comb') & (data_type != 'nonclin'):
        X_train = X_train[[col for col in data.columns if '{}_'.format(data_type) in col]]
        X_test = X_test[[col for col in data.columns if '{}_'.format(data_type) in col]]
    elif data_type == 'nonclin':
        X_train = X_train[[col for col in data.columns if 'mut_' in col or 'cna_' in col or 'fus' in col]]
        X_test = X_test[[col for col in data.columns if 'mut_' in col or 'cna_' in col or 'fus' in col]]
    else:
        X_train = X_train[[col for col in data.columns if 'mut_' in col or 'cna_' in col or 'clin_' in col or 'fus' in col]]
        X_test = X_test[[col for col in data.columns if 'mut_' in col or 'cna_' in col or 'clin_' in col or 'fus' in col]]

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
    results.sort_values(by='rank_test_score').to_csv('../../results/hp_search/results_rf_lasso_{}_{}_{}_{}_{}.csv'.format(drug, outcome, data_type, today_str, fold_count))
    best_rf = rf_random.best_estimator_

    y_pred = best_rf.predict_proba(X_test)[:,1]
        
    test_auroc_mean = roc_auc_score(y_test, y_pred)
        
    test_auprc_mean = average_precision_score(y_test, y_pred)

    val_auroc = rf_random.best_score_
    val_auroc_mean = val_auroc
    val_auroc_std = str(val_auroc - results['std_test_score'][rf_random.best_index_]) + '-' + str(val_auroc + results['std_test_score'][rf_random.best_index_])
    val_auroc_ci = str(val_auroc - 2*results['std_test_score'][rf_random.best_index_]) + '-' + str(val_auroc + 2*results['std_test_score'][rf_random.best_index_])
    res_df.loc[fold_count] = [fold_count, val_auroc_mean, val_auroc_ci, test_auroc_mean, test_auprc_mean]

    del rf_random
    del best_rf

    fold_count += 1

out_folder = '../../results/lasso_runs/{}'.format(today_str)
#if out_folder does not exist, create it
if not os.path.exists(out_folder):
    os.mkdir(out_folder)
    res_df.to_csv('../../results/lasso_runs/{}/results_rf_{}_{}_{}_{}.csv'.format(today_str, drug, outcome, data_type, today_str))
else:
    res_df.to_csv('../../results/lasso_runs/{}/results_rf_{}_{}_{}_{}.csv'.format(today_str, drug, outcome, data_type, today_str))
