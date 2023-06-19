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




data = pd.read_csv('../data/crc_{}_mut_cna_fus_clin.csv'.format(drug), index_col=0)
data = data.dropna(subset=[outcome])

#test set is VICC
# X_train = data[data['id_institution'].isin(['DFCI', 'MSKCC'])]
# y = train[outcome]
# X_test = data[data['id_institution'] == 'VICC']
# y_test = test[outcome]

#create 5 train, test splits, within each train, create 5 train, valid splits
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1)
groups = data.index
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

res_df = pd.DataFrame(columns=['fold', 'val_auroc_mean', 'val_auroc_ci', 'test_auroc_mean', 'test_auroc_ci', 'test_auprc_mean', 'test_auprc_ci'])
fold_count = 0
for train_index, test_index in skf.split(X, y, groups):
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y[train_index]
    y_test = y[test_index]

    if data_type != 'comb':
        X_train = X_train[[col for col in data.columns if '{}_'.format(data_type) in col]]
        X_test = X_test[[col for col in data.columns if '{}_'.format(data_type) in col]]
    else:
        X_train = X_train[[col for col in data.columns if 'mut_' in col or 'cna_' in col or 'clin_' in col or 'fus' in col]]
        X_test = X_test[[col for col in data.columns if 'mut_' in col or 'cna_' in col or 'clin_' in col or 'fus' in col]]

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.8, test_size=0.2, random_state=1)

    keras_clf = keras.wrappers.scikit_learn.KerasClassifier(build_model)

    param_distribs = {
        "n_hidden": [1, 2, 3, 4],
        "n_neurons": [25, 50, 200, 500, 1000, 1500],
        "dropout": [0.2, 0.4, 0.6, 0.8],
        "activation": ["relu", "elu"],
        "learning_rate": [3e-5, 3e-4, 3e-3, 3e-2],
    }

    # In[ ]:
    early_stopping = keras.callbacks.EarlyStopping(
        patience=15,
        min_delta=1e-6,
        restore_best_weights=True,)

    rnd_search_cv = BayesSearchCV(keras_clf, param_distribs, n_iter=100, scoring='roc_auc', cv=5, verbose=2)

    rnd_search_cv.fit(X_train, y_train, epochs=100, batch_size=32,
                    validation_data=(X_valid, y_valid),
                    callbacks=[early_stopping])


    results = pd.DataFrame(rnd_search_cv.cv_results_)



    best_keras = rnd_search_cv.best_estimator_


    y_pred = best_keras.predict_proba(X_test)[:,1]
    test_auroc = auroc_ci(y_test, y_pred)
    test_auroc_mean = test_auroc[1]
    test_auroc_ci = str(test_auroc[0]) + '-' + str(test_auroc[2])
    test_auprc = auprc_ci(y_test, y_pred)
    test_auprc_mean = test_auprc[1]
    test_auprc_ci = str(test_auprc[0]) + '-' + str(test_auprc[2])

    val_auroc = rnd_search_cv.best_score_
    val_auroc_mean = val_auroc
    val_auroc_std = str(val_auroc - results['std_test_score'][rnd_search_cv.best_index_]) + '-' + str(val_auroc + results['std_test_score'][rnd_search_cv.best_index_])
    val_auroc_ci = str(val_auroc - 2*results['std_test_score'][rnd_search_cv.best_index_]) + '-' + str(val_auroc + 2*results['std_test_score'][rnd_search_cv.best_index_])
    res_df.loc[fold_count] = [fold_count, val_auroc_mean, val_auroc_ci, test_auroc_mean, test_auroc_ci, test_auprc_mean, test_auprc_ci]

    del rnd_search_cv
    del best_keras

    fold_count += 1

ave_val_auroc = res_df['val_auroc_mean'].mean()
variance = res_df['val_auroc_mean'].var()


out_folder = '../results/runs/{}'.format(today_str)
#if out_folder does not exist, create it
if not os.path.exists(out_folder):
    os.mkdir(out_folder)
    res_df.to_csv('../results/runs/{}/results_nn_{}_{}_{}_{}.csv'.format(today_str, drug, outcome, data_type, today_str))
else:
    res_df.to_csv('../results/runs/{}/results_nn_{}_{}_{}_{}.csv'.format(today_str, drug, outcome, data_type, today_str))