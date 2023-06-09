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
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2)

#clinical data for test set
test = pd.read_csv('../data/crc_ib_mut_act.csv', index_col=0)
X_test = test[[col for col in test.columns if 'mut_' in col or 'act_' in col in col]]
y_test = test[outcome]


if data_type != 'comb':
    X_train = X_train[[col for col in data.columns if '{}_'.format(data_type) in col]]
    X_valid = X_valid[[col for col in data.columns if '{}_'.format(data_type) in col]]
    X_test = X_test[[col for col in data.columns if '{}_'.format(data_type) in col]]
else:
    X_train = X_train[[col for col in data.columns if 'mut_' in col or 'act_' in col in col]]
    X_valid = X_valid[[col for col in data.columns if 'mut_' in col or 'act_' in col in col]]
    X_test = X_test[[col for col in data.columns if 'mut_' in col or 'act_' in col in col]]


input_shape = [X_train.shape[1]]

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

results.sort_values(by='rank_test_score').to_csv('../results/hp_search/pdx/results_nn_{}_{}_{}_{}.csv'.format(drug, outcome, data_type, today_str))

best_keras = rnd_search_cv.best_estimator_

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

y_pred = best_keras.predict_proba(X_test)[:,1]
test_auroc = auroc_ci(y_test, y_pred)
test_auroc_mean = test_auroc[1]
test_auroc_ci = str(test_auroc[0]) + '-' + str(test_auroc[2])
test_auprc = auprc_ci(y_test, y_pred)
test_auprc_mean = test_auprc[1]
test_auprc_ci = str(test_auprc[0]) + '-' + str(test_auprc[2])

val_auroc = rnd_search_cv.best_score_

res_df = pd.DataFrame({'drug': [drug], 'outcome': [outcome], 'data_type': [data_type], 'val_auroc': [val_auroc], 'test_auroc_mean': [test_auroc_mean], 'test_auroc_ci': [test_auroc_ci], 'test_auprc_mean': [test_auprc_mean], 'test_auprc_ci': [test_auprc_ci]})

#if results csv file exists, append to it, otherwise create it
try:
    results = pd.read_csv('../results/runs/pdx/res_nn_{}.csv'.format(today_str), index_col=0)
    results = pd.concat([results, res_df])
    results.to_csv('../results/runs/pdx/res_nn_{}.csv'.format(today_str))
except:
    res_df.to_csv('../results/runs/pdx/res_nn_{}.csv'.format(today_str))

test_preds = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred}, index=X_test.index)
test_preds.to_csv('../results/preds/pdx/preds_nn_{}_{}_{}_{}.csv'.format(drug, outcome, data_type, today_str), index=True, index_label='record_id')