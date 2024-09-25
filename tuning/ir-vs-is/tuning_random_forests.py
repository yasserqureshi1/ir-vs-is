import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from statistics import mode
from xgboost import XGBClassifier

import itertools
import time
import random
import joblib
import sys

import multiprocessing

np.random.seed(0)
random.seed(0)

job_number = int(sys.argv[1])

n_groups = 60

# df of all fold records
df_results_path = '/home/eng/esrdfn/tuning/ir-vs-is/results/results.dat'

# File to save results to
file = f'random-forests/random_forests_results_{job_number}.csv'

# Model type (either `random forests` or `xgboost`)
model_type = 'random forests'

# Path to store dataframes
df_path = '/home/eng/esrdfn/tuning/ir-vs-is/final-data/'


# ---------------------- MODEL PARAMS ----------------------
random_forest_params = [
    # n estimators
    [100,200,300,400],
    # criterion
    ['gini', 'entropy'],
    # max depth
    [5, 10, 15],
    #min samples split
    [2,3,4],
    # min samples leaf
    [1,2,3],  
    # max features
    ['sqrt', 'log2'],
    #bootstrap
    [False, True]   
]

n_jobs = 1


def get_track_prediction(y_true, scores, preds, groups):
    '''
    Combines segment predictions and returns whole track predictions
    '''
    unique_groups = groups.unique()
    track_preds = []
    track_true = []
    avg_scores = []
    for val in unique_groups:
        indexes = np.where(groups == val)[0]
        track_true.append(mode(y_true.values[indexes]))
        avg_scores.append(np.mean(scores[indexes]))
        if np.mean(preds[indexes]) >= 0.5:
            track_preds.append(1)
        else:
            track_preds.append(0)

    return track_true, track_preds, avg_scores


def produce_report(y_test, y_pred, scores):
    '''
    Produces a performance report as a dict
    '''
    data = {
        'balanced accuracy': metrics.balanced_accuracy_score(y_test,y_pred),
        'roc auc': metrics.roc_auc_score(y_test, scores),
        'f1 score (ir)': metrics.f1_score(y_test,y_pred),
        'precision (ir)': metrics.precision_score(y_test, y_pred),
        'recall (ir)': metrics.recall_score(y_test, y_pred),
        'f1 score (is)': metrics.f1_score(np.array(y_test)*-1 +1, np.array(y_pred)*-1 +1),
        'precision (is)': metrics.precision_score(np.array(y_test)*-1 +1, np.array(y_pred)*-1 +1),
        'recall (is)': metrics.recall_score(np.array(y_test)*-1 +1, np.array(y_pred)*-1 +1),
    }
    return data


def model(model_type, x_train_os, y_train_os, x_train, y_train, x_test, y_test, params):
    '''
    Runs model and returns performance report as dict
    '''
    if model_type == 'random forests':
        model = RandomForestClassifier(
            **params,
            n_jobs=n_jobs,
            random_state=0
        )
    elif model_type == 'xgboost':
        model = XGBClassifier(
            **params,
            n_jobs=n_jobs,
            random_state=0
        )
    else:
        raise ValueError('`model_type` does not exist')
    model.fit(x_train_os, y_train_os)
    y_pred = model.predict(x_test)
    scores = model.predict_proba(x_test)[:,1]
    track_true, track_preds, avg_scores = get_track_prediction(y_test['Target'], scores, y_pred, y_test['TrackGroup'])
    test_results = produce_report(track_true, track_preds, avg_scores)

    y_pred = model.predict(x_train)
    scores = model.predict_proba(x_train)[:,1]
    track_true, track_preds, avg_scores = get_track_prediction(y_train['Target'], scores, y_pred, y_train['TrackGroup'])
    train_results = produce_report(track_true, track_preds, avg_scores)
    return (test_results, train_results)


def run_model(model_type, metadata, param, q):
    fold_id = metadata['fold id']
    window_size = metadata['window size']
    overlap = metadata['window overlap']
    data_base = metadata['basename']

    x_train_os = joblib.load(df_path + f'xtrainos_{data_base}')
    y_train_os = joblib.load(df_path + f'ytrainos_{data_base}')
    x_train = joblib.load(df_path + f'xtrain_{data_base}')
    y_train = joblib.load(df_path + f'ytrain_{data_base}')
    x_test = joblib.load(df_path + f'xtest_{data_base}')
    y_test = joblib.load(df_path + f'ytest_{data_base}')
    
    if model_type == 'random forests':
        n_estimators = param['n_estimators']
        criterion = param['criterion']
        max_depth = param['max_depth']
        min_samples_split = param['min_samples_split']
        min_samples_leaf = param['min_samples_leaf']
        max_features = param['max_features']
        bootstrap = param['bootstrap']

        test_perf, train_perf = model(model_type, x_train_os, y_train_os, x_train, y_train, x_test, y_test, param)

        result_string = f'{str(fold_id)},{window_size},{overlap},{n_estimators},{criterion},{max_depth},{min_samples_split},{min_samples_leaf},{max_features},{bootstrap},{test_perf["balanced accuracy"]},{test_perf["roc auc"]},{train_perf["balanced accuracy"]},{train_perf["roc auc"]}\n'
        q.put(result_string)

        print(f'{str(fold_id)},{window_size},{overlap},{n_estimators},{criterion},{max_depth},{min_samples_split},{min_samples_leaf},{max_features},{bootstrap},{test_perf["balanced accuracy"]},{test_perf["roc auc"]},{train_perf["balanced accuracy"]},{train_perf["roc auc"]}')
    
    elif model_type == 'xgboost':
        learning_rate = param['learning_rate']
        n_estimators = param['n_estimators']
        max_depth = param['max_depth']
        subsample = param['subsample']
        colsample_bytree = param['colsample_bytree']
        reg_alpha = param['reg_alpha']
        reg_lambda = param['reg_lambda']
        min_child_weight = param['min_child_weight']
        
        test_perf, train_perf = model(model_type, x_train_os, y_train_os, x_test, y_test, param)

        result_string = f'{str(fold_id)},{window_size},{overlap},{learning_rate},{n_estimators},{max_depth},{subsample},{colsample_bytree},{reg_alpha},{reg_lambda},{min_child_weight},{test_perf["balanced accuracy"]},{test_perf["roc auc"]},{train_perf["balanced accuracy"]},{train_perf["roc auc"]}\n'
        q.put(result_string)

        print(f'{fold_id},{window_size},{overlap},{learning_rate},{n_estimators},{max_depth},{subsample},{colsample_bytree},{reg_alpha},{reg_lambda},{min_child_weight},{test_perf["balanced accuracy"]},{test_perf["roc auc"]},{train_perf["balanced accuracy"]},{train_perf["roc auc"]}')
    
    else:
        raise ValueError('`model_type` does not exist')


def list_of_params(model_type, params):
    '''
    Create a list of dictionaries of parameters
    '''
    all_params = []
    if model_type == 'random forests':
        for (n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap) in itertools.product(*params):
            param = {
                'n_estimators': n_estimators,
                'criterion': criterion,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'max_features': max_features,
                'bootstrap': bootstrap
            }
            all_params.append(param)
    else:
        raise ValueError('`model_type` does not exist')
    return all_params


def listener(q):
    '''
    Listens to messages on queue and writes to file
    '''
    with open(file, 'a+') as f:
        while 1:
            m = q.get()
            if m == 'kill':
                f.write('killed')
                break
            f.write(str(m))
            f.flush()


if __name__ == '__main__':

    # ---------------------- INITALISE RESULTS CSV ----------------------
    if model_type == 'random forests':
        try:
            RESULTS = open(file, 'r+')
        except:
            RESULTS = open(file, 'w+')
            RESULTS.write('train trials,window size,window overlap,n estimators,criterion,max depth,min samples split,min samples leaf,max features,bootstrap,balanced accuracy,roc auc score,train balanced accuracy,train roc auc score\n')

    elif model_type == 'xgboost':
        try:
            RESULTS = open(file, 'r+')
        except:
            RESULTS = open(file, 'w+')
            RESULTS.write('train trials,window size,window overlap,learning rate,n estimators,max depth,subsample,colsample bytree,gamma,reg alpha,reg lambda,min child weight,balanced accuracy,roc auc score,train balanced accuracy,train roc auc score\n')
    else:
        raise ValueError('`model_type` does not exist')
    
    RESULTS.close()
    
    # ---------------------- CV FOLDS ----------------------

    if model_type == 'random forests':
        all_params = list_of_params(model_type, random_forest_params)
    else:
        raise ValueError('`model_type` does not exist')

    df_fold_metadata = joblib.load(df_results_path)
    values = df_fold_metadata.to_dict(orient='records')

    param_combinations = [i for i in itertools.product(values, all_params)]

    group_size = len(param_combinations) // n_groups
    groups = [param_combinations[i:i + group_size] for i in range(0, len(param_combinations), group_size)]
    params_to_use = groups[job_number]

    print('TOTAL PARAM COMBINATIONS: ', len(param_combinations))
    print('TOTAL PARAM USING: ', len(params_to_use))
    
    # Initalise time
    t0 = time.time()

    manager = multiprocessing.Manager()
    q = manager.Queue()

    with multiprocessing.Pool() as pool:
        watcher = pool.apply_async(listener, (q,))

        pool.starmap(run_model, [
            (model_type, metadata, param, q) 
            for metadata, param in params_to_use
        ])
        
    print('Completed in ', time.time()-t0)
