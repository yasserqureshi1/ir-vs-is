import pandas as pd
import numpy as np

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
df_results_path = '/home/eng/esrdfn/tuning/multiclass/results/results_multiclass.dat'

# File to save results to
file = f'xgboost/multi-class-xgboost_{job_number}.csv'

# Model type (either `random forests` or `xgboost`)
model_type = 'xgboost'

# Path to store dataframes
df_path = '/home/eng/esrdfn/tuning/multiclass/final-data/'


# ---------------------- MODEL PARAMS ----------------------
xgboost_params = [
    # learning rate
    [0.01, 0.1, 0.3],
    # n estimators
    [100,150,200],
    # max depth
    [3, 5, 7],
    #subsample
    [0.5, 0.7, 0.9],
    #colsample bytree
    [0.5, 0.7, 0.9],
    #reg alpha
    [0, 0.01, 0.1],
    #reg lambda
    [0, 0.01, 0.1],
    # min child weight
    [1, 5, 15]
]

random_forest_params = []

n_jobs = 1


def get_track_prediction(y_true, scores, preds, groups, classes):
    unique_groups = groups.unique()
    track_preds = []
    track_true = []
    avg_scores = []
    for val in unique_groups:
        indexes = np.where(groups == val)[0]
        track_true.append(mode(y_true.values[indexes]))
        score = np.mean(scores[indexes], axis=0)
        score_index = np.argmax(score)
        avg_scores.append(score[score_index])
        track_preds.append(classes[score_index])
    return track_true, track_preds, avg_scores


def produce_report(y_test, y_pred, scores):
    report = metrics.classification_report(y_test, y_pred, output_dict=True)
    data = {
        'accuracy': metrics.accuracy_score(y_test,y_pred),
        'balanced accuracy': metrics.balanced_accuracy_score(y_test,y_pred),
        'report': report
    }
    return data


def model(model_type, x_train_os, y_train_os, x_train, y_train, x_test, y_test, params):
    '''
    Runs model and returns performance report as dict
    '''
    if model_type == 'xgboost':
        model = XGBClassifier(
            **params,
            num_classes=4,
            objective='multi:softmax',
            n_jobs=n_jobs,
            random_state=0
        )
    else:
        raise ValueError('`model_type` does not exist')
    model.fit(x_train_os, y_train_os)
    y_pred = model.predict(x_test)
    scores = model.predict_proba(x_test)
    track_true, track_preds, avg_scores = get_track_prediction(y_test['Target'], scores, y_pred, y_test['TrackGroup'], model.classes_)
    test_results = produce_report(track_true, track_preds, avg_scores)

    y_pred = model.predict(x_train)
    scores = model.predict_proba(x_train)
    track_true, track_preds, avg_scores = get_track_prediction(y_train['Target'], scores, y_pred, y_train['TrackGroup'], model.classes_)
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
    
    if model_type == 'xgboost':
        learning_rate = param['learning_rate']
        n_estimators = param['n_estimators']
        max_depth = param['max_depth']
        subsample = param['subsample']
        colsample_bytree = param['colsample_bytree']
        reg_alpha = param['reg_alpha']
        reg_lambda = param['reg_lambda']
        min_child_weight = param['min_child_weight']
        
        test_perf, train_perf = model(model_type, x_train_os, y_train_os, x_train, y_train, x_test, y_test, param)

        result_string = f'{str(fold_id)},{window_size},{overlap},{learning_rate},{n_estimators},{max_depth},{subsample},{colsample_bytree},{reg_alpha},{reg_lambda},{min_child_weight},{test_perf["balanced accuracy"]},{train_perf["balanced accuracy"]}\n'
        q.put(result_string)

        print(f'{fold_id},{window_size},{overlap},{learning_rate},{n_estimators},{max_depth},{subsample},{colsample_bytree},{reg_alpha},{reg_lambda},{min_child_weight},{test_perf["balanced accuracy"]},{train_perf["balanced accuracy"]}')
    
    else:
        raise ValueError('`model_type` does not exist')


def list_of_params(model_type, xgboost_params):
    '''
    Create a list of dictionaries of parameters
    '''
    all_params = []
    if model_type == 'xgboost':
        for (learning_rate, n_estimators, max_depth, subsample, colsample_bytree, reg_alpha, reg_lambda, min_child_weight) in itertools.product(*xgboost_params):
            param = {
                'learning_rate': learning_rate,
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda,
                'min_child_weight': min_child_weight
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
            RESULTS.write('train trials,window size,window overlap,learning rate,n estimators,max depth,subsample,colsample bytree,gamma,reg alpha,reg lambda,min child weight,balanced accuracy,train balanced accuracy\n')
    else:
        raise ValueError('`model_type` does not exist')
    
    RESULTS.close()
    
    # ---------------------- CV FOLDS ----------------------
    if model_type == 'xgboost':
        all_params = list_of_params(model_type, xgboost_params)
    else:
        raise ValueError('`model_type` does not exist')

    df_fold_metadata = joblib.load(df_results_path)
    values = df_fold_metadata.to_dict(orient='records')

    param_combinations = [i for i in itertools.product(values, all_params)]

    group_size = len(param_combinations) // n_groups
    groups = [params_to_use[i:i + group_size] for i in range(0, len(param_combinations), group_size)]
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

