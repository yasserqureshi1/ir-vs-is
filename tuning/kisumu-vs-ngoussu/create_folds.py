import joblib

import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

import multiprocessing
import itertools

import extract


files_path = '/home/eng/esrdfn/tuning/ir-vs-is/dfs/'
data_path = '/home/eng/esrdfn/tuning/kisumu-vs-ngoussu/final-data/'
results_path = '/home/eng/esrdfn/tuning/kisumu-vs-ngoussu/results/'

threshold_type = 'mutual_info' # 'gradient'

trials = [4, 5, 9, 10]

folds = [
    [4,9], [4,10], [5,10], [5,9]
]

window_sizes = np.arange(0.5, 10, 0.5)
window_overlap = np.arange(0.5, 10, 0.5)

def feature_selection(df_hyp, df_hyp_target):
    '''
    Performs Mann Whitney U-test and removes highly correlated features
    '''
    df_hyp = df_hyp.drop(columns=['TrialID'])

    sus_indexes = df_hyp_target[df_hyp_target['Target'] == 0].index.values
    res_indexes = df_hyp_target[df_hyp_target['Target'] == 1].index.values

    sus = df_hyp.loc[sus_indexes]
    res = df_hyp.loc[res_indexes]

    _, p_val = mannwhitneyu(sus, res)
    rej, _, _, _ = multipletests(p_val, alpha=0.05, method='holm')
    columns = df_hyp.columns[rej]

    df_hyp = df_hyp.reset_index(drop=True)
    corr_matrix = df_hyp.corr(method='spearman').abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]

    cols = np.setdiff1d(columns, to_drop)
    return cols


def create_threshold_list(start, end):
    current_value = start
    step_size = 1
    score_thresholds = []
    while current_value <= end:
        score_thresholds.append(int(current_value))
        if current_value < end/2:
            current_value += step_size
        else:
            current_value += min(step_size, 500)
        step_size = step_size * 1.1
    score_thresholds = np.array(score_thresholds)
    return score_thresholds


def run_score_threshold_mutual_info(df, df_target, scores):
    score_thresholds = create_threshold_list(0, max(scores))

    unique_features = []
    max_mutual_info_dict = {}
    for threshold in score_thresholds:
        mask = np.where(scores <= threshold)[0]
        df_temp = df.iloc[mask]
        df_target_temp = df_target.iloc[mask]

        indexes = df_temp[df_temp.isna().any(axis=1)].index
        df_temp = df_temp.drop(index=indexes)
        df_target_temp = df_target_temp.drop(index=indexes)

        df_temp = extract.remove_nans(df_temp)        

        df_temp = df_temp.drop(columns=['TrialID'])
        df_target_temp = df_target_temp['Target']

        mutual_info_values = mutual_info_classif(df_temp, df_target_temp)

        unique_features += df_temp.columns.values.tolist()

        for feature, mutual_info_value in zip(df_temp.columns, mutual_info_values):
            if feature not in max_mutual_info_dict or max_mutual_info_dict[feature]['mutual_info'] < mutual_info_value:
                max_mutual_info_dict[feature] = {
                    'mutual_info': mutual_info_value,
                    'score_threshold': threshold
                }
            
            elif max_mutual_info_dict[feature]['mutual_info'] == mutual_info_value and max_mutual_info_dict[feature]['score_threshold'] < threshold:
                max_mutual_info_dict[feature] = {
                    'mutual_info': mutual_info_value,
                    'score_threshold': threshold
                }

    unique_features = list(set(unique_features))
    corresponding_values = [max_mutual_info_dict[feature]['score_threshold'] for feature in unique_features]

    max_values = [max_mutual_info_dict[feature]['mutual_info'] for feature in unique_features]
    exp = np.exp(np.array(max_values)/(np.array(corresponding_values)+1))
    weights = exp / np.sum(exp)

    weighted_average_threshold = np.average(corresponding_values, weights=weights)

    return weighted_average_threshold



def calculate_gradient(data, window_size):
    gradients = np.gradient(data)
    smoothed_gradients = np.convolve(gradients, np.ones(window_size)/window_size, mode='valid')
    return smoothed_gradients

def find_gradient(data, threshold=0.1, window_size=3):
    gradients = calculate_gradient(data, window_size)
    total_change = data[-1] - data[0]
    threshold_value = threshold * total_change
    for i, gradient in enumerate(gradients):
        if gradient < threshold_value:
            return i + window_size // 2 
    return i

def run_score_threshold_gradient(df, df_target, scores):
    score_thresholds = create_threshold_list(0, max(scores))

    unique_features = []
    all_scores_xls = []
    for threshold in score_thresholds:
        mask = np.where(scores <= threshold)[0]
        df_temp = df.iloc[mask]
        df_target_temp = df_target.iloc[mask]

        indexes = df_temp[df_temp.isna().any(axis=1)].index
        df_temp = df_temp.drop(index=indexes)
        df_target_temp = df_target_temp.drop(index=indexes)

        df_temp = extract.remove_nans(df_temp)        

        df_temp = df_temp.drop(columns=['TrialID'])
        df_target_temp = df_target_temp['Target']

        mutual_info_values = mutual_info_classif(df_temp, df_target_temp)

        unique_features += df_temp.columns.values.tolist()

        for feature, mutual_info_value in zip(df_temp.columns.values, mutual_info_values):
            all_scores_xls.append({'feature': feature, 'mutual_info': mutual_info_value, 'threshold': threshold})
            
    d = pd.DataFrame(all_scores_xls)
    d = d.pivot(index='threshold', columns='feature', values='mutual_info')
    d.reset_index(inplace=True)

    feature_thresholds = {}
    for feature in d.columns.values:
        if feature != 'threshold':
            index = find_gradient(d[feature].values)
            feature_thresholds[feature] = {'threshold': d['threshold'].iloc[index], 'mutual_info': d[feature].iloc[index]}

    unique_features = list(set(unique_features))
    corresponding_values = [feature_thresholds[feature]['threshold'] for feature in unique_features]

    max_values = [feature_thresholds[feature]['mutual_info'] for feature in unique_features]
    exp = np.exp(max_values)
    weights = exp / np.sum(exp)

    weighted_average_threshold = np.average(corresponding_values, weights=weights)

    return weighted_average_threshold



def run(window_size, overlap, train_trials, index):
    df = joblib.load(files_path + f'df_{window_size}_{overlap}.dat')
    df_target = joblib.load(files_path + f'df_target_{window_size}_{overlap}.dat')
    scores = joblib.load(files_path + f'scores_{window_size}_{overlap}.dat')

    df_target.loc[df_target['TrialID'].isin([4,5,6,7,8]), 'Target'] = 0
    df_target.loc[df_target['TrialID'].isin([9,10,11,12]), 'Target'] = 1
    
    print(f' - fold {index} -')
    mask = df_target['TrialID'].isin(trials)
    df = df[mask]
    df_target = df_target[mask]
    scores = scores[mask]

    mask = df_target['TrialID'].isin(train_trials)
    x_train = df[mask]
    y_train = df_target[mask]
    x_test = df[~mask]
    y_test = df_target[~mask]
    scores_train = scores[mask]
    scores_test = scores[~mask]

    print('starting score thresholds')
    if threshold_type == 'mutual_info':
        score_threshold = run_score_threshold_mutual_info(x_train, y_train, scores_train)
    elif threshold_type == 'gradient':
        score_threshold = run_score_threshold_gradient(x_train, y_train, scores_train)
    else:
        raise ValueError('threshold_type invalid')
    mask = np.where(scores_train <= score_threshold)[0]
    x_train = x_train.iloc[mask]
    y_train = y_train.iloc[mask]
    mask = np.where(scores_test <= score_threshold)[0]
    x_test = x_test.iloc[mask]
    y_test = y_test.iloc[mask]

    remove_indexes = x_train[x_train.isna().any(axis=1)].index
    x_train = x_train.drop(index=remove_indexes)
    y_train = y_train.drop(index=remove_indexes)

    remove_indexes = x_test[x_test.isna().any(axis=1)].index
    x_test = x_test.drop(index=remove_indexes)
    y_test = y_test.drop(index=remove_indexes)

    print('starting feature selection')
    columns = feature_selection(x_train, y_train)
    x_train = x_train[columns]
    x_test = x_test[columns]

    print('standardise')
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    y_test = y_test.astype(int)
    y_train = y_train.astype(int)

    print('smote')
    sm = SMOTE(
        random_state=0
    )
    train_os, train_targets_os = sm.fit_resample(x_train, y_train['Target'])

    print('dump files')
    joblib.dump(x_train, data_path + f'xtrain_{window_size}_{overlap}_fold{index}.dat')
    joblib.dump(y_train, data_path + f'ytrain_{window_size}_{overlap}_fold{index}.dat')
    joblib.dump(train_os, data_path + f'xtrainos_{window_size}_{overlap}_fold{index}.dat')
    joblib.dump(train_targets_os, data_path + f'ytrainos_{window_size}_{overlap}_fold{index}.dat')
    joblib.dump(x_test, data_path + f'xtest_{window_size}_{overlap}_fold{index}.dat')
    joblib.dump(y_test, data_path + f'ytest_{window_size}_{overlap}_fold{index}.dat')

    jobs = {'window size': window_size, 'window overlap': overlap, 'fold id': index, 'basename': f'{window_size}_{overlap}_fold{index}.dat'}
    return jobs


if __name__ == '__main__':
    window_params = [(x, y) for x, y in itertools.product(window_sizes, window_overlap) if y < x]
    params = []
    for w_param in window_params:
        for index, fold in enumerate(folds):
            params.append((w_param[0], w_param[1], fold, index))

    with multiprocessing.Pool() as pool:
        jobs = pool.starmap(run, params)
