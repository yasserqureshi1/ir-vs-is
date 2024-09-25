import pandas as pd
import numpy as np

import itertools
import time
import random

import extract, split
import multiprocessing
import joblib

np.random.seed(0)
random.seed(0)

# Path of original tracks
path = '/home/eng/esrdfn/tuning/banfora-vs-vk7/'

# Path to store dataframes
df_path = '/home/eng/esrdfn/tuning/banfora-vs-vk7/dfs/'

# ---------------------- WINDOW PARAMS ----------------------
window_sizes = [1,2,3,4,5,6,7,8]
window_overlap = [0.5,1,1.5,2,2.5,3,4]


hyp_trials = np.array([0,1, 13,14])
banfora = np.array([0, 1])
vk7 = np.array([13, 14])


def segmentation(tracks, trackTargets, tracksTrialId, window_size, window_overlap):
    tracks, trackTargets, tracksTrialId, trackGroup = split.split_tracks(tracks, trackTargets, tracksTrialId, window_size, window_overlap)

    df, df_target = feature_extraction(tracks, trackTargets, tracksTrialId, trackGroup)

    scores = []
    for segment in tracks:
        mask = segment[:, -1]
        scores.append(penalty_function(mask, n=1, m=1.05))
    scores = np.array(scores)

    return df, df_target, scores


def feature_extraction(tracks, trackTargets, tracksTrialId, trackGroup):
    '''
    Extracts features of flight from tracks
    '''

    feature_columns = [
        'X Velocity',
        'Y Velocity',
        'X Acceleration', 
        'Y Acceleration',
        'Velocity',
        'Acceleration',
        'Jerk',
        'Angular Velocity',
        'Angular Acceleration',
        'Angle of Flight',
        'Centroid Distance Function',
        'Persistence Velocity',
        'Turning Velocity'
    ]   
    indexes = [12,13,14,15,3,10,17,4,11,18,19,20,21]
    feature_stats = [
        'mean',
        'median',
        'std', 
        '1st quartile',
        '3rd quartile',
        'kurtosis', 
        'skewness',
        'number of local minima',
        'number of local maxima',
        'number of zero-crossings'
    ]     

    track_statistics = dict()

    for col in feature_columns:
        for stat in feature_stats:
            track_statistics[f'{col} ({stat})'] = []

    for track in tracks:
        data = extract.track_stats(track, indexes=indexes, columns=feature_columns)
        for d in data:
            track_statistics[d].append(data[d])

    df = pd.DataFrame(data=track_statistics)
    to_add = extract.add_other_features(tracks, (0,1))
    df = pd.concat([df, to_add], axis=1)
    df = df.join(pd.DataFrame({'TrialID': tracksTrialId}))
    df_target = pd.DataFrame({'Target': trackTargets, 'TrialID': tracksTrialId, 'TrackGroup': trackGroup})
    return df, df_target


def prep_folds(path, window_size, overlap, tracks, trackTargets, tracksTrialId):
    # 5 folds each in a process
    print(f'window size: {window_size} ; window overlap: {overlap}')
    
    df, df_target, scores = segmentation(
        tracks, trackTargets, tracksTrialId, window_size, overlap
    )
    df_target.loc[df_target['TrialID'].isin(banfora), 'Target'] = 1
    df_target.loc[df_target['TrialID'].isin(vk7), 'Target'] = 0

    joblib.dump(df, path + f'df_{window_size}_{overlap}.dat')
    joblib.dump(df_target, path + f'df_target_{window_size}_{overlap}.dat')
    joblib.dump(scores, path + f'scores_{window_size}_{overlap}.dat')


def penalty_function(segment, n, m):
    penalty_score = 0
    k = 0

    for position in segment:
        if position == 0:
            penalty_score += n * (m ** k)
            k += 1
        else:
            k = max(0, k-1)

    return penalty_score/len(segment)



if __name__ == '__main__':
    # ---------------------- LOAD/GENERATE DATA ----------------------
    tracks = np.load(path + 'tracks_features_gaps_marked.npy', allow_pickle=True)
    trackTargets = np.load(path + 'raw_trackTargets.npy', allow_pickle=True)
    tracksTrialId = np.load(path + 'raw_tracksTrialId.npy', allow_pickle=True)


    print('ORIGINAL: ', len(tracks))
    mask = np.isin(tracksTrialId, hyp_trials)
    tracks = tracks[mask]
    trackTargets = trackTargets[mask]
    tracksTrialId = tracksTrialId[mask]
    print('MASKED: ', len(tracks))

    mask = np.isin(tracksTrialId, vk7)
    trackTargets[mask] = 0
    trackTargets[~mask] = 1

    # Initalise time
    t0 = time.time()

    with multiprocessing.Pool() as pool:
        window_params = [(x, y) for x, y in itertools.product(window_sizes, window_overlap) if y < x]
        
        pool.starmap(prep_folds, [
                (df_path, window_size, overlap, tracks, trackTargets, tracksTrialId)
                for window_size, overlap in window_params
            ])
    print('TIME: ', time.time()-t0)
        