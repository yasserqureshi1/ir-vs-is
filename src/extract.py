import numpy as np
import mosquitotracks.IO as IO
import mosquitotracks.preprocessing as pp
import os
import features

from scipy import stats as st
from scipy import signal
import pandas as pd


def load(files, path, is_resistant, data_path, fillgaps=True):
    tracks,images,istwelve,filelist = IO.loadfiles(files, data_path)

    if os.path.isfile(path + 'results/preprocess/feetPosall.csv') == False:
        pp.identifyNoseFeet(images,filelist, fileSuffix='all', dirpath=path+'results/preprocess/')

    nosePos,feetPos,npFiles = pp.loadNoseFeet('all', dirpath=path+ 'results/preprocess/')
    nosePos,feetPos = pp.unifyNoseFeet(nosePos,feetPos,npFiles,filelist)

    trials = np.arange(0,len(tracks))

    isResistant = np.array(is_resistant)

    TrialRefs = [i for i in range(len(tracks))]

    trainsTrials = pp.prepocessDataset(
        trials,TrialRefs,tracks,
        istwelve,filelist,images[0].shape[1],
        schema='central',nosePos=nosePos,feetPos=feetPos,
        doFillGaps=fillgaps,generateDynamics=True,
        doAbsRA=True,doAbsXY=False,
        minTrackTime=1,fps=50)

    selectedColumns = (1,2,6,11,13,9,10,3,7,8,12,14,17,18,19,20,5)
    tracks,tracksTargets,tracksTrialId = pp.trialsToTracks(trainsTrials,TrialRefs,isResistant,selectedColumns)

    print('Files loaded.')
    return tracks, tracksTargets, tracksTrialId


def generate_features(tracks, position_indexes, timestamp_indexes):
    print('STARTING GENERATE FEATURES...')
    track_id = 0
    while track_id < len(tracks):
        # Jerk 
        jerk = features.jerk(tracks[track_id], position_indexes, timestamp_indexes)
        tracks[track_id] = np.insert(tracks[track_id], len(tracks[track_id][0]), np.append([np.nan,np.nan,np.nan,np.nan], jerk), axis=1)

        # Direction of Flight Change
        dof = features.direction_of_flight_change(tracks[track_id], position_indexes)
        tracks[track_id] = np.insert(tracks[track_id], len(tracks[track_id][0]), np.append([np.nan,np.nan], dof), axis=1)

        # Centroid Distance Function 
        centroid_distance_function = features.centroid_distance_function(tracks[track_id], position_indexes)
        tracks[track_id] = np.insert(tracks[track_id], len(tracks[track_id][0]), centroid_distance_function, axis=1)

        # Peristance Velocity 
        pv, tv = features.orthogonal_components(tracks[track_id], position_indexes, timestamp_indexes)
        tracks[track_id] = np.insert(tracks[track_id], len(tracks[track_id][0]), np.append([np.nan, np.nan, np.nan], pv), axis=1)
        tracks[track_id] = np.insert(tracks[track_id], len(tracks[track_id][0]), np.append([np.nan, np.nan, np.nan], tv), axis=1)

        track_id += 1
    return tracks


def track_stats(track, indexes, columns):
    stats = dict()
    for index, col in enumerate(columns):
        elements = track[:, indexes[index]] 

        mask = track[:, -1].astype(bool)
        elements = elements[mask]
        element = elements[~np.isnan(elements)]

        try:
            stats[col + ' (mean)'] = np.mean(element)
        except:
            stats[col + ' (mean)'] = np.nan

        try:
            stats[col + ' (median)'] = np.median(element)
        except:
            stats[col + ' (median)'] = np.nan

        try:
            stats[col + ' (std)'] = np.std(element)
        except:
            stats[col + ' (std)'] = np.nan
        
        try:
            stats[col + ' (1st quartile)'] = np.percentile(element, 25)
        except:
            stats[col + ' (1st quartile)'] = np.nan

        try:
            stats[col + ' (3rd quartile)'] = np.percentile(element, 75)
        except:
            stats[col + ' (3rd quartile)'] = np.nan
        
        try:
            stats[col + ' (kurtosis)'] = st.kurtosis(element)
        except:
            stats[col + ' (kurtosis)'] = np.nan
        
        try:
            stats[col + ' (skewness)'] = st.skew(element)
        except:
            stats[col + ' (skewness)'] = np.nan
        
        try:
            stats[col + ' (number of local minima)'] = signal.argrelextrema(element, np.less)[0].shape[0]
        except:
            stats[col + ' (number of local minima)'] = np.nan

        try:
            stats[col + ' (number of local maxima)'] = signal.argrelextrema(element, np.greater)[0].shape[0]
        except:
            stats[col + ' (number of local maxima)'] = np.nan

        try:
            stats[col + ' (number of zero-crossings)'] = len(np.where(np.diff(np.sign(element)))[0])
        except:
            stats[col + ' (number of zero-crossings)'] = np.nan

    return stats


def add_other_features(data, pos):
    def feature_stats(element):
        element = element[~np.isnan(element)]
        return (
            np.mean(element),
            np.median(element),
            np.std(element),
            np.percentile(element, 25),
            np.percentile(element, 75),
            st.kurtosis(element),
            st.skew(element),
            signal.argrelextrema(element, np.less)[0].shape[0],
            signal.argrelextrema(element, np.greater)[0].shape[0],
            len(np.where(np.diff(np.sign(element)))[0])
        )

    other_features = {
        'Tortuosity': [],
        'Convex hull (area)': [],
        'Convex hull (perimeter)': [],
        'Curvature Scale Space (mean)': [],
        'Curvature Scale Space (median)': [],
        'Curvature Scale Space (std)': [],
        'Curvature Scale Space (1st quartile)': [],
        'Curvature Scale Space (3rd quartile)': [],
        'Curvature Scale Space (kurtosis)': [],
        'Curvature Scale Space (skewness)': [],
        'Curvature Scale Space (number of local minima)': [],
        'Curvature Scale Space (number of local maxima)': [],
        'Curvature Scale Space (number of zero-crossings)': [],
        'Fractal dimension': [],
        'Curvature (mean)': [],
        'Curvature (median)': [],
        'Curvature (std)': [],
        'Curvature (1st quartile)': [],
        'Curvature (3rd quartile)': [],
        'Curvature (kurtosis)': [],
        'Curvature (skewness)': [],
        'Curvature (number of local minima)': [],
        'Curvature (number of local maxima)': [],
        'Curvature (number of zero-crossings)': [],
    }

    track_id = 0
    while track_id < len(data):
        other_features['Tortuosity'].append(features.straightness(data[track_id], pos))
        try:
            other_features['Convex hull (area)'].append(features.convex_hull_area(data[track_id], pos))
            other_features['Convex hull (perimeter)'].append(features.convex_hull_perimeter(data[track_id], pos))
        except:
            other_features['Convex hull (area)'].append(np.nan)
            other_features['Convex hull (perimeter)'].append(np.nan)

        other_features['Fractal dimension'].append(features.fractal_dimension(data[track_id], pos))

        (
            css_mean, 
            css_median,
            css_std,
            css_1q,
            css_3q,
            css_kurtosis,
            css_skew,
            css_num_minima,
            css_num_maxima,
            css_num_zero
        ) = feature_stats(features.curvature_scale_space(data[track_id], pos))

        other_features['Curvature Scale Space (mean)'].append(css_mean)
        other_features['Curvature Scale Space (median)'].append(css_median)
        other_features['Curvature Scale Space (std)'].append(css_std)
        other_features['Curvature Scale Space (1st quartile)'].append(css_1q)
        other_features['Curvature Scale Space (3rd quartile)'].append(css_3q)
        other_features['Curvature Scale Space (kurtosis)'].append(css_kurtosis)
        other_features['Curvature Scale Space (skewness)'].append(css_skew)
        other_features['Curvature Scale Space (number of local minima)'].append(css_num_minima)
        other_features['Curvature Scale Space (number of local maxima)'].append(css_num_maxima)
        other_features['Curvature Scale Space (number of zero-crossings)'].append(css_num_zero)

        (
            c_mean, 
            c_median,
            c_std,
            c_1q,
            c_3q,
            c_kurtosis,
            c_skew,
            c_num_minima,
            c_num_maxima,
            c_num_zero
        ) = feature_stats(features.curvature(data[track_id], pos, timestamp_index=2))

        other_features['Curvature (mean)'].append(c_mean)
        other_features['Curvature (median)'].append(c_median)
        other_features['Curvature (std)'].append(c_std)
        other_features['Curvature (1st quartile)'].append(c_1q)
        other_features['Curvature (3rd quartile)'].append(c_3q)
        other_features['Curvature (kurtosis)'].append(c_kurtosis)
        other_features['Curvature (skewness)'].append(c_skew)
        other_features['Curvature (number of local minima)'].append(c_num_minima)
        other_features['Curvature (number of local maxima)'].append(c_num_maxima)
        other_features['Curvature (number of zero-crossings)'].append(c_num_zero)
        track_id += 1
    
    feat_df = pd.DataFrame(data=other_features)
    return feat_df


def remove_nans(df, ind=False):
    columns_to_drop = df.columns.to_series()[np.isinf(df).any()]
    for column in columns_to_drop:
        df = df.drop(columns=str(column))

    columns_to_drop = df.columns.to_series()[np.isnan(df).any()]
    for column in columns_to_drop:
        df = df.drop(columns=str(column))
    
    if ind == True:
        indexes = df[df.isna().any(axis=1)].index
        df = df.drop(index=indexes)
    return df
