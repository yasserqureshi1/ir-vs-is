import numpy as np
import features

from scipy import stats as st
from scipy import signal
import pandas as pd


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
    def stats(features):
        return np.mean(features), np.std(features) 

    other_features = {
        'Straightness': [],
        'Convex hull (area)': [],
        'Convex hull (perimeter)': [],
        'Curvature Scale Space (mean)': [],
        'Curvature Scale Space (std)': [],
        'Fractal dimension': [],
        'Curvature (mean)': [],
        'Curvature (std)': []
    }

    track_id = 0
    while track_id < len(data):
        other_features['Straightness'].append(features.straightness(data[track_id], pos))
        try:
            other_features['Convex hull (area)'].append(features.convex_hull_area(data[track_id], pos))
            other_features['Convex hull (perimeter)'].append(features.convex_hull_perimeter(data[track_id], pos))
        except:
            other_features['Convex hull (area)'].append(np.nan)
            other_features['Convex hull (perimeter)'].append(np.nan)

        other_features['Fractal dimension'].append(features.fractal_dimension(data[track_id], pos))
        css_mean, css_std = stats(features.curvature_scale_space(data[track_id], pos))
        other_features['Curvature Scale Space (mean)'].append(css_mean)
        other_features['Curvature Scale Space (std)'].append(css_std)
        c1_mean, c1_std = stats(features.curvature(data[track_id], pos, timestamp_index=2))
        other_features['Curvature (mean)'].append(c1_mean)
        other_features['Curvature (std)'].append(c1_std)
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
