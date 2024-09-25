import numpy as np

def get_time(track):
    track = np.array(track)
    return track[-1, 2] - track[0, 2]

def split_tracks(tracks, target, trial, size, over_lap):
    def window(track, size, over_lap):
        segments = []
        cumulative_time = 0
        position_id = 1
        start = 0 

        while position_id < len(track):
            cumulative_time = get_time(track[start:position_id])
            if (cumulative_time >= size) and (cumulative_time < size*2):

                segments.append(track[start:position_id])
                overlap_index = position_id - 1
                cumulative_overlap = 0
                while cumulative_overlap < over_lap:
                    if overlap_index == 1:
                        break
                    cumulative_overlap = get_time(track[overlap_index:position_id])
                    overlap_index -= 1
                start = overlap_index
                cumulative_time = 0
            elif (cumulative_time > size*2):
                start = position_id
                
            position_id += 1
        return segments

    split_track = []
    split_target = []
    split_trial = []
    split_group = []
    track_id = 0
    while track_id < len(tracks):
        track_time = get_time(tracks[track_id])
        if track_time >= size:
            tk = window(tracks[track_id], size, over_lap)
            if tk == []:
                pass
            else:
                split_track += tk
                split_target += [target[track_id] for _ in range(len(tk))]
                split_trial += [trial[track_id] for _ in range(len(tk))]
                split_group += [track_id for _ in range(len(tk))]
        track_id += 1
    return np.array(split_track, dtype=object), np.array(split_target), np.array(split_trial), np.array(split_group)
