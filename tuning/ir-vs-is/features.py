import numpy as np
from scipy.spatial import ConvexHull
from css.css import CurvatureScaleSpace


def check_track_parameter(track):
    if type(track) == list:
        return np.asarray(track)
    
    elif type(track) == np.ndarray:
        return track

    else:
        raise AttributeError(
            '`track` is of an unsuitable datatype. Please ensure it is '
            'either a list or NumPy array.'
            )


def direction_of_flight_change(track, position_indexes_in_track):
    """
    Calculates the change in the direction of flight between consectutive
    positions using a vector method.

    Parameters
    ----------
    track : NumPy array
        2-dimensional array containing track coordinates
    position_indexes_in_track : tuple
        List of the indexes within track array of the x and y (and z) 
        co-ordinates (e.g. (0,1,2))

    Returns
    -------
    NumPy array
        An array of the angle change between positions
    """
    track = check_track_parameter(track)
  
    if len(position_indexes_in_track) == 3:
        x = track[:, position_indexes_in_track[0]]
        y = track[:, position_indexes_in_track[1]]
        z = track[:, position_indexes_in_track[2]]
        v = np.column_stack((x[1:]-x[:-1], y[1:]-y[:-1], z[1:]-z[:-1]))
    elif len(position_indexes_in_track) == 2:
        x = track[:, position_indexes_in_track[0]]
        y = track[:, position_indexes_in_track[1]]
        v = np.column_stack((x[1:]-x[:-1], y[1:]-y[:-1]))

    change_in_angles = np.arccos(
        np.sum(v[1:,:]*v[:-1,:],axis=1) / np.sqrt( 
            np.sum(v[1:,:]*v[1:,:],axis=1) * np.sum(v[:-1,:]*v[:-1,:],axis=1) 
            ) 
        )
        
    change_in_angles[np.isnan(change_in_angles)] = 0

    return change_in_angles


def velocity(
    track, 
    position_indexes_in_track, 
    timestamp_index=None,
    ):
    """
    Calculates the velocity between positions of a given track.

    Parameters
    ----------
    track : NumPy array
        2-dimensional array containing track coordinates
    position_indexes_in_track : tuple
        List of the indexes within track array of the x, y and z)
        co-ordinates (e.g. (0,1,2))
    time_step : float
        Time between each position
    schema : string
        Method to calculate angles either using `forward` or `central`
        differences (default is 'forward')

    Returns
    -------
    NumPy array
        An array of the velocities between positions
    """ 
    track = check_track_parameter(track)
    
    x = track[:, position_indexes_in_track[0]]
    y = track[:, position_indexes_in_track[1]]
    t = track[:, timestamp_index]
    xs = x[2:] - x[:-2]
    ys = y[2:] - y[:-2]
    ts = t[2:] - t[:-2]
    _sum = np.power(xs, 2) + np.power(ys, 2) 
    return np.sqrt(_sum.astype(float))/ts


def acceleration(
    track, 
    position_indexes_in_track, 
    timestamp_index=None
    ):
    """
    Calculates the acceleration between positions of a given 
    track.

    Parameters
    ----------
    track : NumPy array
        2-dimensional array containing track coordinates
    position_indexes_in_track : tuple
        List of the indexes within track array of the x, y and z)
        co-ordinates (e.g. (0,1,2))
    time_step : float
        Time between each position
    schema : string
        Method to calculate angles either using `forward` or `central`
        differences (default is 'forward')

    Returns
    -------
    NumPy array
        An array of the acceleration between positions
    """
    vel = velocity(
        track=track, 
        position_indexes_in_track=position_indexes_in_track, 
        timestamp_index=timestamp_index
    )
    t = track[:, timestamp_index]
    ts = t[2:] - t[:-2]
    return np.diff(vel)/ts[1:]


def jerk(
    track, 
    position_indexes_in_track, 
    timestamp_index=None
    ):
    """
    Calculates the jerk between positions of a given track.

    Parameters
    ----------
    track : NumPy array
        2-dimensional array containing track coordinates
    position_indexes_in_track : tuple
        List of the indexes within track array of the x, y and z
        co-ordinates (e.g. (0,1,2))
    time_step : float
        Time between each position
    schema : string
        Method to calculate angles either using `forward` or `central`
        differences (default is 'forward')

    Returns
    -------
    NumPy array
        An array of the jerk between positions
    """
    acc = acceleration(
        track=track, 
        position_indexes_in_track=position_indexes_in_track,  
        timestamp_index=timestamp_index
    )
    
    t = track[:, timestamp_index]
    ts = t[2:] - t[:-2]
    return np.diff(acc)/ts[2:]


def angular_velocity(
    track, 
    position_indexes_in_track, 
    timestamp_index=None 
    ):
    """
    Calculates the instananeous angular velocity between positions of a given 
    track.

    Parameters
    ----------
    track : NumPy array
        2-dimensional array containing track coordinates
    position_indexes_in_track : tuple
        List of the indexes within track array of the x, y and z
        co-ordinates (e.g. (0,1,2))
    time_step : float
        Time between each position
    schema : string
        Method to calculate angles either using `forward` or `central`
        differences (default is 'forward')

    Returns
    -------
    NumPy array
        An array of the angular velocity between positions
    """

    track = check_track_parameter(track)
    t = track[:, timestamp_index]
    ts = t[2:] - t[:-2]
    a1_change = direction_of_flight_change(track, position_indexes_in_track)  
    return a1_change/ts


def angular_acceleration(
    track, 
    position_indexes_in_track, 
    timestamp_index=None
    ):
    """
    Calculates the instananeous angular acceleration between positions of a 
    given track.

    Parameters
    ----------
    track : NumPy array
        2-dimensional array containing track coordinates
    position_indexes_in_track : tuple
        List of the indexes within track array of the x, y and z
        co-ordinates (e.g. (0,1,2))
    time_step : float
        Time between each position
    schema : string
        Method to calculate angles either using `forward` or `central`
        differences (default is 'forward')

    Returns
    -------
    NumPy array
        An array of the angular acceleration between positions
    """
    track = check_track_parameter(track)

    angular_vel_1 = angular_velocity(
        track, 
        position_indexes_in_track, 
        timestamp_index
    )
    t = track[:, timestamp_index]
    ts = t[2:] - t[:-2]
    return np.diff(angular_vel_1)/ts
                

def axial_velocity(
    track, 
    position_indexes_in_track, 
    timestamp_index=None,
    ):
    """
    Calculates the axial velocity (i.e. the velocity in a given axis)

    Parameters
    ----------
    track : NumPy array
        2-dimensional array containing track coordinates
    position_indexes_in_track : tuple
        An index within track array of the x or y (or z) 
        co-ordinates (e.g. (0,1,2))
    time_step : float
        Time between each position
    schema : string
        Method to calculate angles either using `forward` or `central`
        differences (default is 'forward')

    Returns
    -------
    NumPy array
        Array of the velocity values for a given axis
    """
    track = check_track_parameter(track)

    v = track[:, position_indexes_in_track]
    t = track[:, timestamp_index]

    vd = v[2:] - v[:-2]
    ts = t[2:] - t[:-2]

    return vd/ts


def axial_acceleration(
    track, 
    position_indexes_in_track, 
    timestamp_index=None
    ):
    """
    Calculates the axial acceleration (i.e. the acceleration in a given axis)

    Parameters
    ----------
    track : NumPy array
        2-dimensional array containing track coordinates
    position_indexes_in_track : int
        An index within track array of the x or y (or z) 
        co-ordinate
    time_step : float
        Time between each position
    schema : string
        Method to calculate angles either using `forward` or `central`
        differences (default is 'forward')

    Returns
    -------
    NumPy array
        Array of the acceleration values for a particular axis
    """
    dv = axial_velocity(
        track, 
        position_indexes_in_track, 
        timestamp_index
    )
    
    track = check_track_parameter(track) 
    t = track[:, timestamp_index]
    ts = t[2:] - t[:-2]

    return np.diff(dv)/ts


def __cartesian_to_circular(x, y):
    """
    Converts from the cartesian co-ordinates to the circular co-ordinate 
    system
    """
    rho = np.sqrt(np.power(x, 2) + np.power(y, 2))
    theta = np.arctan(y/x)
    return rho, theta


def __circular_to_cartesian(rho, theta):
    """
    Converts from the circular co-ordinates to the cartesian co-ordinate 
    system
    """
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def orthogonal_components(
    track, 
    position_indexes_in_track, 
    timestamp_index=None
    ):
    """
    Calculates the orthogonal components of velocity known as Persistence and 
    turning (and inclination in for 3D tracks) velocities.

    Persistence is the tendency for a movement to persist orthogonal to the 
    basis vector. Turing and inclination are the tencencies to head normal to 
    the trajectory and orthogonal to each other.

    Parameters
    ----------
    track : NumPy array
        2-dimensional array containing track coordinates
    position_indexes_in_track : tuple
        List of the indexes within track array of the x, y and z
        co-ordinates (e.g. (0,1,2))
    time_step : float
        Time between each position
    schema : string
        Method to calculate angles either using `forward` or `central`
        differences (default is 'forward')

    Returns
    -------
    tuple
        For 3-dimensional tracks (x,y,z), a tuple containing the arrays for 
        persistence, turning and inclination velocities

    References
    ----------
    .. [1] Edelhoff, H., Signer, J. & Balkenhol, N. Path segmentation for 
    beginners: an overview of current methods for detecting changes in animal 
    movement patterns. Mov Ecol 4, 21 (2016). 
    """
    track = check_track_parameter(track)

    vels = np.abs(velocity(
        track, 
        position_indexes_in_track, 
        timestamp_index
        ))

    x = track[:, position_indexes_in_track[0]]
    y = track[:, position_indexes_in_track[1]]

    _, theta = __cartesian_to_circular(
    x[2:]-x[:-2], y[2:]-y[:-2]
    )
    persistence, turning = __circular_to_cartesian(
        vels[1:], np.abs(theta[1:]-theta[:-1])
        )
    return persistence, turning
    

def total_distance_travelled(track, position_indexes_in_track):
    """
    Returns the total physical distance of a given track
    
    Parameters
    ----------
    track : NumPy array
        2-dimensional array containing track coordinates
    position_indexes_in_track : tuple
        List of the indexes within track array of the x, y and z
        co-ordinates (e.g. (0,1,2))

    Returns
    -------
    float
        Physical length of the track
    """
    track = check_track_parameter(track)

    x = track[:, position_indexes_in_track[0]]
    y = track[:, position_indexes_in_track[1]]
    x_sum = x[1:] - x[:-1]
    y_sum = y[1:] - y[:-1]
    _sum = np.power(x_sum, 2) + np.power(y_sum, 2)
    return np.sqrt(_sum.astype(float)).sum()


def end_to_end_distance(track, position_indexes_in_track):
    """
    Returns the end-to-end distance of a given track
    
    Parameters
    ----------
    track : NumPy array
        2-dimensional array containing track coordinates
    position_indexes_in_track : tuple
        List of the indexes within track array of the x, y and z
        co-ordinates (e.g. (0,1,2))

    Returns
    -------
    float
        Physical length of the track
    """
    track = check_track_parameter(track)
        
    x = track[:, position_indexes_in_track[0]]
    y = track[:, position_indexes_in_track[1]]
    x_change = x[-1] - x[0]
    y_change = y[-1] - y[0]
    _sum = (
        np.power(x_change, 2) + 
        np.power(y_change, 2)
        )
    return np.sqrt(_sum.astype(float))


def straightness(track, indexes_in_track):
    """
    Returns the ratio between the total distance travelled and the end-to-end 
    distance
    
    Parameters
    ----------
    track : NumPy array
        2-dimensional array containing track coordinates
    position_indexes_in_track : tuple
        List of the indexes within track array of the x, y and z
        co-ordinates (e.g. (0,1,2))

    Returns
    -------
    float
        Physical length of the track

    References
    ----------
    .. [1] Edelhoff, H., Signer, J. and Balkenhol, N., 2016. Path 
    segmentation for beginners: an overview of current methods for detecting 
    changes in animal movement patterns. Movement Ecology, 4(1).
    """
    total_d = total_distance_travelled(track, indexes_in_track)
    end_to_end = end_to_end_distance(track, indexes_in_track)
    return total_d/end_to_end


def curvature(
    track, 
    position_indexes_in_track, 
    timestamp_index=None
    ):
    """
    Calculates the curvature of each point in the track

    Parameters
    ----------
    track : NumPy array
        2-dimensional array containing track coordinates
    position_indexes_in_track : tuple
        List of the indexes within track array of the x, y and z
        co-ordinates (e.g. (0,1,2))
    time_step : float
        Time between each position

    Returns
    -------
    NumPy array or tuple of NumPy arrays
        For 3-dimensional tracks (x,y,z), a tuple containing three NumPy 
        arrays with curvature values for the projections in the xy, yz and zx 
        planes
    
    References
    ----------
    .. [1] Atanbori, J., Duan, W., Murray, J., Appiah, K. and Dickinson, P., 
    2016. Automatic classification of flying bird species using computer 
    vision techniques. Pattern Recognition Letters, 81, pp.53-62.
    """
    x = track[:, position_indexes_in_track[0]]
    y = track[:, position_indexes_in_track[1]]
    t = track[:, timestamp_index]
    ts = np.diff(t)
    x_change = x[1:] - x[:-1]
    y_change = y[1:] - y[:-1]
    x_d = x_change/ts
    x_dd = np.diff(x_d)/ts[1:]
    y_d = y_change/ts
    y_dd = np.diff(y_d)/ts[1:]
    
    k_xy = (x_d[1:]*y_dd - y_d[1:]*x_dd)/(
        np.power(np.power(x_d[1:], 2) + np.power(y_d[1:], 2), 3/2)
        )
    return k_xy


def convex_hull_area(track, position_indexes_in_track):
    """
    Calculates the area (or volume for 3D tracks) of a given track

    Parameters
    ----------
    track : NumPy array
        2-dimensional array containing track coordinates
    position_indexes_in_track : tuple
        List of the indexes within track array of the x, y and z
        co-ordinates (e.g. (0,1,2))

    Returns
    -------
    float
        Value for the area or volume of the convex hull

    References
    ----------
    .. [1] Rintoul, M. and Wilson, A., 2015. Trajectory analysis via a 
    geometric feature space approach. Statistical Analysis and Data Mining: 
    The ASA Data Science Journal, 8(5-6), pp.287-301.
    """
    ch = ConvexHull(points=track[:, position_indexes_in_track])
    return ch.volume


def convex_hull_perimeter(track, position_indexes_in_track):
    """
    Calculates the perimeter (or surface area for 3D tracks) of a given track

    Parameters
    ----------
    track : NumPy array
        2-dimensional array containing track coordinates
    position_indexes_in_track : tuple
        List of the indexes within track array of the x, y and z
        co-ordinates (e.g. (0,1,2))

    Returns
    -------
    float
        Value for the perimeter or surface area of the convex hull

    References
    ----------
    .. [1] Rintoul, M. and Wilson, A., 2015. Trajectory analysis via a 
    geometric feature space approach. Statistical Analysis and Data Mining: 
    The ASA Data Science Journal, 8(5-6), pp.287-301.
    """
    ch = ConvexHull(points=track[:, position_indexes_in_track])
    return ch.area


def centroid_distance_function(track, position_indexes_in_track):
    """
    Returns the distance between each point on the track to the centre point 
    of the track

    Parameters
    ----------
    track : NumPy array
        2-dimensional array containing track coordinates
    position_indexes_in_track : tuple
        List of the indexes within track array of the x, y and z
        co-ordinates (e.g. (0,1,2))

    Returns
    -------
    NumPy array
        Array of the distances between each point on the track to the centre 
        point of the track

    References
    ----------
    .. [1] Atanbori, J., Duan, W., Murray, J., Appiah, K. and Dickinson, P., 
    2016. Automatic classification of flying bird species using computer 
    vision techniques. Pattern Recognition Letters, 81, pp.53-62.
    """
    track = check_track_parameter(track)
    x = track[:, position_indexes_in_track[0]]
    y = track[:, position_indexes_in_track[1]]
    xc = x.sum()/len(x)
    yc = y.sum()/len(y)
    _sum = np.power(x - xc, 2) + np.power(y - yc, 2)
    return np.sqrt(_sum.astype(float))


def curvature_scale_space(track, position_indexes_in_track):
    """
    Returns a 1-dimensional representation of the CSS image

    Codebase adapted from reference [1]. GitHub repository can be found 
    at https://github.com/makokal/pycss

    Parameters
    ----------
    track : NumPy array
        2-dimensional array containing track coordinates
    position_indexes_in_track : tuple
        List of the indexes within track array of the x, y and z
        co-ordinates (e.g. (0,1,2))

    Returns
    -------
    NumPy array
        Array of the distances between each point on the track to the centre 
        point of the track

    References
    ----------
    .. [1] Okal, B. and Nuchter, A., 2013. Sliced curvature scale space for 
    representing and recognizing 3D objects. 2013 16th International 
    Conference on Advanced Robotics (ICAR),.

    .. [2] Mokhtarian, F. and Mackworth, A., 1986. Scale-Based Description 
    and Recognition of Planar Curves and Two-Dimensional Shapes. IEEE 
    Transactions on Pattern Analysis and Machine Intelligence, PAMI-8(1), 
    pp.34-43.
    """
    track = check_track_parameter(track)

    x = track[:, position_indexes_in_track[0]]
    y = track[:, position_indexes_in_track[1]]
    curve = np.array([x.reshape(len(x)), y.reshape(len(y))])

    c = CurvatureScaleSpace()
    cs = c.generate_css(curve, curve.shape[1], 1)
    return c.generate_visual_css(cs, 9)


def fractal_dimension(
    track, 
    position_indexes_in_track, 
    element_length_index=None, 
    element_accuracy=2
    ):
    """
    Computes the fractal dimension of a particular track.

    The fractal dimension is a value that describes whether a track is 
    tending towards more planar or linear behaviour. 

    Parameters
    ----------
    track : array
        2-dimensional array containing track coordinates
    position_indexes_in_track : tuple
        List of the indexes within track array of the x and y (and z) 
        co-ordinates (e.g. (0,1,2))
    element_length_index : int , optional
        Index within track array for the length between positions. If None, 
        then it will calculate this (default is None)
    element_accuracy : int , optional
        Accuracy of the calculation for the lengths of each element

    Returns
    -------
    int
        a fractal dimension value for the track

    References
    ----------
    .. [1] Mandelbrot, B., 1967. How Long Is the Coast of Britain? 
    Statistical Self-Similarity and Fractional Dimension. Science, 156(3775), 
    pp.636-638.

    .. [2] Turchin, P., 1996. Fractal Analyses of Animal Movement: A 
    Critique. Ecology, 77(7), pp.2086-2090.
    
    """
    track = check_track_parameter(track)
    
    x = track[:, position_indexes_in_track[0]]
    y = track[:, position_indexes_in_track[1]]

    ref_dist = np.sqrt(
        np.power(np.ptp(x), 2) + 
        np.power(np.ptp(y), 2)
        )
    if element_length_index is None:
        element_lengths = np.sqrt(
            np.power(x[1:] - x[:-1], 2) + 
            np.power(y[1:] - y[:-1], 2)
        )

    else:
        element_lengths = track[:, element_length_index]
    
    adjustment = np.power(10, element_accuracy)

    ref_dist = np.round(ref_dist*adjustment).astype(int)
    element_lengths = np.round(element_lengths*adjustment).astype(int)

    N = np.sum(element_lengths)
    epsilon = 1/ref_dist

    return -1*np.log(N)/np.log(epsilon)
