# coding: utf-8 
	
from based_noise_blinks_detection import *
import warnings
import pandas as pd
import numpy as np

def extract_events(data, event_name):
    if event_name == 'samps':
        col = ['time','trackertime', 'x','y','size']
        df = data[col].explode(col)
    else:
        if event_name == 'Efix':
            col = ['starttime', 'endtime', 'duration', 'endx', 'endy']
        elif event_name == 'Esac':
            col = ['starttime', 'endtime', 'duration', 'startx', 'starty', 'endx', 'endy']
        elif event_name == 'Eblk':
            col = ['starttime', 'endtime', 'duration']
        elif event_name == 'msg':
            col = ['trackertime', 'message']
        df = data.events.apply(lambda x: x[event_name]).explode().dropna()
        df = pd.DataFrame(df.tolist(), columns=col)

    if event_name != 'msg': # convert to numeric
        df = pd.DataFrame(df.values.astype('float'), columns=col)

    return df


def borrow_events(data, time_col, event_data, events_to_borrow):
    for i in event_data.index:
        row = event_data.loc[i]
        s = int(row.starttime)
        e = int(row.endtime)
        inrange = data[time_col].between(s,e)
        for e in events_to_borrow:
            data.loc[inrange, e] = row[e]
    return data

    
def correlation_eye(data, left, right):
    """calculate the correlation between two columns
    
    arguments: 
    data: a pandas dataframe
    left: string. The column name for the left eye
    right: string. The column name for the right eye
    
    returns:
    pearson correlation coefficient
    """
    corr = data[left].corr(data[right], method='pearson')
    return corr
	
	
def select_eye(data, use_eye):
    """select which eye to use. Data of the unselected eye will be removed.
    
    arguments:
    data: a pandas dataframe
    use_eye: 'left' or 'right'.
    
    returns:
    a pandas dataframe for the selected eye
    """
    
    if (use_eye == 'left') & ('pup_l' in data.columns):
        data[['x','y','pup']] = data[['x_l','y_l','pup_l']]
    elif (use_eye == 'right') & ('pup_r' in data.columns):
        data[['x','y','pup']] = data[['x_r','y_r','pup_r']]
	# when the eye requested is not available
    elif use_eye == 'left':
        data[['x','y','pup']] = data[['x_r','y_r','pup_r']]
        warnings.warn("Left eye not available. Right eye is used.")
    elif use_eye == 'right':
        data[['x','y','pup']] = data[['x_l','y_l','pup_l']]
        warnings.warn("Right eye not available. Left eye is used.")
    return data[['x','y','pup']]
	

"""
def get_trial_event_times(events, phase_events): 
    number_of_messages = len(phase_events)*2
    trial_event_times = events.MSG[events.MSG.label.isin(['start_trial','end_trial','start_phase','end_phase'])]
    trial_event_times['trial_id'] = pd.to_numeric(trial_event_times.content, errors='coerce').ffill(limit=number_of_messages)
    trial_event_times['event'] = trial_event_times.apply(lambda x: x.label + '_' + x.content if x.content in phase_events else x.label, axis=1)
    trial_event_times = trial_event_times.reset_index().pivot_table(index='trial_id',columns=['event'], values=['index'])
    trial_event_times.columns = trial_event_times.columns.droplevel(0)
    
    # sort columns by first row
    trial_event_times.sort_values(trial_event_times.first_valid_index(),axis=1, ascending=True, inplace=True)
    return trial_event_times
"""

def trial_events_from_cili(df, trial_events):
    """extract timestamps of trial events from cili output
    
    arguments:
    df: a pandas dataframe (events.MSG)
    trial_events: a list of messages to be extracted from the Event object
    
    returns:
    a dataframe with trial event timestamps
    """
    # select rows
    trial_event_times = df[df.label.isin(trial_events)].reset_index()
    # create a trial id column
    trial_event_times['trial_id'] = pd.to_numeric(trial_event_times.content, errors='raise')
    # reshape dataframe
    trial_event_times = trial_event_times.pivot_table(index='trial_id',columns='label', values='index')
    
    # sort columns
    trial_event_times = trial_event_times.reindex(columns = trial_events)
    
    return trial_event_times
	
	
def select_data(time_data, eye_data, start, end, align_on=None):
    
    # get timestamp of start and end message
    s = int(time_data[start].values[0])
    e = int(time_data[end].values[0])
    
    # assuming the index is the timestamp in the raw data file
    df = eye_data.loc[s:e]
    
    # align the timestamps to the start or the end of message
    if align_on:
        t = int(time_data[align_on])
        df['onset_aligned'] = df.index - t

    return df
	
def loss_ratio(df, column, missing_indicator=0):
    """compute missing value ratio
    
    arguments:
    df: a pandas dataframe
    column: the name of the column to be checked
    missing_indicator: optional. An integer used to indicate missing data. default is 0 (as in eyelink).
    
    returns:
    the ratio of missing data (0 to 1)
    """
    
    # if np.nan is passed
    if np.isnan(missing_indicator):
        ratio = float(len(df[df[column].isnull()]))/len(df)
    else:
        ratio = float(len(df[df[column]==missing_indicator]))/len(df)
        
    return ratio
	
	
def check_tracking_loss(df, column, threshold, missing_indicator=0):

    """compute tracking loss for each group

    arguments:
    df: a pandas dataframe
    column: the column to be checked
    threshold: threshold for data loss (e.g., .30). Groups with signal loss beyond this threhold will be marked as discarded
    missing_indicator: optional. the value used to indicate missing data. default is 0 (as in eyelink).
    
    return:
    a pandas dataframe with a new column marking traces should be removed
    """
    
    # compute trial level signal loss
    r = loss_ratio(df, column=column, missing_indicator=missing_indicator)
    df['signal_loss'] = r
    #print r
    if r > threshold:
        #print "Too much tracking loss at group", df.name
        df['low_tracking_ratio'] = 1
    else:
        df['low_tracking_ratio'] = 0
    
    return df
        
  
	
	
def check_signal_loss_2(df, trial_id, column, trial_threshold, subject_threshold, missing_indicator=0, remove=True):
    """compute missing value for each trial and remove those who passed beyond certain threshold.
    Further removing a subject if there were too many trials removed
    
    arguments:
    df: a pandas dataframe
    trial_id: the column name for trial ID.
    column: the column to be checked
    trial_threshold: trial level threshold for data loss (e.g., .30). Trials with signal loss beyond this threhold will be removed
    subject_threshold: subject level threshold for data loss (e.g., .30). Subjects will removed trials beyond this threshold will be removed.
    missing_indicator: optional. the value used to indicate missing data. default is 0 (as in eyelink).
    remove: optional. should I remove the trials/subjects? default is True. If False, those trials/subjects will not be removed. 
    Instead, a new column is created to mark if the data should be removed.
    
    return:
    a pandas dataframe
    """
    
    # compute trial level signal loss
    trial_level_ratio = df.groupby(trial_id).apply(loss_ratio, column=column, missing_indicator=missing_indicator)
    bad_trials = trial_level_ratio[trial_level_ratio > trial_threshold]
    
    print ("%d trial(s) had tracking loss beyond the specified threshold" % len(bad_trials))
    print (bad_trials)
        
    # compute subject level missing ratio
    subject_level_ratio = float(len(bad_trials))/len(trial_level_ratio)
    if subject_level_ratio > subject_threshold:
        print ('Too many trials missing for this subject')
        df['discard'] = 1 # all data from this subject should be removed
    else:
        df['discard'] = df[trial_id].apply(lambda x: 1 if x in bad_trials.index else 0) # only bad trials should be removed
        
    # remove bad trials
    if remove:
        df = df[df.discard==0].reset_index(drop=True)
        df = df.drop('discard', axis=1)
    return df


def deblink_pupil(df, column, samp_freq):
    """Detect blinks in pupil sizes and set pupil size around blinks as NaN. 
    Blink detection is based on Hershman et al. (2018).
    
    argument:
    df: a pandas dataframe. 
    column: the name of the pupil size column
    samp_freq: sampling frequency (integer)
	
    returns:
    a pandas dataframe with a new column of deblinked pupil 
    """
    
    # detecting blinks
    blinks = based_noise_blinks_detection(df[column].values, samp_freq)
    
    # set samps during blinks as NaN
    df[column + '_deblink'] = df[column]
    for i in zip(blinks['blink_onset'],blinks['blink_offset']):
        #df[column + '_deblink'].iloc[i[0]:i[1]+1] = np.nan # +1 used to include the last zero
        df.iloc[slice(i[0], i[1]+1), df.columns.get_loc(column + '_deblink')] = np.nan # +1 used to include the last zero
    # set other missing values to be NaN
    # zeroes are retained after deblink if the entire trial is missing
    df[column + '_deblink'] = df[column + '_deblink'].replace({0:np.nan})
    
    return df
	
	
def remove_outliers(df, column_pup, maxdev = 2.5, allowp=0.1, 
                    column_x = None, column_y = None, left = None, right = None, top = None, bottom = None):
    """remove outliers based on pupil location and pupil size
    
    arguments:
    df: a pandas dataframe
    column_pup: pupil size column
    maxdev: z-score threshold for setting outliers. Samples beyond this threshold will be set to NaN. Default is 2.5.
    allowp: If the standard deviation is below this proportion of the mean, outliers will not be removed; 
            this is to prevent erroneous removal of outliers in a very steady signal (default = 0.1)
            from https://github.com/esdalmaijer/PyGazeAnalyser/blob/master/pygazeanalyser/traces.py
    column_x & column_y : column name for the x & y coordinate. 
    screen_x & screen_y: the screen resolution, e.g.(1024, 768). 
    
    Note: If any of them (column x & y, screen x & y) is left as default (None), out-of-screen samples cleaning will not be performed.
           
    returns:
    the original dataframe plus a new column with outliers set to NaN
    """
    
    # off-screen samples
    ## check if proper argumnets are passed
    if None in [column_x, column_y, left, right, top, bottom]:
        warnings.warn("Screen information not properly specified. Out-of-screen samples will not be removed.")
        df[column_pup+'_rm'] = df[column_pup]
    ## remove out-of-screen samples
    else:
        conditions = ((df[column_x] < left) | (df[column_x] > right) | (df[column_y] < top) | (df[column_y] > bottom))
        df[column_pup+'_rm'] = np.where(conditions, np.nan, df[column_pup])
        
    # samples with a large SD
    mean = df[column_pup+'_rm'].mean(skipna=True)
    std = df[column_pup+'_rm'].std(skipna=True)
    
    # if std is reasonably small then no outlier will be declared
    if std >= allowp*mean:
        lower = mean - maxdev*std
        upper = mean + maxdev*std
        conditions2 = ((df[column_pup+'_rm']<lower) | (df[column_pup+'_rm']>upper))
        df[column_pup+'_rm'] = np.where(conditions2, np.nan, df[column_pup+'_rm'])
        
    return df
	



def smooth_pupil(df, column, method='hann', window=10):
    """
    perform rolling mean or hann smoothing on a column
    
    arguments:
    df: a pandas dataframe
    column: name of the pupil size column
    method: Optional. A string indicating the method of smoothing. "rollingmean" -- rolling mean, "hann" -- hann window. Default is 'hann'.
    window: Optional. An integer indicating the window size (in the number of samples) for smoothing. Default is 10.
    
    returns:
    a new pandas dataframe of smoothed pupil sizes
    """
    
    if len(df[column]) < window:
        raise Exception('Length of data smaller than window size.')
    
    if window < 3:
        warnings.warn("Window size is too small (< 3). Smoothing not performed.")
            
    if method not in ['rollingmean','hann']:
        raise Exception("Invalid smoothing method. Please use 'rollingmean' or 'hann'.")
        
    if method == 'rollingmean':
        df[column+'_smooth'] = df[column].rolling(window=window, win_type=None, min_periods = window).mean()
    if method == 'hann':
        df[column+'_smooth'] = df[column].rolling(window=window, win_type='hann', min_periods = window).mean()
        
    return df


def interpolate_pupil(df, column, method = 'linear'):
    """
    interpolate missing data. 
    
    arguments:
    df: a pandas dataframe
    column: name of the pupil size column
    method: Optional. default is 'linear'. Use 'spline' for a cubic-spline interpolation.
    
    returns:
    a new pandas dataframe with interpolated pupil sizes
    """
    if method not in ['spline','linear']:
        raise Exception("Invalid interpolation method. Please use 'linear' or 'spline'.")
        
    if method == 'linear':
        df[column+'_interp'] = df[column].interpolate(method=method)
    elif method == 'spline':
        df[column+'_interp'] = df[column].interpolate(method='spline',order=3)
        
    # handle start and end missing values
    df[column+'_interp'].bfill(inplace=True)
    df[column+'_interp'].ffill(inplace=True)
    
    return df
	
	
def downsample_pupil(df, pup_col, time_col, bin_size, method='median'):
    """
    downsampling data in selected columns based on time index
    
    arguments:
    df: a pandas dataframe.
    pup_col: the name of the column to be downsampled.
    time_col: the name of the timestamp column. Unit should be in milliseconds.
    bin_size: a string indicating the duration of each bin. E.g.'200ms'.
    method: optional. 'median' - get the median for each bin (default). 'mean' - use the mean.
    
    returns:
    pandas dataframe with newly resampled data
    """ 
    
    if method not in ['mean','median']:
        raise Exception("Invalid sampling method. Please use 'mean' or 'median'.")
    
    # convert the microsecond timestamp to datetime timestamp
    df[time_col] = pd.to_datetime(df[time_col], unit = 'ms')
    
    # resampling on the datetime timestamp
    df[pup_col+'_resamp'] = df[pup_col]
    resampler = df[[time_col] + [pup_col+'_resamp']].resample(bin_size, on=time_col, loffset='0ms',label='left')
    
    # decide which method to calculate results
    if method == 'median':
        resampled_samps = resampler.median()
    elif method == 'mean':
        resampled_samps = resampler.mean()
    
    # convert the datetime timestamp back to microsecond timestamp
    resampled_samps.index = (resampled_samps.index - pd.Timestamp("1970-01-01")) // pd.Timedelta('1ms')
    
    return resampled_samps
	

def baseline_correction(df, pup_col, time_col, baseline_range, method='subtractive'):
    """performs baseline correction on pupil data
    
    arguments:
    df: a pandas dataframe.
    pup_col: the name of the pupil size column
    time_col: the name of the timestamp column
    baseline_range: a tuple indicating the time range for the baseline. E.g., (-200, 0)
    method: "subtractive" (default) or "divisive".
    
    returns:
    a new dataframe with the corrected pupil size
    """
    
    if method not in ['subtractive','divisive']:
        raise Exception("Invalid baseline correction method. Please use 'subtractive' or 'divisive'.")
        
    bl_mean = df.loc[(df[time_col] >= baseline_range[0]) & (df[time_col] < baseline_range[1])][pup_col].median()
    
    if method == 'subtractive':
        df[pup_col+'_bsl'] = df[pup_col] - bl_mean
        
    elif method == 'divisive':
        df[pup_col+'_bsl'] = df[pup_col]/bl_mean
    
    return df