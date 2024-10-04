import pathlib
import math

import numpy as np
import pandas as pd
from typing import Any

import I2MC
import matplotlib.pyplot as plt

from glassesTools.eyetracker import EyeTracker
from glassesTools import gaze_worldref, recording

from .. import config
from .. import utils


def process(working_dir, config_dir=None, do_global_shift=True, max_dist_fac=.5,
            I2MC_settings_override: dict[str,Any]=None,
            marker_interval_file_name: str='markerInterval.tsv',
            world_gaze_file_name: str='gazePosterPos.tsv',
            fixation_detection_file_name_prefix: str='targetSelection_I2MC_',
            output_analysis_interval_file_name: str='analysisInterval.tsv'):
    working_dir  = pathlib.Path(working_dir)
    if config_dir is not None:
        config_dir = pathlib.Path(config_dir)

    print('processing: {}'.format(working_dir.name))
    utils.update_recording_status(working_dir, utils.Task.Fixation_Intervals_Determined, utils.Status.Running, skip_if_missing=True)

    # get info about this recording
    rec_info = recording.Recording.load_from_json(working_dir / recording.Recording.default_json_file_name)

    # open file with information about Aruco marker and Gaze target locations
    validationSetup = config.get_validation_setup(config_dir)

    # get interval coded to be analyzed
    analyzeFrames = utils.readMarkerIntervalsFile(working_dir / marker_interval_file_name)
    if analyzeFrames is None:
        print('  no marker intervals defined for this recording, skipping')
        return

    # Read gaze on poster data
    gazePoster = gaze_worldref.read_dict_from_file(working_dir / world_gaze_file_name, analyzeFrames)

    # get info about markers on our poster
    poster    = config.poster.Poster(config_dir, validationSetup)
    targets   = {ID: poster.targets[ID].center for ID in poster.targets}   # get centers of targets
    markerHalfSizeMm = poster.marker_size/2.

    # run I2MC on data in poster space
    # set I2MC options
    opt = {'xres': None, 'yres': None}  # dummy values for required options
    opt['missingx']         = math.nan
    opt['missingy']         = math.nan
    opt['maxdisp']          = 50        # mm
    opt['windowtimeInterp'] = .25       # s
    opt['maxMergeDist']     = 20        # mm
    opt['maxMergeTime']     = 81        # ms
    opt['minFixDur']        = 50        # ms
    if rec_info.eye_tracker in [EyeTracker.Tobii_Glasses_2, EyeTracker.Tobii_Glasses_3]:
        opt['cutoffstd'] = 1.8
    # decide what sampling frequency to tell I2MC about. It doesn't work with varying sampling frequency, nor
    # any random sampling frequency. For our purposes, getting it right is not important (internally I2MC only
    # uses sampling frequency for converting some of the time units to samples, other things are taken directly
    # from the time signal. So, we have working I2MC settings for a few sampling frequencies, and just choose
    # the nearest based on empirically determined sampling frequency.
    ts          = np.array([s.timestamp for v in gazePoster.values() for s in v])
    recFreq     = np.round(np.mean(1000./np.diff(ts)))    # Hz
    knownFreqs  = [30., 50., 60., 90., 120., 200.]
    opt['freq'] = knownFreqs[np.abs(knownFreqs - recFreq).argmin()]
    if opt['freq']==200.:
        pass    # defaults are good
    elif opt['freq']==120.:
        opt['downsamples']      = [2, 3, 5]
        opt['chebyOrder']       = 7
    elif opt['freq'] in [50., 60.]:
        opt['downsamples']      = [2, 5]
        opt['downsampFilter']   = False
    else:
        # 90 Hz, 30 Hz
        opt['downsamples']      = [2, 3]
        opt['downsampFilter']   = False
    # apply setting overrides from caller, if any
    if I2MC_settings_override:
        for k in I2MC_settings_override:
            if I2MC_settings_override[k] is not None:
                opt[k] = I2MC_settings_override[k]

    # collect data
    qHasLeft        = np.any(np.logical_not(np.isnan([s.gazePosPlane2DLeft               for v in gazePoster.values() for s in v])))
    qHasRight       = np.any(np.logical_not(np.isnan([s.gazePosPlane2DRight              for v in gazePoster.values() for s in v])))
    qHasWorld       = np.any(np.logical_not(np.isnan([s.gazePosPlane2DWorld              for v in gazePoster.values() for s in v])))
    qHasRay         = np.any(np.logical_not(np.isnan([s.gazePosPlane2D_vidPos_ray        for v in gazePoster.values() for s in v])))
    qHasHomography  = np.any(np.logical_not(np.isnan([s.gazePosPlane2D_vidPos_homography for v in gazePoster.values() for s in v])))
    for idx,iv in enumerate(analyzeFrames):
        gazePosterToAnal = {k:v for (k,v) in gazePoster.items() if k>=iv[0] and k<=iv[1]}
        # need these for the plots. Doing detection on the world data if available is good, but we should
        # plot using the ray (if available) or homography data, as that corresponds to the gaze visualization
        # provided in the software, and for some recordings/devices the world-based coordinates can be far off.
        if qHasRay:
            ray_x  = np.array([s.gazePosPlane2D_vidPos_ray[0] for v in gazePosterToAnal.values() for s in v])
            ray_y  = np.array([s.gazePosPlane2D_vidPos_ray[1] for v in gazePosterToAnal.values() for s in v])
        elif qHasHomography:
            homography_x  = np.array([s.gazePosPlane2D_vidPos_homography[0] for v in gazePosterToAnal.values() for s in v])
            homography_y  = np.array([s.gazePosPlane2D_vidPos_homography[1] for v in gazePosterToAnal.values() for s in v])
        data = {}
        data['time'] = np.array([s.timestamp for v in gazePosterToAnal.values() for s in v])
        qNeedRecalcFix = False
        if qHasLeft and qHasRight:
            # prefer using separate left and right eye signals, if available. Better I2MC robustness
            data['L_X']  = np.array([s.gazePosPlane2DLeft[0]  for v in gazePosterToAnal.values() for s in v])
            data['L_Y']  = np.array([s.gazePosPlane2DLeft[1]  for v in gazePosterToAnal.values() for s in v])
            data['R_X']  = np.array([s.gazePosPlane2DRight[0] for v in gazePosterToAnal.values() for s in v])
            data['R_Y']  = np.array([s.gazePosPlane2DRight[1] for v in gazePosterToAnal.values() for s in v])
            qNeedRecalcFix = True
        elif qHasWorld:
            # prefer over the below if provided, eye tracker may provide an 'improved' signal
            # here, e.g. AdHawk has an optional parallax correction
            data['average_X']  = np.array([s.gazePosPlane2DWorld[0] for v in gazePosterToAnal.values() for s in v])
            data['average_Y']  = np.array([s.gazePosPlane2DWorld[1] for v in gazePosterToAnal.values() for s in v])
            qNeedRecalcFix = True
        elif qHasRay:
            data['average_X']  = ray_x
            data['average_Y']  = ray_y
        elif qHasHomography:
            data['average_X']  = homography_x
            data['average_Y']  = homography_y
        else:
            raise RuntimeError('No data available to process')

        # run event classification to find fixations
        fix,data_I2MC,par_I2MC = I2MC.I2MC(data,opt,False)
        if qNeedRecalcFix:
            # replace data with gaze position on video data
            data_I2MC = data_I2MC.drop(columns=['L_X','L_Y','R_X','R_Y'],errors='ignore')
            data_I2MC['average_X'] = ray_x if qHasRay else homography_x
            data_I2MC['average_Y'] = ray_y if qHasRay else homography_y
            # recalculate fixation positions based on gaze position on video data
            fix = I2MC.get_fixations(data_I2MC['finalweights'].array, data_I2MC['time'].array, data_I2MC['average_X'], data_I2MC['average_Y'], data_I2MC['average_missing'], par_I2MC)

        # for each target, find closest fixation
        minDur      = 100       # ms
        used        = np.zeros((fix['start'].size),dtype='bool')
        selected    = np.empty((len(targets),),dtype='int')
        selected[:] = -999

        t_x = np.array([targets[t][0] for t in targets])
        t_y = np.array([targets[t][1] for t in targets])
        off_x = off_y = off_t_x = off_t_y = 0.
        if do_global_shift:
            # first, center the problem. That means determine and remove any overall shift from the
            # data and the targets, to improve robustness of assigning fixations points to targets.
            # Else, if all data is e.g. shifted up by more than half the distance between
            # validation targets, target assignment would fail
            # N.B.: use output data from I2MC, which always has an average gaze signal
            off_x = data_I2MC['average_X'].mean()
            off_y = data_I2MC['average_Y'].mean()
            off_t_x = t_x.mean()
            off_t_y = t_y.mean()

        # we furthermore do not assign a fixation to a target if the closest fixation is more than
        # half the intertarget distance away
        # determine intertarget distance, if possible
        dist_lim = np.inf
        if len(t_x)>1:
            # arbitrarily take first target and find closest target to it
            dist = np.hypot(t_x[0]-t_x[1:], t_y[0]-t_y[1:])
            min_dist = dist.min()
            if min_dist > 0:
                dist_lim = min_dist*max_dist_fac

        for i,t in zip(range(len(targets)),targets):
            if np.all(used):
                # all fixations used up, can't assign anything to remaining targets
                continue
            # select fixation
            dist                    = np.hypot(fix['xpos']-off_x-(targets[t][0]-off_t_x), fix['ypos']-off_y-(targets[t][1]-off_t_y))
            dist[used]              = np.inf    # make sure fixations already bound to a target are not used again
            dist[fix['dur']<minDur] = np.inf    # make sure that fixations that are too short are not selected
            iFix        = np.argmin(dist)
            if dist[iFix]<=dist_lim:
                selected[i] = iFix
                used[iFix]  = True

        # make plot of data overlaid on poster, and show for each target which fixation
        # was selected
        f       = plt.figure(dpi=300)
        imgplot = plt.imshow(poster.get_ref_image(as_RGB=True),extent=(np.array(poster.bbox)[[0,2,3,1]]),alpha=.5)
        plt.plot(fix['xpos'],fix['ypos'],'b-')
        plt.plot(fix['xpos'],fix['ypos'],'go')
        plt.xlim([poster.bbox[0]-markerHalfSizeMm, poster.bbox[2]+markerHalfSizeMm])
        plt.ylim([poster.bbox[1]-markerHalfSizeMm, poster.bbox[3]+markerHalfSizeMm])
        plt.gca().invert_yaxis()
        for i,t in zip(range(len(targets)),targets):
            if selected[i]==-999:
                continue
            plt.plot([fix['xpos'][selected[i]], targets[t][0]], [fix['ypos'][selected[i]], targets[t][1]],'r-')

        plt.xlabel('mm')
        plt.ylabel('mm')

        f.savefig(str(working_dir / f'{fixation_detection_file_name_prefix}interval_{idx}.png'))
        plt.close(f)

        # also make timseries plot of gaze data with fixations
        f = I2MC.plot.data_and_fixations(data_I2MC, fix, fix_as_line=True, unit='mm', res=[[poster.bbox[0]-2*markerHalfSizeMm, poster.bbox[2]+2*markerHalfSizeMm], [poster.bbox[1]-2*markerHalfSizeMm, poster.bbox[3]+2*markerHalfSizeMm]])
        plt.gca().invert_yaxis()
        f.savefig(str(working_dir / f'{fixation_detection_file_name_prefix}interval_{idx}_fixations.png'))
        plt.close(f)

        # store selected intervals
        df = pd.DataFrame()
        df.index.name = 'target'
        for i,t in zip(range(len(targets)),targets):
            if selected[i]==-999:
                continue
            df.loc[t,'marker_interval'] = idx+1
            df.loc[t,'start_timestamp'] = fix['startT'][selected[i]]
            df.loc[t,  'end_timestamp'] = fix[  'endT'][selected[i]]

        df.to_csv(str(working_dir / output_analysis_interval_file_name), mode='w' if idx==0 else 'a', header=idx==0, sep='\t', na_rep='nan', float_format="%.3f")

    utils.update_recording_status(working_dir, utils.Task.Fixation_Intervals_Determined, utils.Status.Finished, skip_if_missing=True)