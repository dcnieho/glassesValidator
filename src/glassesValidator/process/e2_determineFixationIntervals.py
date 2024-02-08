#!/usr/bin/python

import pathlib
import math

import numpy as np
import pandas as pd

import I2MC
import matplotlib.pyplot as plt

from .. import config
from .. import utils


def process(working_dir, do_global_shift=True, max_dist_fac=.5, config_dir=None):
    working_dir  = pathlib.Path(working_dir)
    if config_dir is not None:
        config_dir = pathlib.Path(config_dir)

    print('processing: {}'.format(working_dir.name))
    utils.update_recording_status(working_dir, utils.Task.Fixation_Intervals_Determined, utils.Status.Running)

    # get info about this recording
    rec_info = utils.Recording.load_from_json(working_dir / utils.Recording.default_json_file_name)

    # open file with information about Aruco marker and Gaze target locations
    validationSetup = config.get_validation_setup(config_dir)

    # get interval coded to be analyzed
    analyzeFrames = utils.readMarkerIntervalsFile(working_dir / "markerInterval.tsv")
    if analyzeFrames is None:
        print('  no marker intervals defined for this recording, skipping')
        return

    # Read gaze on poster data
    gazePoster = utils.GazePoster.readDataFromFile(working_dir / 'gazePosterPos.tsv',analyzeFrames[0],analyzeFrames[-1],True)

    # get info about markers on our poster
    poster    = utils.Poster(config_dir, validationSetup, imHeight=-1)
    targets   = {ID: poster.targets[ID].center for ID in poster.targets}   # get centers of targets
    markerHalfSizeMm = poster.markerSize/2.

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
    if rec_info.eye_tracker in [utils.EyeTracker.Tobii_Glasses_2, utils.EyeTracker.Tobii_Glasses_3]:
        opt['cutoffstd'] = 1.8
    # decide what sampling frequency to tell I2MC about. It doesn't work with varying sampling frequency, nor
    # any random sampling frequency. For our purposes, getting it right is not important (internally I2MC only
    # uses sampling frequency for converting some of the time units to samples, other things are taken directly
    # from the time signal. So, we have working I2MC settings for a few sampling frequencies, and just choose
    # the nearest based on empirically determined sampling frequency.
    ts          = np.array([s.ts for v in gazePoster.values() for s in v])
    recFreq     = np.round(np.mean(1000./np.diff(ts)))    # Hz
    knownFreqs  = [30., 50., 60., 90., 120.]
    opt['freq'] = knownFreqs[np.abs(knownFreqs - recFreq).argmin()]
    if opt['freq']==120.:
        opt['downsamples']      = [2, 3, 5]
        opt['chebyOrder']       = 7
    elif opt['freq'] in [50., 60.]:
        opt['downsamples']      = [2, 5]
        opt['downsampFilter']   = False
    else:
        # 90 Hz, 30 Hz
        opt['downsamples']      = [2, 3]
        opt['downsampFilter']   = False

    # collect data
    qHasLeft        = np.any(np.logical_not(np.isnan([s.lGaze2D          for v in gazePoster.values() for s in v])))
    qHasRight       = np.any(np.logical_not(np.isnan([s.rGaze2D          for v in gazePoster.values() for s in v])))
    qHasWorld       = np.any(np.logical_not(np.isnan([s.wGaze2D          for v in gazePoster.values() for s in v])))
    qHasRay         = np.any(np.logical_not(np.isnan([s.gaze2DRay        for v in gazePoster.values() for s in v])))
    qHasHomography  = np.any(np.logical_not(np.isnan([s.gaze2DHomography for v in gazePoster.values() for s in v])))
    for ival in range(0,len(analyzeFrames)//2):
        gazePosterToAnal = {k:v for (k,v) in gazePoster.items() if k>=analyzeFrames[ival*2] and k<=analyzeFrames[ival*2+1]}
        data = {}
        data['time'] = np.array([s.ts for v in gazePosterToAnal.values() for s in v])
        if qHasLeft and qHasRight:
            # prefer using separate left and right eye signals, if available. Better I2MC robustness
            data['L_X']  = np.array([s.lGaze2D[0] for v in gazePosterToAnal.values() for s in v])
            data['L_Y']  = np.array([s.lGaze2D[1] for v in gazePosterToAnal.values() for s in v])
            data['R_X']  = np.array([s.rGaze2D[0] for v in gazePosterToAnal.values() for s in v])
            data['R_Y']  = np.array([s.rGaze2D[1] for v in gazePosterToAnal.values() for s in v])
        elif qHasWorld:
            # prefer over the below if provided, eye tracker may provide an 'improved' signal
            # here, e.g. AdHawk has an optional parallax correction
            data['average_X']  = np.array([s.wGaze2D[0] for v in gazePosterToAnal.values() for s in v])
            data['average_Y']  = np.array([s.wGaze2D[1] for v in gazePosterToAnal.values() for s in v])
        elif qHasRay:
            data['average_X']  = np.array([s.gaze2DRay[0] for v in gazePosterToAnal.values() for s in v])
            data['average_Y']  = np.array([s.gaze2DRay[1] for v in gazePosterToAnal.values() for s in v])
        elif qHasHomography:
            data['average_X']  = np.array([s.gaze2DHomography[0] for v in gazePosterToAnal.values() for s in v])
            data['average_Y']  = np.array([s.gaze2DHomography[1] for v in gazePosterToAnal.values() for s in v])
        else:
            raise RuntimeError('No data available to process')

        # run event classification to find fixations
        fix,data_I2MC,_ = I2MC.I2MC(data,opt,False)

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
        imgplot = plt.imshow(poster.getImgCopy(asRGB=True),extent=(np.array(poster.bbox)[[0,2,3,1]]),alpha=.5)
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

        f.savefig(str(working_dir / 'targetSelection_I2MC_interval_{}.png'.format(ival)))
        plt.close(f)

        # also make timseries plot of gaze data with fixations
        f = I2MC.plot.data_and_fixations(data, fix, fix_as_line=True, unit='mm', res=[[poster.bbox[0]-2*markerHalfSizeMm, poster.bbox[2]+2*markerHalfSizeMm], [poster.bbox[1]-2*markerHalfSizeMm, poster.bbox[3]+2*markerHalfSizeMm]])
        plt.gca().invert_yaxis()
        f.savefig(str(working_dir / 'targetSelection_I2MC_interval_{}_fixations.png'.format(ival)))
        plt.close(f)

        # store selected intervals
        df = pd.DataFrame()
        df.index.name = 'target'
        for i,t in zip(range(len(targets)),targets):
            if selected[i]==-999:
                continue
            df.loc[t,'marker_interval'] = ival+1
            df.loc[t,'start_timestamp'] = fix['startT'][selected[i]]
            df.loc[t,  'end_timestamp'] = fix[  'endT'][selected[i]]

        df.to_csv(str(working_dir / 'analysisInterval.tsv'), mode='w' if ival==0 else 'a', header=ival==0, sep='\t', na_rep='nan', float_format="%.3f")

    utils.update_recording_status(working_dir, utils.Task.Fixation_Intervals_Determined, utils.Status.Finished)