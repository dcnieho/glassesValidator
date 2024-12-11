import pathlib
import math

import numpy as np
import pandas as pd
import polars as pl

from glassesTools import gaze_worldref, plane, transforms
from glassesTools import utils as gt_utils

from .. import config
from .. import utils


def process(working_dir: str|pathlib.Path, config_dir: str|pathlib.Path=None,
            marker_interval_file_name: str='markerInterval.tsv',
            pose_file_name: str='posterPose.tsv',
            world_gaze_file_name: str='gazePosterPos.tsv',
            output_gaze_offset_file_name: str='gazeTargetOffset.tsv'):
    from . import DataQualityType
    working_dir  = pathlib.Path(working_dir)
    if config_dir is not None:
        config_dir = pathlib.Path(config_dir)

    print('processing: {}'.format(working_dir.name))
    utils.update_recording_status(working_dir, utils.Task.Target_Offsets_Computed, utils.Status.Running, skip_if_missing=True)

    # open file with information about Aruco marker and Gaze target locations
    validationSetup = config.get_validation_setup(config_dir)

    # get interval coded to be analyzed
    analyzeFrames = utils.readMarkerIntervalsFile(working_dir / marker_interval_file_name)
    if analyzeFrames is None:
        print('  no marker intervals defined for this recording, skipping')
        return

    # Read camera pose w.r.t. poster
    poses = plane.read_dict_from_file(working_dir / pose_file_name, analyzeFrames)

    # Read gaze on poster data
    gazesPoster = gaze_worldref.read_dict_from_file(working_dir / world_gaze_file_name, analyzeFrames)

    # get info about markers on our poster
    poster          = config.poster.Poster(config_dir, validationSetup)
    targets         = {ID: np.append(poster.targets[ID].center,  0.  ) for ID in poster.targets}   # get centers of targets
    distMm          = validationSetup['distance']*10.
    targets_for_homo= {ID: np.append(poster.targets[ID].center,distMm) for ID in poster.targets}   # get centers of targets

    # get types of data quality to compute
    dq_types = [DataQualityType.viewpos_vidpos_homography,DataQualityType.pose_vidpos_homography,DataQualityType.pose_vidpos_ray,DataQualityType.pose_world_eye,DataQualityType.pose_left_eye,DataQualityType.pose_right_eye]

    # for each frame during analysis interval, determine offset
    # (angle) of gaze (each eye) to each of the targets
    dfs = []
    for idx,iv in enumerate(analyzeFrames):
        gazesPosterToAnal= {k:v for (k,v) in gazesPoster.items() if k>=iv[0] and k<=iv[1]}
        if not gazesPosterToAnal:
            raise RuntimeError(f'There is no gaze data on the poster for validation interval (frames {iv[0]} to {iv[1]}), cannot proceed. This may be because there was no gaze during this interval or because the poster was not detected.')

        frameIdxs        =           [k                                     for k,v in gazesPosterToAnal.items()  for s in v]
        ts               = np.vstack([s.timestamp                           for v in gazesPosterToAnal.values() for s in v])
        oriLeft          = np.vstack([s.gazeOriCamLeft                      for v in gazesPosterToAnal.values() for s in v])
        oriRight         = np.vstack([s.gazeOriCamRight                     for v in gazesPosterToAnal.values() for s in v])
        gaze3DLeft       = np.vstack([s.gazePosCamLeft                      for v in gazesPosterToAnal.values() for s in v])
        gaze3DRight      = np.vstack([s.gazePosCamRight                     for v in gazesPosterToAnal.values() for s in v])
        gaze2DLeft       = np.vstack([s.gazePosPlane2DLeft                  for v in gazesPosterToAnal.values() for s in v])
        gaze2DRight      = np.vstack([s.gazePosPlane2DRight                 for v in gazesPosterToAnal.values() for s in v])
        gaze3DWorld      = np.vstack([s.gazePosCamWorld                     for v in gazesPosterToAnal.values() for s in v])
        gaze2DWorld      = np.vstack([s.gazePosPlane2DWorld                 for v in gazesPosterToAnal.values() for s in v])
        gaze3DRay        = np.vstack([s.gazePosCam_vidPos_ray               for v in gazesPosterToAnal.values() for s in v])
        gaze2DRay        = np.vstack([s.gazePosPlane2D_vidPos_ray           for v in gazesPosterToAnal.values() for s in v])
        gaze3DHomography = np.vstack([s.gazePosCam_vidPos_homography        for v in gazesPosterToAnal.values() for s in v])
        gaze2DHomography = np.vstack([s.gazePosPlane2D_vidPos_homography    for v in gazesPosterToAnal.values() for s in v])

        offset = np.empty((oriLeft.shape[0],len(dq_types),len(targets),2))
        offset[:] = np.nan

        for s in range(oriLeft.shape[0]):
            if frameIdxs[s] not in poses:
                continue

            # all based on pose info
            targets_cam: dict[int,np.ndarray] = {}
            for e in range(len(dq_types)):
                match dq_types[e]:
                    case DataQualityType.viewpos_vidpos_homography | DataQualityType.pose_vidpos_homography:
                        # from camera perspective, using homography
                        # pose_vidpos_homography   : using pose info
                        # viewpos_vidpos_homography: using assumed viewing distance
                        ori         = np.zeros(3)
                        gaze        = gaze3DHomography[s,:]
                        gazePoster  = gaze2DHomography[s,:]
                    case DataQualityType.pose_vidpos_ray:
                        # from camera perspective, using 3D gaze point ray
                        ori         = np.zeros(3)
                        gaze        = gaze3DRay[s,:]
                        gazePoster  = gaze2DRay[s,:]
                    case DataQualityType.pose_world_eye:
                        # using 3D world gaze position, with respect to eye tracker reference frame's origin
                        ori         = np.zeros(3)
                        gaze        = gaze3DWorld[s,:]
                        gazePoster  = gaze2DWorld[s,:]
                    case DataQualityType.pose_left_eye:
                        ori         = oriLeft[s,:]
                        gaze        = gaze3DLeft[s,:]
                        gazePoster  = gaze2DLeft[s,:]
                    case DataQualityType.pose_right_eye:
                        ori         = oriRight[s,:]
                        gaze        = gaze3DRight[s,:]
                        gazePoster  = gaze2DRight[s,:]

                if np.any(np.isnan(ori)) or np.any(np.isnan(gazePoster)) or (dq_types[e]!=DataQualityType.viewpos_vidpos_homography and np.any(np.isnan(gaze))):
                    continue

                for ti,t in enumerate(targets):
                    if dq_types[e]==DataQualityType.viewpos_vidpos_homography:
                        # get vectors based on assumed viewing distance (from config), without using pose info
                        vGaze   = np.array([gazePoster[0], gazePoster[1], distMm])
                        vTarget = targets_for_homo[t]
                    else:
                        # use 3D vectors known given pose information
                        if t not in targets_cam:
                            targets_cam[t] = poses[frameIdxs[s]].world_frame_to_cam(targets[t])
                        target = targets_cam[t]

                        # get vectors from origin to target and to gaze point
                        vGaze   = gaze  -ori
                        vTarget = target-ori

                    # get offset
                    ang2D           = transforms.angle_between(vTarget,vGaze)
                    # decompose in horizontal/vertical (in poster space)
                    onPosterAngle   = math.atan2(gazePoster[1]-targets[t][1], gazePoster[0]-targets[t][0])
                    offset[s,e,ti,:]= ang2D*np.array([math.cos(onPosterAngle), math.sin(onPosterAngle)])

        # organize for output and write to file
        # 1. create cartesian product of sample index, eye and target indices
        # order of inputs needed to get expected output is a mystery to me, but screw it, works
        dat = gt_utils.cartesian_product(np.arange(len(dq_types)),np.arange(offset.shape[0]),[t for t in targets])
        # 2. put into data frame
        df                      = pd.DataFrame()
        df['timestamp']         = ts[dat[:,1],0]
        df['marker_interval']   = idx+1
        df['type']              = [str(dq_types[e]) for e in dat[:,0]]
        df['target']            = dat[:,2]
        df                      = pd.concat([df, pd.DataFrame(np.reshape(offset,(-1,2)),columns=['offset_x','offset_y'])],axis=1)
        df                      = df.dropna(axis=0, subset=['offset_x','offset_y'])  # drop any missing data
        # 3. store for writing to file
        dfs.append(df)

    # all done, write to file (use polars as that library saves to file waaay faster)
    df = pd.concat(dfs)
    df = pl.from_pandas(df)
    df.write_csv(working_dir / output_gaze_offset_file_name, separator='\t', null_value='nan', float_precision=3)

    utils.update_recording_status(working_dir, utils.Task.Target_Offsets_Computed, utils.Status.Finished, skip_if_missing=True)
