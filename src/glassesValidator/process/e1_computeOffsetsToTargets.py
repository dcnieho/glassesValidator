#!/usr/bin/python

import pathlib
import math

import cv2
import numpy as np
import pandas as pd

from .. import config
from .. import utils


def process(inputDir, configDir=None):
    from . import DataQualityType
    inputDir  = pathlib.Path(inputDir)
    if configDir is not None:
        configDir = pathlib.Path(configDir)

    print('processing: {}'.format(inputDir.name))
    utils.update_recording_status(inputDir, utils.Task.Target_Offsets_Computed, utils.Status.Running)
    
    # open file with information about Aruco marker and Gaze target locations
    validationSetup = config.get_validation_setup(configDir)

    # get interval coded to be analyzed
    analyzeFrames = utils.readMarkerIntervalsFile(inputDir / "markerInterval.tsv")
    if analyzeFrames is None:
        print('  no marker intervals defined for this recording, skipping')
        return

    # Read pose of marker board
    poses = utils.BoardPose.readDataFromFile(inputDir / 'boardPose.tsv',analyzeFrames[0],analyzeFrames[-1],True)

    # Read gaze on board data
    gazeWorld = utils.GazeWorld.readDataFromFile(inputDir / 'gazeWorldPos.tsv',analyzeFrames[0],analyzeFrames[-1],True)

    # get info about markers on our board
    reference = utils.Reference(configDir, validationSetup)
    targets = {ID: reference.targets[ID].center for ID in reference.targets}   # get centers of targets

    # get types of data quality to compute
    dq_types = [DataQualityType.viewdist_vidpos_homography,DataQualityType.pose_vidpos_homography,DataQualityType.pose_vidpos_ray,DataQualityType.pose_left_eye,DataQualityType.pose_right_eye]
    
    # for each frame during analysis interval, determine offset
    # (angle) of gaze (each eye) to each of the targets
    for ival in range(0,len(analyzeFrames)//2):
        gazeWorldToAnal = {k:v for (k,v) in gazeWorld.items() if k>=analyzeFrames[ival*2] and k<=analyzeFrames[ival*2+1]}
        frameIdxs       =           [k                for k,v in gazeWorldToAnal.items()  for s in v]
        ts              = np.vstack([s.ts               for v in gazeWorldToAnal.values() for s in v])
        oriLeft         = np.vstack([s.lGazeOrigin      for v in gazeWorldToAnal.values() for s in v])
        oriRight        = np.vstack([s.rGazeOrigin      for v in gazeWorldToAnal.values() for s in v])
        gaze3DLeft      = np.vstack([s.lGaze3D          for v in gazeWorldToAnal.values() for s in v])
        gaze3DRight     = np.vstack([s.rGaze3D          for v in gazeWorldToAnal.values() for s in v])
        gaze2DLeft      = np.vstack([s.lGaze2D          for v in gazeWorldToAnal.values() for s in v])
        gaze2DRight     = np.vstack([s.rGaze2D          for v in gazeWorldToAnal.values() for s in v])
        gaze3DRay       = np.vstack([s.gaze3DRay        for v in gazeWorldToAnal.values() for s in v])
        gaze2DRay       = np.vstack([s.gaze2DRay        for v in gazeWorldToAnal.values() for s in v])
        gaze3DHomography= np.vstack([s.gaze3DHomography for v in gazeWorldToAnal.values() for s in v])
        gaze2DHomography= np.vstack([s.gaze2DHomography for v in gazeWorldToAnal.values() for s in v])

        offset = np.empty((oriLeft.shape[0],len(dq_types),len(targets),2))
        offset[:] = np.nan

        for s in range(oriLeft.shape[0]):
            if frameIdxs[s] not in poses:
                continue
            if poses[frameIdxs[s]].rVec is not None:
                RBoard  = cv2.Rodrigues(poses[frameIdxs[s]].rVec)[0]
                RtBoard = np.hstack((RBoard, poses[frameIdxs[s]].tVec.reshape(3,1)))
            else:
                RtBoard = np.full([3,4], np.nan)

            # all based on pose info
            for e in range(len(dq_types)):
                match dq_types[e]:
                    case DataQualityType.viewdist_vidpos_homography | DataQualityType.pose_vidpos_homography:
                        # from camera perspective, using homography
                        # pose_vidpos_homography    : using pose info
                        # viewdist_vidpos_homography: using assumed viewing distance
                        ori         = np.zeros(3)
                        gaze        = gaze3DHomography[s,:]
                        gazeBoard   = gaze2DHomography[s,:]
                    case DataQualityType.pose_vidpos_ray:
                        # from camera perspective, using 3D gaze point ray
                        ori         = np.zeros(3)
                        gaze        = gaze3DRay[s,:]
                        gazeBoard   = gaze2DRay[s,:]
                    case DataQualityType.pose_left_eye:
                        ori         = oriLeft[s,:]
                        gaze        = gaze3DLeft[s,:]
                        gazeBoard   = gaze2DLeft[s,:]
                    case DataQualityType.pose_right_eye:
                        ori         = oriRight[s,:]
                        gaze        = gaze3DRight[s,:]
                        gazeBoard   = gaze2DRight[s,:]
                    

                for ti,t in enumerate(targets):
                    if dq_types[e]==DataQualityType.viewdist_vidpos_homography:
                        # get vectors based on assumed viewing distance (from config), without using pose info
                        distMm  = validationSetup['distance']*10.
                        vGaze   = np.array([gazeBoard[0] , gazeBoard[1] , distMm])
                        vTarget = np.array([targets[t][0], targets[t][1], distMm])
                    else:
                        # use 3D vectors known given pose information
                        target  = np.matmul(RtBoard,np.array([targets[t][0], targets[t][1], 0., 1.]))
            
                        # get vectors from origin to target and to gaze point
                        vGaze   = gaze  -ori
                        vTarget = target-ori
        
                    # get offset
                    ang2D           = utils.angle_between(vTarget,vGaze)
                    # decompose in horizontal/vertical (in board space)
                    onBoardAngle    = math.atan2(gazeBoard[1]-targets[t][1],gazeBoard[0]-targets[t][0])
                    offset[s,e,ti,:]= ang2D*np.array([math.cos(onBoardAngle), math.sin(onBoardAngle)])

        # organize for output and write to file
        # 1. create cartesian product of sample index, eye and target indices
        # order of inputs needed to get expected output is a mystery to me, but screw it, works
        dat = utils.cartesian_product(np.arange(5),np.arange(offset.shape[0]),[t for t in targets])
        # 2. put into data frame
        df                      = pd.DataFrame()
        df['timestamp']         = ts[dat[:,1],0]
        df['marker_interval']   = ival+1
        df['type']              = [dq_types[e] for e in dat[:,0]]
        df['target']            = dat[:,2]
        df                      = pd.concat([df, pd.DataFrame(np.reshape(offset,(-1,2)),columns=['offset_x','offset_y'])],axis=1)
        df                      = df.dropna(axis=0, subset=['offset_x','offset_y'])  # drop any missing data
        # 3. write to file
        df.to_csv(str(inputDir / 'gazeTargetOffset.tsv'), mode='w' if ival==0 else 'a', header=ival==0, index=False, sep='\t', na_rep='nan', float_format="%.3f")

    utils.update_recording_status(inputDir, utils.Task.Target_Offsets_Computed, utils.Status.Finished)
