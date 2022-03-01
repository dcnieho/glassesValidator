#!/usr/bin/python

import sys
from pathlib import Path
import math

import cv2
import numpy as np
import pandas as pd
import csv
import warnings

import utils
import I2MC

import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def process(inputDir,basePath):
    print('processing: {}'.format(inputDir.name))
    
    configDir = basePath / "config"
    # open file with information about Aruco marker and Gaze target locations
    validationSetup = utils.getValidationSetup(configDir)

    # get interval coded to be analyzed
    analyzeFrames = utils.getAnalysisIntervals(inputDir / "analysisInterval.tsv")
    if analyzeFrames is None:
        print('  no analysis intervals defined for this file, skipping')
        return

    # Read pose of marker board
    rVec,tVec = utils.getMarkerBoardPose(inputDir / 'boardPose.tsv',analyzeFrames[0],analyzeFrames[-1],True)
    reference = utils.Reference(str(inputDir / 'referenceBoard.png'), configDir, validationSetup)

    # Read gaze on board data
    gazeWorld = utils.getGazeWorldData(inputDir / 'gazeWorldPos.tsv',analyzeFrames[0],analyzeFrames[-1],True)

    # get info about markers on our board
    # Aruco markers have numeric keys, gaze targets have keys starting with 't'
    knownMarkers, markerBBox = utils.getKnownMarkers(configDir, validationSetup)
    targets = {}
    for key in knownMarkers:
        if key.startswith('t'):
            targets[int(key[1:])] = knownMarkers[key].center
    cellSizeMm       = 2.*math.tan(math.radians(.5))*validationSetup['distance']*10
    markerHalfSizeMm = cellSizeMm*validationSetup['markerSide']/2.
    
    # run I2MC on data in board space
    # 1. set I2MC options
    opt = {'xres': None, 'yres': None}  # dummy values for required options
    opt['missingx']         = math.nan
    opt['missingy']         = math.nan
    opt['freq']             = 50        # Hz
    opt['downsamples']      = [2, 5]
    opt['downsampFilter']   = False
    opt['maxdisp']          = 50        # mm
    opt['windowtimeInterp'] = .25       # s
    opt['maxMergeDist']     = 20        # mm
    opt['maxMergeTime']     = 81        # ms
    opt['minFixDur']        = 50        # ms
    # 2. collect data
    for ival in range(0,len(analyzeFrames)//2):
        gazeWorldToAnal = {k:v for (k,v) in gazeWorld.items() if k>=analyzeFrames[ival*2] and k<=analyzeFrames[ival*2+1]}
        data = {}
        data['time'] = np.array([s.ts for v in gazeWorldToAnal.values() for s in v])-gazeWorldToAnal[analyzeFrames[ival*2]][0].ts
        data['L_X']  = np.array([s.lGaze2D[0] for v in gazeWorldToAnal.values() for s in v])
        data['L_Y']  = np.array([s.lGaze2D[1] for v in gazeWorldToAnal.values() for s in v])
        data['R_X']  = np.array([s.rGaze2D[0] for v in gazeWorldToAnal.values() for s in v])
        data['R_Y']  = np.array([s.rGaze2D[1] for v in gazeWorldToAnal.values() for s in v])
        # 3. run event classification to find fixations
        fix,dat,par = I2MC.I2MC(data,opt,False)

        # for each target, find closest fixation
        minDur      = 150       # ms
        used        = np.zeros((fix['start'].size),dtype='bool')
        selected    = np.empty((len(targets),),dtype='int')
        selected[:] = -1

        for i,t in zip(range(len(targets)),targets):
            nUsed = np.invert(used)

            # select fixation
            dist                    = np.hypot(fix['xpos']-targets[t][0], fix['ypos']-targets[t][1])
            dist[used]              = math.inf  # make sure fixations already bound to a target are not used again
            dist[fix['dur']<minDur] = math.inf  # make sure that fixations that are too short are not selected
            iFix        = np.argmin(dist)
            selected[i] = iFix
            used[iFix]  = True

        # make plot of data overlaid on board, and show for each target which fixation
        # was selected
        f       = plt.figure(dpi=300)
        imgplot = plt.imshow(reference.getImgCopy(),extent=(np.array(markerBBox)[[0,2,3,1]]),alpha=.5)
        plt.plot(fix['xpos'],fix['ypos'],'b-')
        plt.plot(fix['xpos'],fix['ypos'],'go')
        plt.xlim([markerBBox[0]-markerHalfSizeMm, markerBBox[2]+markerHalfSizeMm])
        plt.ylim([markerBBox[3]-markerHalfSizeMm, markerBBox[1]+markerHalfSizeMm])
        for i,t in zip(range(len(selected)),targets):
            plt.plot([fix['xpos'][selected[i]], targets[t][0]], [fix['ypos'][selected[i]], targets[t][1]],'r-')
        
        f.savefig(str(inputDir / 'target_selection.png'))
        plt.close(f)
    
        # 2. calculate offsets from targets
        # a. determine which samples to process (those during selected fixation)
        whichTarget     = np.empty((len(data['time']),),dtype='int')
        whichTarget[:]  = -1
        for i,t in zip(range(len(selected)),targets):
            whichTarget[fix['start'][selected[i]]:fix['end'][selected[i]]+1] = t
        # b. get offset per sample
        offset = np.empty((2,2,len(whichTarget),))
        offset[:] = np.nan
        frameIdxs = [k for k,v in gazeWorldToAnal.items() for s in v]
        oriL  = np.vstack([s.lGazeOrigin for v in gazeWorldToAnal.values() for s in v])
        oriR  = np.vstack([s.rGazeOrigin for v in gazeWorldToAnal.values() for s in v])
        gaze3L= np.vstack([s.lGaze3D     for v in gazeWorldToAnal.values() for s in v])
        gaze3R= np.vstack([s.rGaze3D     for v in gazeWorldToAnal.values() for s in v])
        gaze2L= np.vstack([s.lGaze2D     for v in gazeWorldToAnal.values() for s in v])
        gaze2R= np.vstack([s.rGaze2D     for v in gazeWorldToAnal.values() for s in v])
        for s in range(len(whichTarget)):
            t = whichTarget[s]
            if t==-1:
                continue

            for e in range(2):
                if e==0:
                    ori         = oriL[s,:]
                    gaze        = gaze3L[s,:]
                    gazeBoard   = gaze2L[s,:]
                else:
                    ori         = oriR[s,:]
                    gaze        = gaze3R[s,:]
                    gazeBoard   = gaze2R[s,:]

                if frameIdxs[s] not in rVec:
                    continue

                RBoard  = cv2.Rodrigues(rVec[frameIdxs[s]])[0]
                RtBoard = np.hstack((RBoard, tVec[frameIdxs[s]].reshape(3,1)))
                target  = np.matmul(RtBoard,np.array([targets[t][0], targets[t][1], 0., 1.]))
            
                # get vectors from origin to target and to gaze point
                vGaze   = gaze  -ori;
                vTarget = target-ori;
        
                # get offset
                ang2D           = utils.angle_between(vTarget,vGaze)
                # decompose in horizontal/vertical (in board space)
                onBoardAngle    = math.atan2(gazeBoard[1]-targets[t][1],gazeBoard[0]-targets[t][0])
                offset[:,e,s]   = ang2D*np.array([math.cos(onBoardAngle), math.sin(onBoardAngle)])

        # Add average of the two eyes
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # ignore warnings from np.nanmean and np.nanstd
            offset = np.concatenate((offset,np.nanmean(offset,axis=1,keepdims=True)),axis=1)

        # determine order in which targets were looked at
        u,i=np.unique(whichTarget, return_index=True)
        i=np.delete(i,u==-1)
        i=np.argsort(i)
        lookOrder = np.argsort(i)+1

        # 3. calculate metrics per target
        accuracy2D = np.empty((len(targets),2,3))       # nTarget x [X Y] x nEye (L,R,Avg)
        accuracy2D[:] = np.nan
        accuracy1D = np.empty((len(targets),3))         # nTarget x nEye (L,R,Avg)
        accuracy1D[:] = np.nan
    
        RMS2D = np.empty((len(targets),2,3))            # nTarget x [X Y] x nEye (L,R,Avg)
        RMS2D[:] = np.nan
        RMS1D = np.empty((len(targets),3))              # nTarget x nEye (L,R,Avg)
        RMS1D[:] = np.nan
    
        STD2D = np.empty((len(targets),2,3))            # nTarget x [X Y] x nEye (L,R,Avg)
        STD2D[:] = np.nan
        STD1D = np.empty((len(targets),3))              # nTarget x nEye (L,R,Avg)
        STD1D[:] = np.nan
    
        dataLoss = np.empty((len(targets),3))           # nTarget x nEye (L,R,Avg)
        dataLoss[:] = np.nan
        for i,t in zip(range(len(selected)),targets):
            qData = whichTarget==t
        
            with warnings.catch_warnings():
                warnings.simplefilter("ignore") # ignore warnings from np.nanmean and np.nanstd
                accuracy2D[i,:,:] = np.nanmean(offset[:,:,qData],axis=2)
                accuracy1D[i,:]   = np.nanmean(np.hypot(offset[0,:,qData],offset[1,:,qData]),axis=0)
    
                RMS2D[i,:,:]      = np.sqrt(np.nanmean(np.diff(offset[:,:,qData],axis=2)**2,axis=2))
                RMS1D[i,:]        = np.hypot(RMS2D[i,0,:],RMS2D[i,1,:])
    
                STD2D[i,:,:]      = np.nanstd(offset[:,:,qData],axis=2,ddof=1)
                STD1D[i,:]        = np.hypot(STD2D[i,0,:],STD2D[i,1,:])
    
                dataLoss[i,:]     = np.sum(np.isnan(offset[0,:,qData]),axis=0)/np.sum(qData)
        
        # organize for output and write to file
        df = pd.DataFrame()
        df.index.name = 'target'
        for i,t in zip(range(len(selected)),targets):
            df.loc[t,'interval'] = ival+1
            df.loc[t,'pos_x'] = targets[t][0]
            df.loc[t,'pos_y'] = targets[t][1]
            df.loc[t,'order'] = lookOrder[i]
            df.loc[t,'acc_left_x'] = accuracy2D[i,0,0]
            df.loc[t,'acc_left_y'] = accuracy2D[i,1,0]
            df.loc[t,'acc_left'] = accuracy1D[i,0]
            df.loc[t,'acc_right_x'] = accuracy2D[i,0,1]
            df.loc[t,'acc_right_y'] = accuracy2D[i,1,1]
            df.loc[t,'acc_right'] = accuracy1D[i,1]
            df.loc[t,'acc_avg_x'] = accuracy2D[i,0,2]
            df.loc[t,'acc_avg_y'] = accuracy2D[i,1,2]
            df.loc[t,'acc_avg'] = accuracy1D[i,2]
            
            df.loc[t,'rms_left_x'] = RMS2D[i,0,0]
            df.loc[t,'rms_left_y'] = RMS2D[i,1,0]
            df.loc[t,'rms_left'] = RMS1D[i,0]
            df.loc[t,'rms_right_x'] = RMS2D[i,0,1]
            df.loc[t,'rms_right_y'] = RMS2D[i,1,1]
            df.loc[t,'rms_right'] = RMS1D[i,1]
            df.loc[t,'rms_avg_x'] = RMS2D[i,0,2]
            df.loc[t,'rms_avg_y'] = RMS2D[i,1,2]
            df.loc[t,'rms_avg'] = RMS1D[i,2]
            
            df.loc[t,'std_left_x'] = STD2D[i,0,0]
            df.loc[t,'std_left_y'] = STD2D[i,1,0]
            df.loc[t,'std_left'] = STD1D[i,0]
            df.loc[t,'std_right_x'] = STD2D[i,0,1]
            df.loc[t,'std_right_y'] = STD2D[i,1,1]
            df.loc[t,'std_right'] = STD1D[i,1]
            df.loc[t,'std_avg_x'] = STD2D[i,0,2]
            df.loc[t,'std_avg_y'] = STD2D[i,1,2]
            df.loc[t,'std_avg'] = STD1D[i,2]
            
            df.loc[t,'dataLoss_left'] = dataLoss[i,0]
            df.loc[t,'dataLoss_right'] = dataLoss[i,1]
            df.loc[t,'dataLoss_avg'] = dataLoss[i,2]

        df.to_csv(str(inputDir / 'dataQuality.tsv'), mode='w' if ival==0 else 'a', header=ival==0, sep='\t', na_rep='nan', float_format="%.3f")

if __name__ == '__main__':
    basePath = Path(__file__).resolve().parent
    for d in (basePath / 'data' / 'preprocced').iterdir():
        if d.is_dir():
            process(d,basePath)
