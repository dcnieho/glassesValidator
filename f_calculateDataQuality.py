#!/usr/bin/python

import sys
from pathlib import Path
import math

import cv2
import numpy as np
import pandas as pd
import warnings

import utils



def process(inputDir,basePath):
    print('processing: {}'.format(inputDir.name))
    
    configDir = basePath / "config"
    # open file with information about Aruco marker and Gaze target locations
    validationSetup = utils.getValidationSetup(configDir)

    # get time intervals to use for each target
    fileName = inputDir / "analysisInterval.tsv"
    if not fileName.is_file():
        print('  no analysis intervals defined for this recording, skipping')
        return
    analysisIntervals = pd.read_csv(str(fileName), delimiter='\t', dtype={'marker_interval':int},index_col=['marker_interval','target'])
    
    # get offsets
    fileName = inputDir / "gazeTargetOffset.tsv"
    if not fileName.is_file():
        print('  no gaze offsets precomputed defined for this recording, skipping')
        return
    offset = pd.read_csv(str(fileName), delimiter='\t',index_col=['marker_interval','timestamp','eye','target'])

    # prep output data frame
    df  = pd.DataFrame().reindex(index=analysisIntervals.index)
    idx = pd.IndexSlice
    ts  = offset.index.get_level_values('timestamp')
    qHasLeft        = ('left'  in offset.index.levels[2]) and np.any(np.logical_not(np.isnan(offset.loc[idx[:,:,'left'      ,:],:].to_numpy())))
    qHasRight       = ('right' in offset.index.levels[2]) and np.any(np.logical_not(np.isnan(offset.loc[idx[:,:,'right'     ,:],:].to_numpy())))
    qHasRay         = ('ray'   in offset.index.levels[2]) and np.any(np.logical_not(np.isnan(offset.loc[idx[:,:,'ray'       ,:],:].to_numpy())))
    qHasHomography  = ('ray'   in offset.index.levels[2]) and np.any(np.logical_not(np.isnan(offset.loc[idx[:,:,'homography',:],:].to_numpy())))
    todo = []
    if qHasHomography:
        todo.append('homography')
    if qHasRay:
        todo.append('ray')
    if qHasLeft:
        todo.append('left')
    if qHasRight:
        todo.append('right')
    if qHasLeft and qHasRight:
        todo.append('avg')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # ignore warnings from np.nanmean and np.nanstd
        for i in analysisIntervals.index.levels[0]:
            # determine order in which targets were looked at
            df.loc[i,'order'] = np.argsort(analysisIntervals.loc(axis=0)[1,:]['start_timestamp'])+1
        
            for t in analysisIntervals.index.levels[1]:
                st = analysisIntervals.loc[(i,t),'start_timestamp']
                et = analysisIntervals.loc[(i,t),  'end_timestamp']
                qData= np.logical_and(ts>=st, ts<=et)

                # per eye
                for e in todo:
                    if e in ('left','right','ray','homography'):
                        data = offset.loc[idx[i,qData,        e       ,t],:]
                    else:
                        # binocular average
                        data = offset.loc[idx[i,qData,['left','right'],t],:].mean(level=['marker_interval','timestamp','target'],skipna=True)
                    
                    df.loc[(i,t),'acc_'+e+'_x'] = np.nanmean(data['offset_x'])
                    df.loc[(i,t),'acc_'+e+'_y'] = np.nanmean(data['offset_y'])
                    df.loc[(i,t),'acc_'+e     ] = np.nanmean(np.hypot(data['offset_x'],data['offset_y']))
                    
                    df.loc[(i,t),'rms_'+e+'_x'] = np.sqrt(np.nanmean(np.diff(data['offset_x'])**2))
                    df.loc[(i,t),'rms_'+e+'_y'] = np.sqrt(np.nanmean(np.diff(data['offset_y'])**2))
                    df.loc[(i,t),'rms_'+e     ] = np.hypot(df.loc[(i,t),'rms_'+e+'_x'], df.loc[(i,t),'rms_'+e+'_y'])
                    
                    df.loc[(i,t),'std_'+e+'_x'] = np.nanstd(data['offset_x'],ddof=1)
                    df.loc[(i,t),'std_'+e+'_y'] = np.nanstd(data['offset_y'],ddof=1)
                    df.loc[(i,t),'std_'+e     ] = np.hypot(df.loc[(i,t),'std_'+e+'_x'], df.loc[(i,t),'std_'+e+'_y'])

                    df.loc[(i,t),'dataLoss_'+e] = np.sum(np.isnan(data['offset_x']))/len(data)
    

    df.to_csv(str(inputDir / 'dataQuality.tsv'), mode='w', header=True, sep='\t', na_rep='nan', float_format="%.3f")

if __name__ == '__main__':
    basePath = Path(__file__).resolve().parent
    for d in (basePath / 'data' / 'preprocced').iterdir():
        if d.is_dir():
            process(d,basePath)
