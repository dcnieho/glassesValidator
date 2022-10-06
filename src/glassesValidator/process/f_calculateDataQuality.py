#!/usr/bin/python

import pathlib

import numpy as np
import pandas as pd
import warnings

from .. import config, utils


def process(inputDir, configDir=None):
    from . import DataQualityType
    inputDir  = pathlib.Path(inputDir)
    if configDir is not None:
        configDir = pathlib.Path(configDir)

    print('processing: {}'.format(inputDir.name))
    utils.update_recording_status(inputDir, utils.Task.Data_Quality_Calculated, utils.Status.Running)
    
    # open file with information about Aruco marker and Gaze target locations
    validationSetup = config.getValidationSetup(configDir)

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
    offset = pd.read_csv(str(fileName), delimiter='\t',index_col=['marker_interval','timestamp','type','target'])
    # change type index into enum
    offset.index = offset.index.set_levels(pd.CategoricalIndex([getattr(DataQualityType,x) for x in offset.index.levels[offset.index.names.index('type')]]),level='type')

    # check what we have to process
    typeIdx = np.argwhere([x=='type' for x in offset.index.names]).flatten()
    todo = [x for x in DataQualityType if x in offset.index.levels[typeIdx[0]]]
    if (DataQualityType.pose_left_eye in todo) and (DataQualityType.pose_right_eye in todo):
        todo.append(DataQualityType.pose_left_right_avg)
        
    # prep output data frame
    idx  = []
    idxs = analysisIntervals.index.to_frame().to_numpy()
    for e in todo:
        idx.append(np.vstack((idxs[:,0],idxs.shape[0]*[e],idxs[:,1])).T)
    idx = pd.DataFrame(np.vstack(tuple(idx)),columns=[analysisIntervals.index.names[0],'type',analysisIntervals.index.names[1]])
    df  = pd.DataFrame(index=pd.MultiIndex.from_frame(idx.astype({analysisIntervals.index.names[0]: 'int64','type': 'category', analysisIntervals.index.names[1]: 'int64'})))
    idx = pd.IndexSlice
    ts  = offset.index.get_level_values('timestamp')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # ignore warnings from np.nanmean and np.nanstd
        for i in analysisIntervals.index.levels[0]:
            # determine order in which targets were looked at
            for e in todo:
                df.loc[idx[i,e,:],'order'] = np.argsort(analysisIntervals.loc(axis=0)[i,:]['start_timestamp']).to_numpy()+1

            # compute data quality for each eye
            for t in analysisIntervals.index.levels[1]:
                if (i,t) not in analysisIntervals.index:
                    continue
                st = analysisIntervals.loc[(i,t),'start_timestamp']
                et = analysisIntervals.loc[(i,t),  'end_timestamp']
                qData= np.logical_and(ts>=st, ts<=et)

                # per type (e.g. eye, using pose or viewing distance)
                for e in todo:
                    if e==DataQualityType.pose_left_right_avg:
                        # binocular average
                        data = offset.loc[idx[i,qData,[DataQualityType.pose_left_eye,DataQualityType.pose_right_eye],t],:].mean(level=['marker_interval','timestamp','target'],skipna=True)
                    else:
                        data = offset.loc[idx[i,qData,                                e                             ,t],:]
                    
                    df.loc[(i,e,t),'acc_x'] = np.nanmean(data['offset_x'])
                    df.loc[(i,e,t),'acc_y'] = np.nanmean(data['offset_y'])
                    df.loc[(i,e,t),'acc'  ] = np.nanmean(np.hypot(data['offset_x'],data['offset_y']))
                    
                    df.loc[(i,e,t),'rms_x'] = np.sqrt(np.nanmean(np.diff(data['offset_x'])**2))
                    df.loc[(i,e,t),'rms_y'] = np.sqrt(np.nanmean(np.diff(data['offset_y'])**2))
                    df.loc[(i,e,t),'rms'  ] = np.hypot(df.loc[(i,e,t),'rms_x'], df.loc[(i,e,t),'rms_y'])
                    
                    df.loc[(i,e,t),'std_x'] = np.nanstd(data['offset_x'],ddof=1)
                    df.loc[(i,e,t),'std_y'] = np.nanstd(data['offset_y'],ddof=1)
                    df.loc[(i,e,t),'std'  ] = np.hypot(df.loc[(i,e,t),'std_x'], df.loc[(i,e,t),'std_y'])

                    df.loc[(i,e,t),'dataLoss'] = np.sum(np.isnan(data['offset_x']))/len(data)
    

    df.to_csv(str(inputDir / 'dataQuality.tsv'), mode='w', header=True, sep='\t', na_rep='nan', float_format="%.3f")
    
    utils.update_recording_status(inputDir, utils.Task.Data_Quality_Calculated, utils.Status.Finished)