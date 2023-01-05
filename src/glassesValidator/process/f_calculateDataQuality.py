#!/usr/bin/python

import pathlib

import numpy as np
import pandas as pd
import warnings

from .. import utils


def process(working_dir, dq_types=[], allow_dq_fallback=False, include_data_loss=False):
    from . import DataQualityType
    working_dir  = pathlib.Path(working_dir)

    print('processing: {}'.format(working_dir.name))
    utils.update_recording_status(working_dir, utils.Task.Data_Quality_Calculated, utils.Status.Running)

    # get time intervals to use for each target
    fileName = working_dir / "analysisInterval.tsv"
    if not fileName.is_file():
        print('  no analysis intervals defined for this recording, skipping')
        return
    analysisIntervals = pd.read_csv(str(fileName), delimiter='\t', dtype={'marker_interval':int},index_col=['marker_interval','target'])

    # get offsets
    fileName = working_dir / "gazeTargetOffset.tsv"
    if not fileName.is_file():
        print('  no gaze offsets precomputed defined for this recording, skipping')
        return
    offset = pd.read_csv(str(fileName), delimiter='\t',index_col=['marker_interval','timestamp','type','target'])
    # change type index into enum
    typeIdx = offset.index.names.index('type')
    offset.index = offset.index.set_levels(pd.CategoricalIndex([getattr(DataQualityType,x) for x in offset.index.levels[typeIdx]]),level='type')

    # check what we have to process. go with good defaults
    dq_have = list(offset.index.levels[typeIdx])
    if (DataQualityType.pose_left_eye in dq_have) and (DataQualityType.pose_right_eye in dq_have):
        dq_have.append(DataQualityType.pose_left_right_avg)
    if dq_types:
        if not isinstance(dq_types,list):
            dq_types = [dq_types]
        # do some checks on user input
        for i,dq in reversed(list(enumerate(dq_types))):
            if not isinstance(dq, DataQualityType):
                if isinstance(dq, str):
                    if hasattr(DataQualityType, dq):
                        dq = dq_types[i] = getattr(DataQualityType, dq)
                    else:
                        raise ValueError(f"The string '{dq}' is not a known data quality type. Known types: {[e.name for e in DataQualityType]}")
                else:
                    raise ValueError(f"The variable 'dq' should be a string with one of the following values: {[e.name for e in DataQualityType]}")
            if not dq in dq_have:
                if allow_dq_fallback:
                    del dq_types[i]
                else:
                    raise RuntimeError(f'Data quality type {dq} could not be used as its not available for this recording. Available data quality types: {[e.name for e in dq_have]}')

        if DataQualityType.pose_left_right_avg in dq_types:
            if (not DataQualityType.pose_left_eye in dq_have) or (not DataQualityType.pose_right_eye in dq_have):
                if allow_dq_fallback:
                    dq_types.remove(DataQualityType.pose_left_right_avg)
                else:
                    raise RuntimeError(f'Cannot use the data quality type {DataQualityType.pose_left_right_avg} because it requires having data quality types {DataQualityType.pose_left_eye} and {DataQualityType.pose_right_eye} available, but one or both are not available. Available data quality types: {[e.name for e in dq_have]}')

    if not dq_types:
        if DataQualityType.pose_vidpos_ray in dq_have:
            # highest priority is DataQualityType.pose_vidpos_ray
            dq_types.append(DataQualityType.pose_vidpos_ray)
        elif DataQualityType.pose_vidpos_homography in dq_have:
            # else at least try to use pose (shouldn't occur, if we have pose have a calibrated camera, which means we should have the above)
            dq_types.append(DataQualityType.pose_vidpos_homography)
        else:
            # else we're down to falling back on an assumed viewing distance
            if not DataQualityType.viewpos_vidpos_homography in dq_have:
                raise RuntimeError(f'Even data quality type {DataQualityType.viewpos_vidpos_homography} could not be used, bare minimum failed for some weird reason')
            dq_types.append(DataQualityType.viewpos_vidpos_homography)

    # prep output data frame
    idx  = []
    idxs = analysisIntervals.index.to_frame().to_numpy()
    for e in dq_types:
        idx.append(np.vstack((idxs[:,0],idxs.shape[0]*[e],idxs[:,1])).T)
    idx = pd.DataFrame(np.vstack(tuple(idx)),columns=[analysisIntervals.index.names[0],'type',analysisIntervals.index.names[1]])
    df  = pd.DataFrame(index=pd.MultiIndex.from_frame(idx.astype({analysisIntervals.index.names[0]: 'int64','type': 'category', analysisIntervals.index.names[1]: 'int64'})))
    idx = pd.IndexSlice
    ts  = offset.index.get_level_values('timestamp')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # ignore warnings from np.nanmean and np.nanstd
        for i in analysisIntervals.index.levels[0]:
            # determine order in which targets were looked at
            for e in dq_types:
                df.loc[idx[i,e,:],'order'] = np.argsort(analysisIntervals.loc(axis=0)[i,:]['start_timestamp']).to_numpy()+1

            # compute data quality for each eye
            for t in analysisIntervals.index.levels[1]:
                if (i,t) not in analysisIntervals.index:
                    continue
                st = analysisIntervals.loc[(i,t),'start_timestamp']
                et = analysisIntervals.loc[(i,t),  'end_timestamp']
                qData= np.logical_and(ts>=st, ts<=et)

                # per type (e.g. eye, using pose or viewing distance)
                for e in dq_types:
                    hasData = True
                    try:
                        if e==DataQualityType.pose_left_right_avg:
                            # binocular average
                            data = offset.loc[idx[i,qData,[DataQualityType.pose_left_eye,DataQualityType.pose_right_eye],t],:].mean(level=['marker_interval','timestamp','target'],skipna=True)
                        else:
                            data = offset.loc[idx[i,qData,                                e                             ,t],:]
                    except KeyError:
                        # this happens when data for the given type is not available (e.g. no binocular data, only individual eye data)
                        hasData = False
                        for k in ('acc_x','acc_y','acc','rms_x','rms_y','rms','std_x','std_y','std'):
                            df.loc[(i,e,t),k] = np.nan
                        if include_data_loss:
                            df.loc[(i,e,t),'data_loss'] = np.nan

                    if hasData:
                        df.loc[(i,e,t),'acc_x'] = np.nanmean(data['offset_x'])
                        df.loc[(i,e,t),'acc_y'] = np.nanmean(data['offset_y'])
                        df.loc[(i,e,t),'acc'  ] = np.nanmean(np.hypot(data['offset_x'],data['offset_y']))

                        df.loc[(i,e,t),'rms_x'] = np.sqrt(np.nanmean(np.diff(data['offset_x'])**2))
                        df.loc[(i,e,t),'rms_y'] = np.sqrt(np.nanmean(np.diff(data['offset_y'])**2))
                        df.loc[(i,e,t),'rms'  ] = np.hypot(df.loc[(i,e,t),'rms_x'], df.loc[(i,e,t),'rms_y'])

                        df.loc[(i,e,t),'std_x'] = np.nanstd(data['offset_x'],ddof=1)
                        df.loc[(i,e,t),'std_y'] = np.nanstd(data['offset_y'],ddof=1)
                        df.loc[(i,e,t),'std'  ] = np.hypot(df.loc[(i,e,t),'std_x'], df.loc[(i,e,t),'std_y'])

                        if include_data_loss:
                            df.loc[(i,e,t),'data_loss'] = np.sum(np.isnan(data['offset_x']))/len(data)


    df.to_csv(str(working_dir / 'dataQuality.tsv'), mode='w', header=True, sep='\t', na_rep='nan', float_format="%.3f")

    utils.update_recording_status(working_dir, utils.Task.Data_Quality_Calculated, utils.Status.Finished)