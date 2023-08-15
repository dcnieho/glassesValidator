"""
Cast raw AdHawk MindLink data into common format.
"""

import shutil
import pathlib
import json
import csv
import cv2
import pandas as pd
import numpy as np
import math
import datetime

from .. import utils


def preprocessData(output_dir, source_dir=None, rec_info=None):
    """
    Run all preprocessing steps on tobii data
    """
    output_dir = pathlib.Path(output_dir)
    if source_dir is not None:
        source_dir  = pathlib.Path(source_dir)
        if rec_info is not None and pathlib.Path(rec_info.source_directory) != source_dir:
            raise ValueError(f"The provided source_dir ({source_dir}) does not equal the source directory set in rec_info ({rec_info.source_directory}).")
    elif rec_info is None:
        raise RuntimeError('Either the "input_dir" or the "rec_info" input argument should be set.')
    else:
        source_dir  = pathlib.Path(rec_info.source_directory)

    if rec_info is not None:
        if rec_info.eye_tracker!=utils.EyeTracker.AdHawk_MindLink:
            raise ValueError(f'Provided rec_info is for a device ({rec_info.eye_tracker.value}) that is not an {utils.EyeTracker.AdHawk_MindLink.value}. Cannot use.')
        if not rec_info.proc_directory_name:
            rec_info.proc_directory_name = utils.make_fs_dirname(rec_info, output_dir)
        newDir = output_dir / rec_info.proc_directory_name
        if newDir.is_dir():
            raise RuntimeError(f'Output directory specified in rec_info ({rec_info.proc_directory_name}) already exists in the outputDir ({output_dir}). Cannot use.')


    print(f'processing: {source_dir.name}')


    ### check and copy needed files to the output directory
    print('Check and copy raw data...')
    ### check adhawk recording and get export directory
    if rec_info is not None:
        checkRecording(source_dir, rec_info)
    else:
        rec_info = getRecordingInfo(source_dir)
        if rec_info is None:
            raise RuntimeError(f"The folder {source_dir} is not recognized as a {utils.EyeTracker.AdHawk_MindLink.value} recording.")

    # make output dir
    if rec_info.proc_directory_name is None or not rec_info.proc_directory_name:
        rec_info.proc_directory_name = utils.make_fs_dirname(rec_info, output_dir)
    newDataDir = output_dir / rec_info.proc_directory_name
    if not newDataDir.is_dir():
        newDataDir.mkdir()

    # store rec info
    rec_info.store_as_json(newDataDir)

    # make sure there is a processing status file, and update it
    utils.get_recording_status(newDataDir, create_if_missing=True)
    utils.update_recording_status(newDataDir, utils.Task.Imported, utils.Status.Running)


    ### copy the raw data to the output directory
    copyAdhawkRecording(source_dir, newDataDir)
    print(f'Input data copied to: {newDataDir}')

    #### prep the copied data...
    print('Getting camera calibration...')
    sceneVideoDimensions = getCameraFromYaml(source_dir, newDataDir)
    print('Prepping gaze data...')
    gazeDf, frameTimestamps = formatGazeData(source_dir, newDataDir, sceneVideoDimensions)

    # write the gaze data to a csv file
    gazeDf.to_csv(str(newDataDir / 'gazeData.tsv'), sep='\t', na_rep='nan', float_format="%.8f")

    # also store frame timestamps
    frameTimestamps.to_csv(str(newDataDir / 'frameTimestamps.tsv'), sep='\t')

    # indicate import finished
    utils.update_recording_status(newDataDir, utils.Task.Imported, utils.Status.Finished)


def getRecordingInfo(inputDir):
    # returns None if not a recording directory
    recInfo = utils.Recording(source_directory=inputDir, eye_tracker=utils.EyeTracker.AdHawk_MindLink)

    # get recording info
    recInfo.name = inputDir.name

    file = inputDir / 'meta_data.json'
    if not file.is_file():
        return None
    with open(file, 'rb') as j:
        rInfo = json.load(j)
    recInfo.duration = rInfo['manifest']['recording_length_ms']
    recInfo.participant = rInfo['user_profile']['name']
    # get recording start time by reading UTC time associated with first gaze sample
    gaze_entry = getMetaEntry(inputDir, 'gaze')
    file = inputDir / gaze_entry['file_name']
    with open(file, 'r') as read_obj:
        csv_reader = csv.DictReader(read_obj)
        # Iterate over each row in the csv using reader object
        sample = next(csv_reader)
    time_string = sample['UTC_Time']
    if time_string[-1:]=='Z':
        # change Z suffix (if any) to +00:00 for ISO 8601 format that datetime understands
        time_string = time_string[:-1]+'+00:00'
    recInfo.start_time = utils.Timestamp(int(datetime.datetime.fromisoformat(time_string).timestamp()))

    # we got a valid recording and at least some info if we got here
    # return what we've got
    return recInfo

def getMeta(inputDir, key=None):
    file = inputDir / 'meta_data.json'
    if not file.is_file():
        return None
    with open(file, 'rb') as j:
        rInfo = json.load(j)

    if key:
        return rInfo[key]
    else:
        return rInfo

def getMetaEntry(inputDir, entry_name):
    manifest = getMeta(inputDir, key='manifest')
    # get gaze file
    entry = None
    for e in manifest['entries']:
        if e['type'].lower()==entry_name:
            entry = e
            break
    return entry

def checkRecording(inputDir, recInfo):
    actualRecInfo = getRecordingInfo(inputDir)
    if actualRecInfo is None or recInfo.name!=actualRecInfo.name:
        raise ValueError(f"A recording with the name \"{recInfo.name}\" was not found in the folder {inputDir}.")

    # make sure caller did not mess with recInfo
    if recInfo.participant!=actualRecInfo.participant:
        raise ValueError(f"A recording with the participant \"{recInfo.participant}\" was not found in the folder {inputDir}.")
    if recInfo.duration!=actualRecInfo.duration:
        raise ValueError(f"A recording with the duration \"{recInfo.duration}\" was not found in the folder {inputDir}.")
    if recInfo.start_time.value!=actualRecInfo.start_time.value:
        raise ValueError(f"A recording with the start_time \"{recInfo.start_time.display}\" was not found in the folder {inputDir}.")


def copyAdhawkRecording(inputDir, outputDir):
    """
    Copy the relevant files from the specified input dir to the specified output dir
    """
    # figure out what the video file is called
    vid_entry = getMetaEntry(inputDir, 'video')

    # Copy relevent files to new directory
    shutil.copyfile(str(inputDir / vid_entry['file_name']), str(outputDir / 'worldCamera.mp4'))

def getCameraFromYaml(inputDir, outputDir):
    """
    Get camera calibration
    Hardcoded as per info received from AdHawk
    """
    # turn into camera matrix and distortion coefficients as used by OpenCV
    camera = {}
    camera['cameraMatrix'] = np.array([[8.6175611023603130e+02, 0.                    , 6.4220317156609758e+02],
                                       [0.                    , 8.6411314484767183e+02, 3.4611059418088462e+02],
                                       [0.                    , 0.                    , 1.                    ]])
    camera['distCoeff'] = np.array([6.4704736326069179e-01, 6.9842325204621162e+01, -3.8446374749176787e-03, -6.5685769622407693e-03, 3.3239962207009803e+01, 5.0824354805695138e-01, 6.9018441628550974e+01, 3.1191976852198923e+01])
    camera['resolution'] = np.array([1280, 720])
    camera['position'] = np.array([-0.0685, 0.0152028, 0.00340752])*1000  # our positions are in mm, not m
    camera['rotation'] = cv2.Rodrigues(np.radians(np.array([12.000000000000043, 0.0, 0.0])))[0]

    # store to file
    fs = cv2.FileStorage(str(outputDir / 'calibration.xml'), cv2.FILE_STORAGE_WRITE)
    for key,value in camera.items():
        fs.write(name=key,val=value)
    fs.release()

    return camera['resolution']


def formatGazeData(inputDir, outputDir, sceneVideoDimensions):
    """
    load gazedata json file
    format to get the gaze coordinates w/r/t world camera, and timestamps for every frame of video

    Returns:
        - formatted dataframe with cols for timestamp, frame_idx, and gaze data
        - np array of frame timestamps
    """

    # convert the json file to pandas dataframe
    df = csv2df(inputDir, sceneVideoDimensions)

    # read video file, create array of frame timestamps
    frameTimestamps = utils.getFrameTimestampsFromVideo(outputDir / 'worldCamera.mp4')

    # return the gaze data df and frame time stamps array
    return df, frameTimestamps


def csv2df(inputDir, sceneVideoDimensions):
    """
    convert the gaze_data.csv file to a pandas dataframe
    """

    vid_entry = getMetaEntry(inputDir, 'video')
    gaze_entry = getMetaEntry(inputDir, 'gaze')

    file = inputDir / gaze_entry['file_name']
    df = pd.read_csv(file)

    # prepare data frame
    # remove unneeded columns
    df=df.drop(columns=['Screen_X', 'Screen_Y', 'UTC_Time', 'Image_One_Degree_X', 'Image_One_Degree_Y'],errors='ignore') # drop these columns if they exist

    # rename and reorder columns
    lookup = {'Timestamp': 'timestamp',
               'Frame_Index':'frame_idx',
               'Image_X':'vid_gaze_pos_x',
               'Image_Y':'vid_gaze_pos_y',
               'Gaze_X_Left':'l_gaze_dir_x',
               'Gaze_Y_Left':'l_gaze_dir_y',
               'Gaze_Z_Left':'l_gaze_dir_z',
               'Gaze_X_Right':'r_gaze_dir_x',
               'Gaze_Y_Right':'r_gaze_dir_y',
               'Gaze_Z_Right':'r_gaze_dir_z',
               'Gaze_X':'3d_gaze_pos_x',
               'Gaze_Y':'3d_gaze_pos_y',
               'Gaze_Z':'3d_gaze_pos_z',}
    df=df.rename(columns=lookup)
    # reorder
    idx = [lookup[k] for k in lookup if lookup[k] in df.columns]
    idx.extend([x for x in df.columns if x not in idx])   # append columns not in lookup
    df = df[idx]

    # get gaze vector origins
    pp_entry = getMetaEntry(inputDir, 'pupil_position')
    file = inputDir / pp_entry['file_name']
    dfP = pd.read_csv(file)
    dfP = dfP.drop(columns=['UTC_Time'],errors='ignore') # drop these columns if they exist
    # rename and reorder columns
    lookup = {'Timestamp': 'timestamp',
               'Pupil_Pos_X_Left':'l_gaze_ori_x',
               'Pupil_Pos_Y_Left':'l_gaze_ori_y',
               'Pupil_Pos_Z_Left':'l_gaze_ori_z',
               'Pupil_Pos_X_Right':'r_gaze_ori_x',
               'Pupil_Pos_Y_Right':'r_gaze_ori_y',
               'Pupil_Pos_Z_Right':'r_gaze_ori_z',}
    dfP = dfP.rename(columns=lookup)
    # reorder
    idx = [lookup[k] for k in lookup if lookup[k] in dfP.columns]
    idx.extend([x for x in dfP.columns if x not in idx])   # append columns not in lookup
    dfP = dfP[idx]

    # merge
    df = pd.merge(df, dfP, on='timestamp')

    # convert timestamps from s to ms and set as index
    df.loc[:,'timestamp'] *= 1000.0
    # set first gaze timestamp to 0 and express gaze timestamps in video time
    df.loc[:,'timestamp'] -= (df.loc[0,'timestamp'] - (gaze_entry['attribute']['start_time_ms'] - vid_entry['attribute']['start_time_ms']))
    # remove data with negative timestamps
    df = df[df.timestamp >= 0]
    df = df.set_index('timestamp')

    # binocular gaze data
    df.loc[:,'vid_gaze_pos_x'] *= sceneVideoDimensions[0]
    df.loc[:,'vid_gaze_pos_y'] *= sceneVideoDimensions[1]

    # adhawk positive z is backward, ours is forward
    df.loc[:,'l_gaze_ori_z'] = -df.loc[:,'l_gaze_ori_z']
    df.loc[:,'l_gaze_dir_z'] = -df.loc[:,'l_gaze_dir_z']
    df.loc[:,'r_gaze_ori_z'] = -df.loc[:,'r_gaze_ori_z']
    df.loc[:,'r_gaze_dir_z'] = -df.loc[:,'r_gaze_dir_z']
    df.loc[:,'3d_gaze_pos_z'] = -df.loc[:,'3d_gaze_pos_z']

    # adhawk positive y is upward, ours is downward
    df.loc[:,'l_gaze_ori_y'] = -df.loc[:,'l_gaze_ori_y']
    df.loc[:,'l_gaze_dir_y'] = -df.loc[:,'l_gaze_dir_y']
    df.loc[:,'r_gaze_ori_y'] = -df.loc[:,'r_gaze_ori_y']
    df.loc[:,'r_gaze_dir_y'] = -df.loc[:,'r_gaze_dir_y']
    df.loc[:,'3d_gaze_pos_y'] = -df.loc[:,'3d_gaze_pos_y']

    # adhawk gaze pos is in m, ours is in mm
    # NB: gaze ori is in mm!
    df.loc[:,'3d_gaze_pos_x'] *= 1000
    df.loc[:,'3d_gaze_pos_y'] *= 1000
    df.loc[:,'3d_gaze_pos_z'] *= 1000

    # return the dataframe
    return df