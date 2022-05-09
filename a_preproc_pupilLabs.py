"""
Cast raw pupil labs (core and invisible) data into common format.

Tested with Python 3.8, open CV 4.0.1

The output directory will contain:
    - frameTimestamps.tsv: frame number and corresponding timestamps for each frame in video
    - worldCamera.mp4: the video from the point-of-view scene camera on the glasses
    - gazeData.tsv: gaze data, where all 2D gaze coordinates are represented w/r/t the world camera,
                    and 3D coordinates in the coordinate frame of the glasses (may need rotation/translation
                    to represent in world camera's coordinate frame)
"""

import shutil
from pathlib import Path
import json
import cv2
import pandas as pd
import numpy as np
import math
import msgpack
import re


import utils


def preprocessData(inputDir, outputDir, device):
    """
    Run all preprocessing steps on tobii data
    """
    print('processing: {}'.format(inputDir.name))
    ### copy the raw data to the output directory
    print('Copying raw data...')
    newDataDir = copyPupilRecording(inputDir, outputDir, device)
    print('Input data copied to: {}'.format(newDataDir))

    #### prep the copied data...
    print('Getting camera calibration...')
    sceneVideoDimensions = getCameraFromMsgPack(newDataDir)
    print('Prepping gaze data...')
    gazeDf, frameTimestamps = formatGazeData(newDataDir, sceneVideoDimensions)

    # write the gaze data to a csv file
    gazeDf.to_csv(str(newDataDir / 'gazeData.tsv'), sep='\t', na_rep='nan', float_format="%.8f")

    # also store frame timestamps
    frameTimestamps.to_csv(str(newDataDir / 'frameTimestamps.tsv'), sep='\t')

    ### cleanup
    for f in ['world.intrinsics', 'world_timestamps.npy']:
        (newDataDir / f).unlink(missing_ok=True)


def copyPupilRecording(inputDir, outputDir, device):
    """
    Copy the relevant files from the specified input dir (highest number export) to the specified output dir
    """
    
    # check we have an export in the input dir
    inputExpDir = inputDir / 'exports'
    if not inputExpDir.is_dir():
        raise RuntimeError('no exports folder for {}'.format(inputDir))

    # get latest export in that folder
    folds = sorted([f for f in inputExpDir.glob('*') if f.is_dir()])
    if not folds:
        raise RuntimeError('there are no exports in the folder {}'.format(inputExpDir))
    inputExpDir = folds[-1]

    # make output dir
    outputDir = outputDir / ('%s_%s' % (device, inputDir.stem))
    if not outputDir.is_dir():
        outputDir.mkdir()

    # Copy relevent files to new directory
    for f in [(inputExpDir / 'gaze_positions.csv',None), (inputDir / 'world_timestamps.npy',None), (inputDir / 'world.intrinsics',None), (inputDir / 'world.mp4','worldCamera.mp4')]:
        outFileName = f[0].name
        if f[1] is not None:
            outFileName = f[1]
        shutil.copyfile(str(f[0]), str(outputDir / outFileName))

    # return the full path to the output dir
    return outputDir

def getCameraFromMsgPack(inputDir):
    """
    Read camera calibration from recording information file
    """
    with open(inputDir / 'world.intrinsics', 'rb') as f:
        camInfo = msgpack.unpack(f)
    
    # get keys which denote a camera resolution
    rex = re.compile('^\(\d+, \d+\)$')

    keys = [k for k in camInfo if rex.match(k)]
    if len(keys)!=1:
        raise RuntimeError('No camera intrinsics or intrinsics for more than one camera found')
    camera = camInfo[keys[0]]

    # rename some fields, ensure they are numpy arrays
    camera['cameraMatrix'] = np.array(camera.pop('camera_matrix'))
    camera['distCoeff']    = np.array(camera.pop('dist_coefs')).flatten()
    camera['resolution']   = np.array(camera['resolution'])

    # store to file
    fs = cv2.FileStorage(str(inputDir / 'calibration.xml'), cv2.FILE_STORAGE_WRITE)
    for key,value in camera.items():
        fs.write(name=key,val=value)
    fs.release()

    # json no longer needed, remove
    (inputDir / 'recording.g3').unlink(missing_ok=True)

    return camera['resolution']


def formatGazeData(inputDir, sceneVideoDimensions):
    """
    load gazedata json file
    format to get the gaze coordinates w/r/t world camera, and timestamps for every frame of video

    Returns:
        - formatted dataframe with cols for timestamp, frame_idx, and gaze data
        - np array of frame timestamps
    """

    # convert the json file to pandas dataframe
    df = readGazeData(inputDir / 'gaze_positions.csv', sceneVideoDimensions)

    # read video file, create array of frame timestamps
    frameTimestamps = pd.DataFrame()#utils.getVidFrameTimestamps(inputDir / 'worldCamera.mp4')
    frameTimestamps['timestamp'] = np.load(str(inputDir / 'world_timestamps.npy'))*1000.0
    frameTimestamps.index.name = 'frame_idx'

    # set t=0 to video start time
    t0 = frameTimestamps.iloc[0].to_numpy()
    df.loc[:,'timestamp'] -= t0
    frameTimestamps.loc[:,'timestamp'] -= t0
    
    # set timestamps as index for gaze
    df = df.set_index('timestamp')
    
    # return the gaze data df and frame time stamps array
    return df, frameTimestamps


def readGazeData(file,sceneVideoDimensions):
    """
    convert the livedata.json file to a pandas dataframe
    """

    df = pd.read_csv(file)

    # json no longer needed, remove
    file.unlink(missing_ok=True)

    # drop columns with nothing in them
    df = df.dropna(how='all', axis=1)
    df = df.drop(columns=['base_data'],errors='ignore') # drop these columns if they exist)

    # rename and reorder columns
    lookup = {'gaze_timestamp': 'timestamp',
               'world_index': 'frame_idx',
               'eye_center1_3d_x':'l_gaze_ori_x',
               'eye_center1_3d_y':'l_gaze_ori_y',
               'eye_center1_3d_z':'l_gaze_ori_z',
               'gaze_normal1_x':'l_gaze_dir_x',
               'gaze_normal1_y':'l_gaze_dir_y',
               'gaze_normal1_z':'l_gaze_dir_z',
               'eye_center0_3d_x':'r_gaze_ori_x',   # NB: if monocular setup filming left eye, this is the left eye
               'eye_center0_3d_y':'r_gaze_ori_y',
               'eye_center0_3d_z':'r_gaze_ori_z',
               'gaze_normal0_x':'r_gaze_dir_x',
               'gaze_normal0_y':'r_gaze_dir_y',
               'gaze_normal0_z':'r_gaze_dir_z',
               'norm_pos_x':'vid_gaze_pos_x',
               'norm_pos_y':'vid_gaze_pos_y',
               'gaze_point_3d_x': '3d_gaze_pos_x',
               'gaze_point_3d_y': '3d_gaze_pos_y',
               'gaze_point_3d_z': '3d_gaze_pos_z'}
    df=df.rename(columns=lookup)
    # reorder
    idx = [lookup[k] for k in lookup if lookup[k] in df.columns]
    idx.extend([x for x in df.columns if x not in idx])   # append columns not in lookup
    df = df[idx]

    # convert timestamps from s to ms
    df.loc[:,'timestamp'] *= 1000.0

    # turn gaze locations into pixel data with origin in top-left
    df.loc[:,'vid_gaze_pos_x'] *= sceneVideoDimensions[0]
    df.loc[:,'vid_gaze_pos_y'] = (1-df.loc[:,'vid_gaze_pos_y'])*sceneVideoDimensions[1] # turn origin from bottom-left to top-left

    # return the dataframe
    return df


if __name__ == '__main__':
    basePath = Path(__file__).resolve().parent / 'data'
    for dev in ['pupilCore','pupilInvisible']:
        inBasePath = basePath / dev
        outBasePath = basePath / 'preprocced'
        if not outBasePath.is_dir():
            outBasePath.mkdir()
        for d in inBasePath.iterdir():
            if d.is_dir():
                preprocessData(d,outBasePath,dev)
