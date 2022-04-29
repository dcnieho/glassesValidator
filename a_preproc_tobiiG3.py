"""
Cast raw Tobii data into common format.

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
import gzip
import cv2
import pandas as pd
import numpy as np
import math
import mp4analyser.iso

import utils


def preprocessData(inputDir, outputDir):
    """
    Run all preprocessing steps on tobii data
    """
    print('processing: {}'.format(inputDir.name))
    ### copy the raw data to the output directory
    print('Copying raw data...')
    newDataDir = copyTobiiRecording(inputDir, outputDir)
    print('Input data copied to: {}'.format(newDataDir))

    #### prep the copied data...
    print('Getting camera calibration...')
    sceneVideoDimensions = getCameraFromJson(newDataDir)
    print('Prepping gaze data...')
    gazeDf, frameTimestamps = formatGazeData(newDataDir, sceneVideoDimensions)

    # write the gaze data to a csv file
    gazeDf.to_csv(str(newDataDir / 'gazeData.tsv'), sep='\t', na_rep='nan', float_format="%.8f")

    # also store frame timestamps
    frameTimestamps.to_csv(str(newDataDir / 'frameTimestamps.tsv'), sep='\t')

    ### cleanup
    for f in ['livedata.json', 'et.tslv']:
        (newDataDir / f).unlink(missing_ok=True)


def copyTobiiRecording(inputDir, outputDir):
    """
    Copy the relevant files from the specified input dir to the specified output dir
    """
    with open(str(inputDir / 'recording.g3'), 'rb') as j:
        rInfo = json.load(j)
    recording = rInfo['name']
    with open(str(inputDir / rInfo['meta-folder'] / 'participant'), 'rb') as j:
        pInfo = json.load(j)
    participant = pInfo['name']

    outputDir = outputDir / ('tobiiG3_%s_%s' % (participant,recording))
    if not outputDir.is_dir():
        outputDir.mkdir()

    # Copy relevent files to new directory
    for f in [('recording.g3',None), (rInfo['gaze']['file'],'gazedata.gz'), ('scenevideo.mp4','worldCamera.mp4')]:
        outFileName = f[0]
        if f[1] is not None:
            outFileName = f[1]
        shutil.copyfile(str(inputDir / f[0]), str(outputDir / outFileName))

    # Unzip the gaze data file
    for f in ['gazedata.gz']:
        with gzip.open(str(outputDir / f)) as zipFile:
            with open(str(outputDir / Path(f).stem), 'wb') as unzippedFile:
                for line in zipFile:
                    unzippedFile.write(line)
        (outputDir / f).unlink(missing_ok=True)

    # return the full path to the output dir
    return outputDir

def getCameraFromJson(inputDir):
    """
    Read camera calibration from recording information file
    """
    with open(str(inputDir / 'recording.g3'), 'rb') as f:
        rInfo = json.load(f)
    
    camera = rInfo['scenecamera']['camera-calibration']

    # rename some fields, ensure they are numpy arrays
    camera['focalLength'] = np.array(camera.pop('focal-length'))
    camera['principalPoint'] = np.array(camera.pop('principal-point'))
    camera['radialDistortion'] = np.array(camera.pop('radial-distortion'))
    camera['tangentialDistortion'] = np.array(camera.pop('tangential-distortion'))
    
    camera['position'] = np.array(camera['position'])
    camera['resolution'] = np.array(camera['resolution'])
    camera['rotation'] = np.array(camera['rotation'])

    # turn into camera matrix and distortion coefficients as used by OpenCV
    camera['cameraMatrix'] = np.identity(3)
    camera['cameraMatrix'][0,0] = camera['focalLength'][0]
    camera['cameraMatrix'][0,1] = camera['skew']
    camera['cameraMatrix'][1,1] = camera['focalLength'][1]
    camera['cameraMatrix'][0,2] = camera['principalPoint'][0]
    camera['cameraMatrix'][1,2] = camera['principalPoint'][1]

    camera['distCoeff'] = np.zeros(5)
    camera['distCoeff'][:2]  = camera['radialDistortion'][:2]
    camera['distCoeff'][2:4] = camera['tangentialDistortion']
    camera['distCoeff'][4]   = camera['radialDistortion'][2]


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
    df = json2df(inputDir / 'gazedata', sceneVideoDimensions)

    # read video file, create array of frame timestamps
    frameTimestamps = utils.getVidFrameTimestamps(inputDir / 'worldCamera.mp4')

    # use the frame timestamps to assign a frame number to each data point
    df.insert(0,'frame_idx',np.int64(0))
    for ts in df.index:

        # get index where this ts would be inserted into the frame_timestamp array
        idx = np.searchsorted(frameTimestamps['timestamp'], ts)
        if idx == 0:
            df.loc[ts, 'frame_idx'] = math.nan
            continue

        # since idx points to frame timestamp for frame after the one during
        # which the ts ocurred, correct
        idx -= 1

        # set the frame index based on this index value
        df.loc[ts, 'frame_idx'] = frameTimestamps.index[idx]

    # build the formatted dataframe
    df.index.name = 'timestamp'

    # return the gaze data df and frame time stamps array
    return df, frameTimestamps


def json2df(jsonFile,sceneVideoDimensions):
    """
    convert the livedata.json file to a pandas dataframe
    """

    with open(str(jsonFile), 'r') as file:
        entries = json.loads('[' + file.read().replace('\n', ',')[:-1] + ']')

    # json no longer needed, remove
    jsonFile.unlink(missing_ok=True)


    # turn gaze data into data frame
    dfR = pd.json_normalize(entries)
    # convert timestamps from s to ms and set as index
    dfR.loc[:,'timestamp'] *= 1000.0
    dfR = dfR.set_index('timestamp')
    # drop anything thats not gaze
    dfR = dfR.drop(dfR[dfR.type != 'gaze'].index)
    # manipulate data frame to expand columns as needed
    df = pd.DataFrame([],index=dfR.index)
    expander = lambda a,n: [[math.nan]*n if not isinstance(x,list) else x for x in a]
    # monocular gaze data
    for eye in ('left','right'):
        which_eye = eye[:1]
        df[[which_eye + '_gaze_ori_x', which_eye + '_gaze_ori_y', which_eye + '_gaze_ori_z']] = \
            pd.DataFrame(expander(dfR['data.eye'+eye+'.gazeorigin'].tolist(),3), index=dfR.index)
        df[which_eye + '_pup_diam'] = dfR['data.eye'+eye+'.pupildiameter']
        df[[which_eye + '_gaze_dir_x', which_eye + '_gaze_dir_y', which_eye + '_gaze_dir_z']] = \
            pd.DataFrame(expander(dfR['data.eye'+eye+'.gazedirection'].tolist(),3), index=dfR.index)
    
    # binocular gaze data
    df[['3d_gaze_pos_x', '3d_gaze_pos_y', '3d_gaze_pos_z']] = pd.DataFrame(expander(dfR['data.gaze3d'].tolist(),3), index=dfR.index)
    df[['vid_gaze_pos_x', 'vid_gaze_pos_y']] = pd.DataFrame(expander(dfR['data.gaze2d'].tolist(),2), index=dfR.index)
    df.loc[:,'vid_gaze_pos_x'] *= sceneVideoDimensions[0]
    df.loc[:,'vid_gaze_pos_y'] *= sceneVideoDimensions[1]

    # return the dataframe
    return df


if __name__ == '__main__':
    basePath = Path(__file__).resolve().parent / 'data'
    inBasePath = basePath / 'tobiiG3'
    outBasePath = basePath / 'preprocced'
    if not outBasePath.is_dir():
        outBasePath.mkdir()
    for d in inBasePath.iterdir():
        if d.is_dir():
            preprocessData(d,outBasePath)
