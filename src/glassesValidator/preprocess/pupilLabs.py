"""
Cast raw pupil labs (core and invisible) data into common format.

You should first open the recording (whether directly from device or downloaded
from pupil cloud) in Pupil Player, and do an export there (disable the scene video
export, its unneeded and takes lots of space)

The output directory will contain:
    - frameTimestamps.tsv: frame number and corresponding timestamps for each frame in video
    - worldCamera.mp4: the video from the point-of-view scene camera on the glasses
    - gazeData.tsv: gaze data, where all 2D gaze coordinates are represented w/r/t the world camera,
                    and 3D coordinates in the coordinate frame of the glasses (may need rotation/translation
                    to represent in world camera's coordinate frame)
    - calibration.xml: info about the camera intrinsics
"""

import shutil
import pathlib
import json
import cv2
import pandas as pd
import numpy as np
import msgpack
import re
from urllib.request import urlopen


from .. import utils


def preprocessData(output_dir, device=None, input_dir=None, rec_info=None):
    """
    Run all preprocessing steps on pupil data
    """
    output_dir = pathlib.Path(output_dir)
    if input_dir is not None:
        input_dir  = pathlib.Path(input_dir)
        if rec_info is not None and pathlib.Path(rec_info.source_directory) != input_dir:
            raise ValueError(f"The provided source_dir ({input_dir}) does not equal the source directory set in rec_info ({rec_info.source_directory}).")
    elif rec_info is None:
        raise RuntimeError('Either the "input_dir" or the "rec_info" input argument should be set.')
    else:
        input_dir  = pathlib.Path(rec_info.source_directory)

    if rec_info is not None:
        if not rec_info.eye_tracker in [utils.EyeTracker.Pupil_Core, utils.EyeTracker.Pupil_Invisible]:
            raise ValueError(f'Provided rec_info is for a device ({rec_info.eye_tracker.value}) that is not a {utils.EyeTracker.Pupil_Core.value} or a {utils.EyeTracker.Pupil_Invisible.value}. Cannot use.')
        if not rec_info.proc_directory_name:
            rec_info.proc_directory_name = utils.make_fs_dirname(rec_info, output_dir)
        newDir = output_dir / rec_info.proc_directory_name
        if newDir.is_dir():
            raise RuntimeError(f'Output directory specified in rec_info ({rec_info.proc_directory_name}) already exists in the outputDir ({output_dir}). Cannot use.')

    if device is None and rec_info is None:
        raise RuntimeError('Either the "device" or the "rec_info" input argument should be set.')
    if device is not None:
        device = utils.type_string_to_enum(device)
        if not device in [utils.EyeTracker.Pupil_Core, utils.EyeTracker.Pupil_Invisible]:
            raise ValueError(f'Provided device ({rec_info.eye_tracker.value}) is not a {utils.EyeTracker.Pupil_Core.value} or a {utils.EyeTracker.Pupil_Invisible.value}.')
    if rec_info is not None:
        if device is not None:
            if rec_info.eye_tracker != device:
                raise ValueError(f'Provided device ({device.value}) did not match device specific in rec_info ({rec_info.eye_tracker.value}). Provide matching values or do not provide the device input argument.')
        else:
            device = rec_info.eye_tracker

    print(f'processing: {input_dir.name}')


    ### check and copy needed files to the output directory
    print('Check and copy raw data...')
    ### check pupil recording and get export directory
    exportFile = checkPupilRecording(input_dir)
    if rec_info is not None:
        checkRecording(input_dir, rec_info)
    else:
        rec_info = getRecordingInfo(input_dir, device)
        if rec_info is None:
            raise RuntimeError(f"The folder {input_dir} is not recognized as a {device.value} recording.")

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


    # copy world video
    shutil.copyfile(str(input_dir / 'world.mp4'), str(newDataDir / 'worldCamera.mp4'))
    print(f'Input data copied to: {newDataDir}')


    ### get camera cal
    print('Getting camera calibration...')
    match rec_info.eye_tracker:
        case utils.EyeTracker.Pupil_Core:
            sceneVideoDimensions = getCameraFromMsgPack(input_dir, newDataDir)
        case utils.EyeTracker.Pupil_Invisible:
            sceneVideoDimensions = getCameraCalFromOnline(input_dir, newDataDir, rec_info)


    ### get gaze data and video frame timestamps
    print('Prepping gaze data...')
    gazeDf, frameTimestamps = formatGazeData(input_dir, exportFile, sceneVideoDimensions, rec_info)

    # write the gaze data to a csv file
    gazeDf.to_csv(str(newDataDir / 'gazeData.tsv'), sep='\t', na_rep='nan', float_format="%.8f")

    # also store frame timestamps
    frameTimestamps.to_csv(str(newDataDir / 'frameTimestamps.tsv'), sep='\t')

    # indicate import finished
    utils.update_recording_status(newDataDir, utils.Task.Imported, utils.Status.Finished)


def checkPupilRecording(inputDir):
    """
    This checks that the folder is properly prepared
    (i.e. opened in pupil player and an export was run)
    """
    # check we have an info.player.json file
    if not (inputDir / 'info.player.json').is_file():
        raise RuntimeError(f'info.player.json file not found for {inputDir}. Open the recording in Pupil Player before importing into glassesValidator.')

    # check we have an export in the input dir
    inputExpDir = inputDir / 'exports'
    if not inputExpDir.is_dir():
        raise RuntimeError(f'no exports folder for {inputDir}. Perform a recording export in Pupil Player before importing into glassesValidator.')

    # get latest export in that folder that contain a gaze position file
    gpFiles = sorted(list(inputExpDir.rglob('*gaze_positions*.csv')))
    if not gpFiles:
        raise RuntimeError(f'There are no exports in the folder {inputExpDir}. Perform a recording export in Pupil Player before importing into glassesValidator.')

    return gpFiles[-1]


def getRecordingInfo(inputDir, device):
    # returns None if not a recording directory
    recInfo = utils.Recording(source_directory=inputDir, eye_tracker=device)

    match device:
        case utils.EyeTracker.Pupil_Core:
            # check this is not an invisible recording
            file = inputDir / 'info.invisible.json'
            if file.is_file():
                return None

            file = inputDir / 'info.player.json'
            if not file.is_file():
                return None
            with open(file, 'r') as j:
                iInfo = json.load(j)
            recInfo.name = iInfo['recording_name']
            recInfo.start_time = utils.Timestamp(int(iInfo['start_time_system_s'])) # UTC in seconds, keep second part
            recInfo.duration   = int(iInfo['duration_s']*1000)                      # in seconds, convert to ms
            recInfo.recording_software_version = iInfo['recording_software_version']

            # get user name, if any
            user_info_file = inputDir / 'user_info.csv'
            if user_info_file.is_file():
                df = pd.read_csv(user_info_file)
                nameRow = df['key'].str.contains('name')
                if any(nameRow):
                    if not pd.isnull(df[nameRow].value).to_numpy()[0]:
                        recInfo.participant = df.loc[nameRow,'value'].to_numpy()[0]

        case utils.EyeTracker.Pupil_Invisible:
            file = inputDir / 'info.invisible.json'
            if not file.is_file():
                return None
            with open(file, 'r') as j:
                iInfo = json.load(j)
            recInfo.name = iInfo['template_data']['recording_name']
            recInfo.recording_software_version = iInfo['app_version']
            recInfo.start_time = utils.Timestamp(int(iInfo['start_time']//1000000000))  # UTC in nanoseconds, keep second part
            recInfo.duration   = int(iInfo['duration']//1000000)                        # in nanoseconds, convert to ms
            recInfo.glasses_serial = iInfo['glasses_serial_number']
            recInfo.recording_unit_serial = iInfo['android_device_id']
            recInfo.scene_camera_serial = iInfo['scene_camera_serial_number']
            # get participant name
            file = inputDir / 'wearer.json'
            if file.is_file():
                wearer_id = iInfo['wearer_id']
                with open(file, 'r') as j:
                    iInfo = json.load(j)
                if wearer_id==iInfo['uuid']:
                    recInfo.participant = iInfo['name']

        case _:
            print(f"Device {device} unknown")
            return None

    # we got a valid recording and at least some info if we got here
    # return what we've got
    return recInfo


def checkRecording(inputDir, recInfo):
    actualRecInfo = getRecordingInfo(inputDir, recInfo.eye_tracker)
    if actualRecInfo is None or recInfo.name!=actualRecInfo.name:
        raise ValueError(f"A recording with the name \"{recInfo.name}\" was not found in the folder {inputDir}.")

    # make sure caller did not mess with recInfo
    if recInfo.duration!=actualRecInfo.duration:
        raise ValueError(f"A recording with the duration \"{recInfo.duration}\" was not found in the folder {inputDir}.")
    if recInfo.start_time.value!=actualRecInfo.start_time.value:
        raise ValueError(f"A recording with the start_time \"{recInfo.start_time.display}\" was not found in the folder {inputDir}.")
    if recInfo.recording_software_version!=actualRecInfo.recording_software_version:
        raise ValueError(f"A recording with the duration \"{recInfo.duration}\" was not found in the folder {inputDir}.")

    if recInfo.eye_tracker==utils.EyeTracker.Pupil_Invisible:
        # for invisible recordings we have a bit more info
        if recInfo.glasses_serial!=actualRecInfo.glasses_serial:
            raise ValueError(f"A recording with the glasses_serial \"{recInfo.glasses_serial}\" was not found in the folder {inputDir}.")
        if recInfo.recording_unit_serial!=actualRecInfo.recording_unit_serial:
            raise ValueError(f"A recording with the recording_unit_serial \"{recInfo.recording_unit_serial}\" was not found in the folder {inputDir}.")
        if recInfo.scene_camera_serial!=actualRecInfo.scene_camera_serial:
            raise ValueError(f"A recording with the scene_camera_serial \"{recInfo.scene_camera_serial}\" was not found in the folder {inputDir}.")
        if (recInfo.participant is not None or actualRecInfo.participant is not None) and recInfo.participant!=actualRecInfo.participant:
            raise ValueError(f"A recording with the participant \"{recInfo.participant}\" was not found in the folder {inputDir}.")


def getCameraFromMsgPack(inputDir, outputDir):
    """
    Read camera calibration from recording information file
    """
    camera = getCamInfo(inputDir / 'world.intrinsics')

    # rename some fields, ensure they are numpy arrays
    camera['cameraMatrix'] = np.array(camera.pop('camera_matrix'))
    camera['distCoeff']    = np.array(camera.pop('dist_coefs')).flatten()
    camera['resolution']   = np.array(camera['resolution'])

    # store to file
    fs = cv2.FileStorage(str(outputDir / 'calibration.xml'), cv2.FILE_STORAGE_WRITE)
    for key,value in camera.items():
        fs.write(name=key,val=value)
    fs.release()

    return camera['resolution']


def getCameraCalFromOnline(inputDir, outputDir, recInfo):
    """
    Get camera calibration from pupil labs
    """
    url = f'https://api.cloud.pupil-labs.com/v2/hardware/{recInfo.scene_camera_serial}/calibration.v1?json'

    camInfo = json.loads(urlopen(url).read())
    if camInfo['status'] != 'success':
        raise RuntimeError('Camera calibration could not be loaded, response: %s' % (camInfo['message']))

    camInfo = camInfo['result']

    # rename some fields, ensure they are numpy arrays
    camInfo['cameraMatrix'] = np.array(camInfo.pop('camera_matrix'))
    camInfo['distCoeff']    = np.array(camInfo.pop('dist_coefs')).flatten()
    camInfo['rotation']     = np.reshape(np.array(camInfo.pop('rotation_matrix')),(3,3))

    # get resolution from the local intrinsics file
    camInfo['resolution']   = np.array(getCamInfo(inputDir / 'world.intrinsics')['resolution'])

    # store to xml file
    fs = cv2.FileStorage(str(outputDir / 'calibration.xml'), cv2.FILE_STORAGE_WRITE)
    for key,value in camInfo.items():
        fs.write(name=key,val=value)
    fs.release()

    return camInfo['resolution']


def getCamInfo(camInfoFile):
    with open(camInfoFile, 'rb') as f:
        camInfo = msgpack.unpack(f)

    # get keys which denote a camera resolution
    rex = re.compile('^\(\d+, \d+\)$')

    keys = [k for k in camInfo if rex.match(k)]
    if len(keys)!=1:
        raise RuntimeError('No camera intrinsics or intrinsics for more than one camera found')
    return camInfo[keys[0]]


def formatGazeData(inputDir, exportFile, sceneVideoDimensions, recInfo):
    """
    load gazedata json file
    format to get the gaze coordinates w/r/t world camera, and timestamps for every frame of video

    Returns:
        - formatted dataframe with cols for timestamp, frame_idx, and gaze data
        - np array of frame timestamps
    """

    # convert the json file to pandas dataframe
    df = readGazeData(exportFile, sceneVideoDimensions, recInfo)

    # read video file, create array of frame timestamps
    if (inputDir / 'world_lookup.npy').is_file():
        frameTimestamps = pd.DataFrame(np.load(str(inputDir / 'world_lookup.npy')))
        frameTimestamps['timestamp'] *= 1000.0
        frameTimestamps['frame_idx'] = frameTimestamps.index
        frameTimestamps.loc[frameTimestamps['container_idx']==-1,'container_frame_idx'] = -1
        needsAdjust = not frameTimestamps['frame_idx'].equals(frameTimestamps['container_frame_idx'])
        # prep for later clean up
        toDrop = [x for x in frameTimestamps.columns if x not in ['frame_idx','timestamp']]
        # do further adjustment that may be needed
        if needsAdjust:
            # not all video frames were encoded into the video file. Need to adjust
            # frame_idx in the gaze data to match actual video file
            temp = pd.merge(df,frameTimestamps,on='frame_idx')
            temp['frame_idx'] = temp['container_frame_idx']
            temp = temp.rename(columns={'timestamp_x':'timestamp'})
            toDrop.append('timestamp_y')
            df   = temp.drop(columns=toDrop)

        # final setup for output to file
        frameTimestamps['frame_idx'] = frameTimestamps['container_frame_idx']
        frameTimestamps = frameTimestamps.drop(columns=toDrop,errors='ignore')
        frameTimestamps = frameTimestamps[frameTimestamps['frame_idx']!=-1]
        frameTimestamps = frameTimestamps.set_index('frame_idx')
    else:
        frameTimestamps = pd.DataFrame()
        frameTimestamps['timestamp'] = np.load(str(inputDir / 'world_timestamps.npy'))*1000.0
        frameTimestamps.index.name = 'frame_idx'
        # check there are no gaps in the video file
        if df['frame_idx'].max() > frameTimestamps.index.max():
            raise RuntimeError('It appears there are frames missing in the scene video, but the file world_lookup.npy that would be needed to deal with that is missing. You can generate it by opening the recording in pupil player.')

    # set t=0 to video start time
    t0 = frameTimestamps.iloc[0].to_numpy()
    df.loc[:,'timestamp'] -= t0
    frameTimestamps.loc[:,'timestamp'] -= t0

    # set timestamps as index for gaze
    df = df.set_index('timestamp')

    # return the gaze data df and frame time stamps array
    return df, frameTimestamps


def readGazeData(file,sceneVideoDimensions, recInfo):
    """
    convert the gaze_positions.csv file to a pandas dataframe
    """
    isCore = recInfo.eye_tracker is utils.EyeTracker.Pupil_Core

    df = pd.read_csv(file)

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

    # mark data with insufficient confidence as missing.
    # for pupil core, pupil labs recommends a threshold of 0.6,
    # for the pupil invisible its a binary signal, and
    # confidence 0 should be excluded
    confThresh = 0.6 if isCore else 0
    todo = [x for x in idx if x in lookup.values()]
    toRemove = df.confidence <= confThresh
    for c in todo[2:]:
        df.loc[toRemove,c] = np.nan

    # convert timestamps from s to ms
    df.loc[:,'timestamp'] *= 1000.0

    # turn gaze locations into pixel data with origin in top-left
    df.loc[:,'vid_gaze_pos_x'] *= sceneVideoDimensions[0]
    df.loc[:,'vid_gaze_pos_y'] = (1-df.loc[:,'vid_gaze_pos_y'])*sceneVideoDimensions[1] # turn origin from bottom-left to top-left

    # return the dataframe
    return df