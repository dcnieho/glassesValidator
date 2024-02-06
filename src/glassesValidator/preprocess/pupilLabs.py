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


def preprocessData(output_dir, device=None, source_dir=None, rec_info=None):
    """
    Run all preprocessing steps on pupil data
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
        if not rec_info.eye_tracker in [utils.EyeTracker.Pupil_Core, utils.EyeTracker.Pupil_Invisible, utils.EyeTracker.Pupil_Neon]:
            raise ValueError(f'Provided rec_info is for a device ({rec_info.eye_tracker.value}) that is not a {utils.EyeTracker.Pupil_Core.value}, a {utils.EyeTracker.Pupil_Invisible.value} or a {utils.EyeTracker.Pupil_Neon.value}. Cannot use.')
        if not rec_info.proc_directory_name:
            rec_info.proc_directory_name = utils.make_fs_dirname(rec_info, output_dir)
        newDir = output_dir / rec_info.proc_directory_name
        if newDir.is_dir():
            raise RuntimeError(f'Output directory specified in rec_info ({rec_info.proc_directory_name}) already exists in the outputDir ({output_dir}). Cannot use.')

    if device is None and rec_info is None:
        raise RuntimeError('Either the "device" or the "rec_info" input argument should be set.')
    if device is not None:
        device = utils.type_string_to_enum(device)
        if not device in [utils.EyeTracker.Pupil_Core, utils.EyeTracker.Pupil_Invisible, utils.EyeTracker.Pupil_Neon]:
            raise ValueError(f'Provided device ({rec_info.eye_tracker.value}) is not a {utils.EyeTracker.Pupil_Core.value}, a {utils.EyeTracker.Pupil_Invisible.value} or a {utils.EyeTracker.Pupil_Neon.value}.')
    if rec_info is not None:
        if device is not None:
            if rec_info.eye_tracker != device:
                raise ValueError(f'Provided device ({device.value}) did not match device specific in rec_info ({rec_info.eye_tracker.value}). Provide matching values or do not provide the device input argument.')
        else:
            device = rec_info.eye_tracker

    print(f'processing: {source_dir.name}')


    ### check and copy needed files to the output directory
    print('Check and copy raw data...')
    ### check pupil recording and get export directory
    exportFile, is_cloud_export = checkPupilRecording(source_dir)
    if rec_info is not None:
        checkRecording(source_dir, rec_info)
    else:
        rec_info = getRecordingInfo(source_dir, device)
        if rec_info is None:
            raise RuntimeError(f"The folder {source_dir} is not recognized as a {device.value} recording.")

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
    if is_cloud_export:
        scene_vid = list(source_dir.glob('*.mp4'))
        if len(scene_vid)!=1:
            raise RuntimeError(f'Scene video missing or more than one found for Pupil Cloud export in folder {source_dir}')
        shutil.copyfile(str(scene_vid[0]), str(newDataDir / 'worldCamera.mp4'))
    else:
        shutil.copyfile(str(source_dir / 'world.mp4'), str(newDataDir / 'worldCamera.mp4'))
    print(f'Input data copied to: {newDataDir}')


    ### get camera cal
    print('Getting camera calibration...')
    if is_cloud_export:
        sceneVideoDimensions = getCameraCalFromCloudExport(source_dir, newDataDir, rec_info)
    else:
        match rec_info.eye_tracker:
            case utils.EyeTracker.Pupil_Core:
                sceneVideoDimensions = getCameraFromMsgPack(source_dir, newDataDir)
            case utils.EyeTracker.Pupil_Invisible:
                if (source_dir/'calibration.bin').is_file():
                    sceneVideoDimensions = getCameraCalFromBinFile(source_dir, newDataDir)
                else:
                    sceneVideoDimensions = getCameraCalFromOnline(source_dir, newDataDir, rec_info)


    ### get gaze data and video frame timestamps
    print('Prepping gaze data...')
    if is_cloud_export:
        gazeDf, frameTimestamps = formatGazeDataCloudExport(source_dir, exportFile, sceneVideoDimensions, rec_info)
    else:
        gazeDf, frameTimestamps = formatGazeDataPupilPlayer(source_dir, exportFile, sceneVideoDimensions, rec_info)

    # write the gaze data to a csv file
    gazeDf.to_csv(str(newDataDir / 'gazeData.tsv'), sep='\t', na_rep='nan', float_format="%.8f")

    # also store frame timestamps
    frameTimestamps.to_csv(str(newDataDir / 'frameTimestamps.tsv'), sep='\t')

    # indicate import finished
    utils.update_recording_status(newDataDir, utils.Task.Imported, utils.Status.Finished)


def checkPupilRecording(inputDir):
    """
    This checks that the folder is properly prepared,
    i.e., either:
    - opened in pupil player and an export was run (currently Pupil Core or Pupil Invisible)
    - exported from Pupil Cloud (currently Pupil Invisible or Pupil Neon)
    """
    # check we have an info.player.json file
    if not (inputDir / 'info.player.json').is_file():
        # possibly a pupil cloud export
        if not (inputDir / 'info.json').is_file() or not (inputDir / 'gaze.csv').is_file():
            raise RuntimeError(f'Neither the info.player.json file nor the info.json and gaze.csv files are found for {inputDir}. Either export from Pupil Cloud or, if the folder contains raw sensor data, open the recording in Pupil Player and run an export before importing into glassesValidator.')
        return inputDir / 'gaze.csv', True

    else:
        # check we have an export in the input dir
        inputExpDir = inputDir / 'exports'
        if not inputExpDir.is_dir():
            raise RuntimeError(f'no exports folder for {inputDir}. Perform a recording export in Pupil Player before importing into glassesValidator.')

        # get latest export in that folder that contain a gaze position file
        gpFiles = sorted(list(inputExpDir.rglob('*gaze_positions*.csv')))
        if not gpFiles:
            raise RuntimeError(f'There are no exports in the folder {inputExpDir}. Perform a recording export in Pupil Player before importing into glassesValidator.')

        return gpFiles[-1], False


def getRecordingInfo(inputDir, device):
    # returns None if not a recording directory
    recInfo = utils.Recording(source_directory=inputDir, eye_tracker=device)

    if (inputDir / 'info.player.json').is_file():
        # Pupil player export
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

            case utils.EyeTracker.Pupil_Neon:
                return None # there are no pupil player exports for the Neon eye tracker

            case _:
                print(f"Device {device} unknown")
                return None
    else:
        # pupil cloud export, for either Pupil Invisible or Pupil Neon
        if device==utils.EyeTracker.Pupil_Core:
            return None

        # raw sensor data also contain an info.json (checked below), so checking
        # that file is not enough to see if this is a Cloud Export. Check gaze.csv
        # presence
        if not (inputDir / 'gaze.csv').is_file():
            return None

        file = inputDir / 'info.json'
        if not file.is_file():
            return None
        with open(file, 'r') as j:
            iInfo = json.load(j)

        # check this is for the expected device
        is_neon = 'Neon' in iInfo['android_device_name'] or 'frame_name' in iInfo
        if device==utils.EyeTracker.Pupil_Invisible and is_neon:
            return None
        elif device==utils.EyeTracker.Pupil_Neon and not is_neon:
            return None

        recInfo.name = iInfo['template_data']['recording_name']
        recInfo.recording_software_version = iInfo['app_version']
        recInfo.start_time = utils.Timestamp(int(iInfo['start_time']//1000000000))  # UTC in nanoseconds, keep second part
        recInfo.duration   = int(iInfo['duration']//1000000)                        # in nanoseconds, convert to ms
        if is_neon:
            recInfo.glasses_serial = iInfo['module_serial_number']
        else:
            recInfo.glasses_serial = iInfo['glasses_serial_number']
            recInfo.scene_camera_serial = iInfo['scene_camera_serial_number']
        recInfo.recording_unit_serial = iInfo['android_device_id']
        if is_neon:
            recInfo.firmware_version = f"{iInfo['pipeline_version']} ({iInfo['firmware_version'][0]}.{iInfo['firmware_version'][1]})"
        else:
            recInfo.firmware_version = iInfo['pipeline_version']
        recInfo.participant = iInfo['wearer_name']

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

    # for invisible and neon recordings we have a bit more info
    if recInfo.eye_tracker in [utils.EyeTracker.Pupil_Invisible, utils.EyeTracker.Pupil_Neon]:
        if recInfo.glasses_serial!=actualRecInfo.glasses_serial:
            raise ValueError(f"A recording with the glasses_serial \"{recInfo.glasses_serial}\" was not found in the folder {inputDir}.")
        if recInfo.recording_unit_serial!=actualRecInfo.recording_unit_serial:
            raise ValueError(f"A recording with the recording_unit_serial \"{recInfo.recording_unit_serial}\" was not found in the folder {inputDir}.")
        if recInfo.eye_tracker==utils.EyeTracker.Pupil_Invisible and recInfo.scene_camera_serial!=actualRecInfo.scene_camera_serial:
            raise ValueError(f"A recording with the scene_camera_serial \"{recInfo.scene_camera_serial}\" was not found in the folder {inputDir}.")
        if (recInfo.participant is not None or actualRecInfo.participant is not None) and recInfo.participant!=actualRecInfo.participant:
            raise ValueError(f"A recording with the participant \"{recInfo.participant}\" was not found in the folder {inputDir}.")


def getCameraFromMsgPack(inputDir, outputDir):
    """
    Read camera calibration from recording information file
    """
    camInfo = getCamInfo(inputDir / 'world.intrinsics')

    # rename some fields, ensure they are numpy arrays
    camInfo['cameraMatrix'] = np.array(camInfo.pop('camera_matrix'))
    camInfo['distCoeff']    = np.array(camInfo.pop('dist_coefs')).flatten()
    camInfo['resolution']   = np.array(camInfo['resolution'])

    # store to file
    storeCameraCalibration(camInfo, outputDir)

    return camInfo['resolution']

def getCameraCalFromBinFile(inputDir, outputDir):
    # provided by pupil-labs
    cal = np.fromfile(
        inputDir / 'calibration.bin',
        np.dtype(
            [
                ("serial", "5a"),
                ("scene_camera_matrix", "(3,3)d"),
                ("scene_distortion_coefficients", "8d"),
                ("scene_extrinsics_affine_matrix", "(3,3)d"),
            ]
        ),
    )
    camInfo = {}
    camInfo['serial_number']= str(cal["serial"])
    camInfo['cameraMatrix'] = cal["scene_camera_matrix"].reshape((3,3))
    camInfo['distCoeff']    = cal["scene_distortion_coefficients"].reshape((8,1))
    camInfo['extrinsic']    = cal["scene_extrinsics_affine_matrix"].reshape((3,3))

    # get resolution from the local intrinsics file or scene video
    camInfo['resolution']   = getSceneCameraResolution(inputDir, outputDir)

    # store to xml file
    storeCameraCalibration(camInfo, outputDir)

    return camInfo['resolution']


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

    # get resolution from the local intrinsics file or scene video
    camInfo['resolution']   = getSceneCameraResolution(inputDir, outputDir)

    # store to xml file
    storeCameraCalibration(camInfo, outputDir)

    return camInfo['resolution']

def getCameraCalFromCloudExport(inputDir, outputDir, recInfo):
    file = inputDir / 'scene_camera.json'
    if not file.is_file():
        return None
    with open(file, 'r') as j:
        camInfo = json.load(j)

    camInfo['cameraMatrix'] = np.array(camInfo.pop('camera_matrix'))
    if 'dist_coefs' in camInfo:
        camInfo['distCoeff']    = np.array(camInfo.pop('dist_coefs')).flatten()
    else:
        camInfo['distCoeff']    = np.array(camInfo.pop('distortion_coefficients')).flatten()

    # get resolution from the scene video
    camInfo['resolution']   = getSceneCameraResolution(inputDir, outputDir)

    # store to xml file
    storeCameraCalibration(camInfo, outputDir)

    return camInfo['resolution']

def storeCameraCalibration(camInfo, outputDir):
    fs = cv2.FileStorage(str(outputDir / 'calibration.xml'), cv2.FILE_STORAGE_WRITE)
    for key,value in camInfo.items():
        fs.write(name=key,val=value)
    fs.release()

def getCamInfo(camInfoFile):
    with open(camInfoFile, 'rb') as f:
        camInfo = msgpack.unpack(f)

    # get keys which denote a camera resolution
    rex = re.compile('^\(\d+, \d+\)$')

    keys = [k for k in camInfo if rex.match(k)]
    if len(keys)!=1:
        raise RuntimeError('No camera intrinsics or intrinsics for more than one camera found')
    return camInfo[keys[0]]

def getSceneCameraResolution(inputDir, outputDir):
    if (inputDir / 'world.intrinsics').is_file():
        return np.array(getCamInfo(inputDir / 'world.intrinsics')['resolution'])
    else:
        import cv2
        cap = cv2.VideoCapture(str(outputDir / 'worldCamera.mp4'))
        if cap.isOpened():
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        return np.array([width, height])


def formatGazeDataPupilPlayer(inputDir, exportFile, sceneVideoDimensions, recInfo):
    # convert the json file to pandas dataframe
    df = readGazeDataPupilPlayer(exportFile, sceneVideoDimensions, recInfo)

    # get timestamps for the scene video
    frameTs = utils.getFrameTimestampsFromVideo(inputDir / 'world.mp4')

    # check pupil-labs' frame timestamps because we may need to correct
    # frame indices in case of holes in the video
    # also need this to correctly timestamp gaze samples
    if (inputDir / 'world_lookup.npy').is_file():
        ft = pd.DataFrame(np.load(str(inputDir / 'world_lookup.npy')))
        ft['frame_idx'] = ft.index
        ft.loc[ft['container_idx']==-1,'container_frame_idx'] = -1
        needsAdjust = not ft['frame_idx'].equals(ft['container_frame_idx'])
        # prep for later clean up
        toDrop = [x for x in ft.columns if x not in ['frame_idx','timestamp']]
        # do further adjustment that may be needed
        if needsAdjust:
            # not all video frames were encoded into the video file. Need to adjust
            # frame_idx in the gaze data to match actual video file
            temp = pd.merge(df,ft,on='frame_idx')
            temp['frame_idx'] = temp['container_frame_idx']
            temp = temp.rename(columns={'timestamp_x':'timestamp'})
            toDrop.append('timestamp_y')
            df   = temp.drop(columns=toDrop)
    else:
        ft = pd.DataFrame()
        ft['timestamp'] = np.load(str(inputDir / 'world_timestamps.npy'))*1000.0
        ft.index.name = 'frame_idx'
        # check there are no gaps in the video file
        if df['frame_idx'].max() > ft.index.max():
            raise RuntimeError('It appears there are frames missing in the scene video, but the file world_lookup.npy that would be needed to deal with that is missing. You can generate it by opening the recording in pupil player.')

    # set t=0 to video start time
    t0 = ft['timestamp'].iloc[0]*1000-frameTs['timestamp'].iloc[0]
    df.loc[:,'timestamp'] -= t0

    # set timestamps as index for gaze
    df = df.set_index('timestamp')

    return df, frameTs


def readGazeDataPupilPlayer(file, sceneVideoDimensions, recInfo):
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

def formatGazeDataCloudExport(inputDir, exportFile, sceneVideoDimensions, recInfo):
    df = readGazeDataCloudExport(exportFile, sceneVideoDimensions, recInfo)

    frameTimestamps = pd.read_csv(inputDir/'world_timestamps.csv')
    frameTimestamps = frameTimestamps.rename(columns={'timestamp [ns]': 'timestamp'})
    frameTimestamps = frameTimestamps.drop(columns=[x for x in frameTimestamps.columns if x not in ['timestamp']])
    frameTimestamps['frame_idx'] = frameTimestamps.index
    frameTimestamps = frameTimestamps.set_index('frame_idx')

    # set t=0 to video start time
    t0_ns = frameTimestamps['timestamp'].iloc[0]
    df.loc[:,'timestamp']               -= t0_ns
    frameTimestamps.loc[:,'timestamp']  -= t0_ns
    df.loc[:,'timestamp']               /= 1000000.0    # convert timestamps from ns to ms
    frameTimestamps.loc[:,'timestamp']  /= 1000000.0

    # set timestamps as index for gaze
    df = df.set_index('timestamp')

    # use the frame timestamps to assign a frame number to each data point
    frameIdx = utils.tssToFrameNumber(df.index,frameTimestamps['timestamp'].to_numpy())
    df.insert(0,'frame_idx',frameIdx['frame_idx'])

    return df, frameTimestamps


def readGazeDataCloudExport(file, sceneVideoDimensions, recInfo):
    df = pd.read_csv(file)

    # rename and reorder columns
    lookup = {'timestamp [ns]': 'timestamp',
              'gaze x [px]': 'vid_gaze_pos_x',
              'gaze y [px]': 'vid_gaze_pos_y'}
    df=df.rename(columns=lookup)
    df=df.drop(columns=[x for x in df.columns if x not in ['timestamp','vid_gaze_pos_x','vid_gaze_pos_y','worn']])

    # mark data where eye tracker is not worn as missing
    todo = [lookup[k] for k in lookup if lookup[k] in df.columns]
    toRemove = df.worn == 0
    for c in todo[2:]:
        df.loc[toRemove,c] = np.nan

    return df