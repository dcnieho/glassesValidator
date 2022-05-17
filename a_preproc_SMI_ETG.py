"""
Cast raw SMI ETG data into common format.

Tested with Python 3.8, open CV 4.0.1

The output directory will contain:
    - frameTimestamps.tsv: frame number and corresponding timestamps for each frame in video
    - worldCamera.mp4: the video from the point-of-view scene camera on the glasses
    - gazeData.tsv: gaze data, where all 2D gaze coordinates are represented w/r/t the world camera,
                    and 3D coordinates in the coordinate frame of the glasses (may need rotation/translation
                    to represent in world camera's coordinate frame)
    - calibration.xml: info about the camera intrinsics and transform glasses coordinate system to
                       camera coordinate system
"""

import shutil
import os
from pathlib import Path
import cv2
import pandas as pd
import numpy as np
import math
import configparser
from io import StringIO
from scipy.spatial.transform import Rotation

import utils


def preprocessData(inputDir, outputDir):
    """
    Run all preprocessing steps on tobii data
    """
    print('processing: {}'.format(inputDir.name))
    ### copy the raw data to the output directory
    print('Copying raw data...')
    newDataDir = copySMIRecordings(inputDir, outputDir)
    for r in newDataDir:
        print('Input data copied to: {}'.format(r))

    #### prep the copied data...
    for r in newDataDir:
        print('{}...'.format(r.name))
        print('  Getting camera calibration...')
        sceneVideoDimensions = getCameraFromFile(r)
        print('  Prepping gaze data...')
        gazeDf, frameTimestamps = formatGazeData(r, sceneVideoDimensions)

        # write the gaze data to a csv file
        gazeDf.to_csv(str(r / 'gazeData.tsv'), sep='\t', na_rep='nan', float_format="%.8f")

        # also store frame timestamps
        frameTimestamps.to_csv(str(r / 'frameTimestamps.tsv'), sep='\t')


def copySMIRecordings(inputDir, outputDir):
    """
    Copy the relevant files from the specified input dir to the specified output dirs
    NB: an SMI directory may contain multiple recordings
    """
    participant = inputDir.name
    
    # get recordings. We expect the user to rename their exports to have the same format
    # as the other files in a project directory. So e.g., data exported from 001-2-recording.idf
    # for the corresponding 001-2-recording.avi, should be named 001-2-recording.txt
    outputDirs = []
    for r in inputDir.glob('*.txt'):
        if not str(r).endswith('recording.txt'):
            print("file {} not recognized as a recording (wrong name, should end with '-recording.txt'), skipping".format(str(r.name)))
            continue

        recording = str(r.name)[:-len('-recording.txt')]

        outputDirs.append(outputDir / ('SMI_ETG2_%s_%s' % (participant,recording)))
        if not outputDirs[-1].is_dir():
            outputDirs[-1].mkdir()

        # Copy relevent files to new directory
        exportName = r.stem.split('-')
        exportName = '-'.join([exportName[0],exportName[1],'export'])
        if not (inputDir / (exportName+'.avi')).is_file():
            raise RuntimeError("file {} cannot be found in the folder {}, make sure you export the scene video using BeGaze as described in the glassesValidator manual".format(exportName+'.avi',inputDir))

        for f in [('codec1.bin','caminfo.txt'), (r.name,'gazedata.txt'), (exportName+'.avi','worldCamera.avi')]:
            outFileName = f[0]
            if f[1] is not None:
                outFileName = f[1]
            shutil.copyfile(str(inputDir / f[0]), str(outputDirs[-1] / outFileName))

        # if ffmpeg is on path, remux avi to mp4 (reencode audio from flac to aac as flac is not supported in mp4)
        if shutil.which('ffmpeg') is not None:
            # make mp4
            cmd_str = ' '.join(['ffmpeg', '-y', '-i', '"'+str(outputDirs[-1] / 'worldCamera.avi')+'"', '-vcodec', 'copy', '-acodec', 'aac', '"'+str(outputDirs[-1] / 'worldCamera.mp4')+'"'])
            os.system(cmd_str)
            # clean up
            if (outputDirs[-1] / 'worldCamera.mp4').is_file():
                (outputDirs[-1] / 'worldCamera.avi').unlink(missing_ok=True)

    # return the full path to the output dir
    return outputDirs

def getCameraFromFile(inputDir):
    """
    Read camera calibration from information file
    """
    with open(inputDir / 'caminfo.txt', 'r') as file:
        camInfoStr = file.read().replace('## ', '')

    # file no longer needed, remove
    (inputDir / 'caminfo.txt').unlink(missing_ok=True)
    
    caminfo = configparser.ConfigParser(converters={'nparray': lambda x: np.fromstring(x,sep='\t')})
    caminfo.read_string(camInfoStr)

    camera = {}
    camera['FOV'] = caminfo.getfloat("MiiSimulation",'SceneCamFOV')
    camera['resolution'] = np.array([caminfo.getint("MiiSimulation",'SceneCamWidth'), caminfo.getint("MiiSimulation",'SceneCamHeight')])
    camera['sensorOffsets'] = np.array([caminfo.getfloat("MiiSimulation",'SceneCamSensorOffsetX'), caminfo.getfloat("MiiSimulation",'SceneCamSensorOffsetY')])
    
    camera['radialDistortion'] = caminfo.getnparray("MiiSimulation",'SceneCamRadialDistortion')
    camera['tangentialDistortion'] = caminfo.getnparray("MiiSimulation",'SceneCamTangentialDistortion')

    camera['position'] = caminfo.getnparray("MiiSimulation",'SceneCamPos')
    camera['eulerAngles'] = np.array([caminfo.getfloat("MiiSimulation",'SceneCamOrX'), caminfo.getfloat("MiiSimulation",'SceneCamOrY'), caminfo.getfloat("MiiSimulation",'SceneCamOrZ')])

    # now turn these fields into focal length and principal point
    # 1. FOV is horizontal FOV of camera, given resolution we can compute
    # focal length. We assume vertical focal length is the same, best we can do
    fl = camera['resolution'][0]/(2*math.tan(camera['FOV']/2/180*math.pi))
    camera['focalLength'] = np.array([fl, fl])
    # 2. sensor offsets seem to be relative to center of sensor
    camera['principalPoint'] = camera['resolution']/2.+camera['sensorOffsets']
    # 3. turn euler angles into rotation matrix (180-Rz because board space has positive X rightward and positive Y downward)
    camera['rotation'] = Rotation.from_euler('XYZ', [camera['eulerAngles'][0], camera['eulerAngles'][1], 180-camera['eulerAngles'][2]], degrees=True).as_matrix()

    # turn into camera matrix and distortion coefficients as used by OpenCV
    camera['cameraMatrix'] = np.identity(3)
    camera['cameraMatrix'][0,0] = camera['focalLength'][0]
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

    return camera['resolution']


def formatGazeData(inputDir, sceneVideoDimensions):
    """
    load gazedata file
    format to get the gaze coordinates w.r.t. world camera, and timestamps for
    every frame of video

    Returns:
        - formatted dataframe with cols for timestamp, frame_idx, and gaze data
        - np array of frame timestamps
    """

    # convert the text file to pandas dataframe
    df = gazedata2df(inputDir / 'gazedata.txt', sceneVideoDimensions)

    # read video file, create array of frame timestamps
    if (inputDir / 'worldCamera.mp4').is_file():
        frameTimestamps = utils.getFrameTimestampsFromVideo(inputDir / 'worldCamera.mp4')
    else:
        frameTimestamps = utils.getFrameTimestampsFromVideo(inputDir / 'worldCamera.avi')

    # SMI frame counter seems to be of the format HH:MM:SS:FR, where HH:MM:SS is a normal
    # hour, minute, second timecode, and FR is a frame number for within that second. The
    # frame number is zero-based, and runs ranges from 0 to FS-1 where FS is the frame
    # rate of the camera (e.g. 24 Hz). Convert their frame counter to a normal counter
    df_fr = pd.DataFrame(df['Frame'].apply(lambda x: [int(y) for y in x.split(':')]).to_list(),columns=['hour','minute','second','frame'])
    frameRate = df_fr.frame.max()+1
    df.insert(0,'frame_idx',(((df_fr.hour*60 + df_fr.minute)*60 + df_fr.second)*frameRate + df_fr.frame).to_numpy())
    # NB: seems we can get frame numbers in the data which are beyond the length of the video. so be it
    df=df.drop(columns=['Frame'])
    # NB: it seems the SMI export doesn't strictly follow their own timecode. Subtracting min and
    # adding 1 so that the frame_idx starts at 1 for the data empirically seems to line up with the
    # SMI export
    df.frame_idx = df.frame_idx-df.frame_idx.min()+1

    # return the gaze data df and frame time stamps array
    return df, frameTimestamps


def gazedata2df(textFile,sceneVideoDimensions):
    """
    convert the gazedata file to a pandas dataframe
    """
    
    with open(textFile, 'r') as file:
        textData = file.read()

    df = pd.read_table(StringIO(textData),comment='#',index_col=False)

    # json no longer needed, remove
    textFile.unlink(missing_ok=True)

    # get assumed viewing distance
    matchedLines    = [line for line in textData.split('\n') if "Head Distance [mm]" in line]
    viewingDistance = float(matchedLines[0].split('\t')[1])

    # prepare data frame
    # remove unneeded columns
    df=df.drop(columns=['Type', 'Trial', 'Aux1'],errors='ignore') # drop these columns if they exist
    # if we have monocular point of regard data, check if its not identical to binocular
    if 'L POR X [px]' in df.columns:
        xEqual = np.all( np.logical_or(df['L POR X [px]'] == df['B POR X [px]'], np.isnan(df['L POR X [px]']) ))
        yEqual = np.all( np.logical_or(df['L POR Y [px]'] == df['B POR Y [px]'], np.isnan(df['L POR Y [px]']) ))
        if xEqual and yEqual:
            df=df.drop(columns=['L POR X [px]', 'L POR Y [px]'])
    if 'R POR X [px]' in df.columns:
        xEqual = np.all( np.logical_or(df['R POR X [px]'] == df['B POR X [px]'], np.isnan(df['R POR X [px]']) ))
        yEqual = np.all( np.logical_or(df['R POR Y [px]'] == df['B POR Y [px]'], np.isnan(df['R POR Y [px]']) ))
        if xEqual and yEqual:
            df=df.drop(columns=['R POR X [px]', 'R POR Y [px]'])
    
    # rename and reorder columns
    lookup = {'Time': 'timestamp',
               'L EPOS X':'l_gaze_ori_x',
               'L EPOS Y':'l_gaze_ori_y',
               'L EPOS Z':'l_gaze_ori_z',
               'L Pupil Diameter [mm]':'l_pup_diam',
               'L GVEC X':'l_gaze_dir_x',
               'L GVEC Y':'l_gaze_dir_y',
               'L GVEC Z':'l_gaze_dir_z',
               'R EPOS X':'r_gaze_ori_x',
               'R EPOS Y':'r_gaze_ori_y',
               'R EPOS Z':'r_gaze_ori_z',
               'R Pupil Diameter [mm]':'r_pup_diam',
               'R GVEC X':'r_gaze_dir_x',
               'R GVEC Y':'r_gaze_dir_y',
               'R GVEC Z':'r_gaze_dir_z',
               'B POR X [px]':'vid_gaze_pos_x',
               'B POR Y [px]':'vid_gaze_pos_y',
               'L Dia X [px]':'l_pup_diam_x_px',
               'L Dia Y [px]':'l_pup_diam_y_px',
               'R Dia X [px]':'r_pup_diam_x_px',
               'R Dia Y [px]':'r_pup_diam_y_px'}
    df=df.rename(columns=lookup)
    # reorder
    idx = [lookup[k] for k in lookup if lookup[k] in df.columns]
    idx.extend([x for x in df.columns if x not in idx])   # append columns not in lookup
    df = df[idx]

    # convert timestamps from us to ms and set as index
    df.loc[:,'timestamp'] /= 1000.0
    df = df.set_index('timestamp')

    # return the dataframe
    return df


if __name__ == '__main__':
    basePath = Path(__file__).resolve().parent / 'data'
    inBasePath = basePath / 'SMI_ETG2'
    outBasePath = basePath / 'preprocced'
    if not outBasePath.is_dir():
        outBasePath.mkdir()
    for d in inBasePath.iterdir():
        if d.is_dir():
            preprocessData(d,outBasePath)
