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
import struct
import math
import mp4analyser.iso


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
    sceneVideoDimensions = getCameraFromTSLV(newDataDir)
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
    with open(str(inputDir / 'participant.json'), 'rb') as j:
        pInfo = json.load(j)
    participant = pInfo['pa_info']['Name']
    with open(str(inputDir / 'recording.json'), 'rb') as j:
        rInfo = json.load(j)
    recording = rInfo['rec_info']['Name']

    outputDir = outputDir / ('tobiiG2_%s_%s' % (participant,recording))
    if not outputDir.is_dir():
        outputDir.mkdir()

    # Copy relevent files to new directory
    inputDir = inputDir / 'segments' / '1'
    for f in [('livedata.json.gz',None), ('et.tslv.gz',None), ('fullstream.mp4','worldCamera.mp4')]:
        outFileName = f[0]
        if f[1] is not None:
            outFileName = f[1]
        shutil.copyfile(str(inputDir / f[0]), str(outputDir / outFileName))

    # Unzip the gaze data and tslv files
    for f in ['livedata.json.gz', 'et.tslv.gz']:
        with gzip.open(str(outputDir / f)) as zipFile:
            with open(str(outputDir / Path(f).stem), 'wb') as unzippedFile:
                for line in zipFile:
                    unzippedFile.write(line)
        (outputDir / f).unlink(missing_ok=True)

    # return the full path to the output dir
    return outputDir

def getCameraFromTSLV(inputDir):
    """
    Read binary TSLV file until camera calibration information is retrieved
    """
    with open(str(inputDir / 'et.tslv'), "rb") as f:
        # first look for camera item (TSLV type==300)
        while True:
            tslvType= struct.unpack('h',f.read(2))[0]
            status  = struct.unpack('h',f.read(2))[0]
            payloadLength = struct.unpack('i',f.read(4))[0]
            payloadLengthPadded = math.ceil(payloadLength/4)*4
            if tslvType != 300:
                # skip payload
                f.read(payloadLengthPadded)
            else:
                break
        
        # read info about camera
        camera = {}
        camera['id']       = struct.unpack('b',f.read(1))[0]
        camera['location'] = struct.unpack('b',f.read(1))[0]
        f.read(2) # skip padding
        camera['position'] = np.array(struct.unpack('3f',f.read(4*3)))
        camera['rotation'] = np.reshape(struct.unpack('9f',f.read(4*9)),(3,3))
        camera['focalLength'] = np.array(struct.unpack('2f',f.read(4*2)))
        camera['skew'] = struct.unpack('f',f.read(4))[0]
        camera['principalPoint'] = np.array(struct.unpack('2f',f.read(4*2)))
        camera['radialDistortion'] = np.array(struct.unpack('3f',f.read(4*3)))
        camera['tangentialDistortion'] = np.array(struct.unpack('3f',f.read(4*3))[:-1]) # drop last element (always zero), since there are only two tangential distortion parameters
        camera['resolution'] = np.array(struct.unpack('2h',f.read(2*2)))
    
    # turn into camera matrix and distortion coefficients as used by OpenCV
    camera['cameraMatrix'] = np.identity(3)
    camera['cameraMatrix'][0,0] = camera['focalLength'][0]
    camera['cameraMatrix'][0,1] = camera['skew']
    camera['cameraMatrix'][1,1] = camera['focalLength'][1]
    camera['cameraMatrix'][0,2] = camera['principalPoint'][0]
    camera['cameraMatrix'][1,2] = camera['principalPoint'][1]

    camera['distCoeff'] = np.zeros(5)
    camera['distCoeff'][:2]  = camera['radialDistortion'][:2]
    camera['distCoeff'][2:4] = camera['tangentialDistortion'][:2]
    camera['distCoeff'][4]   = camera['radialDistortion'][2]


    # store to file
    fs = cv2.FileStorage(str(inputDir / 'calibration.xml'), cv2.FILE_STORAGE_WRITE)
    for key,value in camera.items():
        fs.write(name=key,val=value)
    fs.release()

    # tslv no longer needed, remove
    (inputDir / 'et.tslv').unlink(missing_ok=True)

    return camera['resolution']


def formatGazeData(inputDir, sceneVideoDimensions):
    """
    load livedata.json
    format to get the gaze coordinates w/r/t world camera, and timestamps for every frame of video

    Returns:
        - formatted dataframe with cols for timestamp, frame_idx, and gaze data
        - np array of frame timestamps
    """

    # convert the json file to pandas dataframe
    df,scene_video_ts_offset = json2df(inputDir / 'livedata.json', sceneVideoDimensions)

    # read video file, create array of frame timestamps
    frameTimestamps = getVidFrameTimestamps(str(inputDir / 'worldCamera.mp4'))
    frameTimestamps['timestamp'] += scene_video_ts_offset

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


def getVidFrameTimestamps(vid_file):
    """
    Parse the supplied video, return an array of frame timestamps
    """
    
    # method 2, parse mp4 file
    boxes   = mp4analyser.iso.Mp4File(vid_file)
    # 1. find mdat box
    moov    = boxes.child_boxes[[i for i,x in enumerate(boxes.child_boxes) if x.type=='moov'][0]]
    # 2. find track boxes
    trakIdxs= [i for i,x in enumerate(moov.child_boxes) if x.type=='trak']
    # 3. check which track contains video
    trakIdx = [i for i,x in enumerate(boxes.get_summary()['track_list']) if x['media_type']=='video'][0]
    trak    = moov.child_boxes[trakIdxs[trakIdx]]
    # 4. get mdia
    mdia    = trak.child_boxes[[i for i,x in enumerate(trak.child_boxes) if x.type=='mdia'][0]]
    # 5. get time_scale field from mdhd
    time_base = mdia.child_boxes[[i for i,x in enumerate(mdia.child_boxes) if x.type=='mdhd'][0]].box_info['timescale']
    # 6. get minf
    minf    = mdia.child_boxes[[i for i,x in enumerate(mdia.child_boxes) if x.type=='minf'][0]]
    # 7. get stbl
    stbl    = minf.child_boxes[[i for i,x in enumerate(minf.child_boxes) if x.type=='stbl'][0]]
    # 8. get sample table from stts
    samp_table = stbl.child_boxes[[i for i,x in enumerate(stbl.child_boxes) if x.type=='stts'][0]].box_info['entry_list']
    # 9. now we have all the info to determine the timestamps of each frame
    df = pd.DataFrame(samp_table) # easier to use that way
    totalFrames = df['sample_count'].sum()
    frameTs = np.zeros(totalFrames)
    # first uncompress delta table
    idx = 0
    for count,dur in zip(df['sample_count'], df['sample_delta']):
        frameTs[idx:idx+count] = dur
        idx = idx+count
    # turn into timestamps, first in time_base units
    frameTs = np.roll(frameTs,1)
    frameTs[0] = 0.
    frameTs = np.cumsum(frameTs)
    # now into timestamps in ms
    frameTs = frameTs/time_base*1000

    ### convert the frame_timestamps to dataframe
    frameIdx = np.arange(0, len(frameTs))
    frameTsDf = pd.DataFrame({'frame_idx': frameIdx, 'timestamp': frameTs})
    frameTsDf.set_index('frame_idx', inplace=True)
    
    return frameTsDf


def json2df(jsonFile,sceneVideoDimensions):
    """
    convert the livedata.json file to a pandas dataframe
    """
    # dicts to store sync points
    vtsSync  = list()       # scene video timestamp sync
    evtsSync = list()       # eye video timestamp sync (only if eye video was recorded)
    df = pd.DataFrame()     # empty dataframe to write data to

    with open(str(jsonFile), 'rb') as j:

        # loop over all lines in json file, each line represents unique json object
        for line in j:
            entry = json.loads(line)
            
            # if non-zero status (error), ensure data found in packet is marked as missing
            isError = False
            if entry['s'] != 0:
                isError = True

            ### a number of different dictKeys are possible, respond accordingly
            if 'vts' in entry.keys(): # "vts" key signfies a scene video timestamp (first frame, first keyframe, and ~1/min afterwards)
                vtsSync.append((entry['ts'], entry['vts'] if not isError else math.nan))
                continue

            ### a number of different dictKeys are possible, respond accordingly
            if 'evts' in entry.keys(): # "evts" key signfies an eye video timestamp (first frame, first keyframe, and ~1/min afterwards)
                evtsSync.append((entry['ts'], entry['evts'] if not isError else math.nan))
                continue

            # if this json object contains "eye" data (e.g. pupil info)
            if 'eye' in entry.keys():
                which_eye = entry['eye'][:1]
                if 'pc' in entry.keys():
                    # origin of gaze vector is the pupil center
                    df.loc[entry['ts'], which_eye + '_gaze_ori_x'] = entry['pc'][0] if not isError else math.nan
                    df.loc[entry['ts'], which_eye + '_gaze_ori_y'] = entry['pc'][1] if not isError else math.nan
                    df.loc[entry['ts'], which_eye + '_gaze_ori_z'] = entry['pc'][2] if not isError else math.nan
                elif 'pd' in entry.keys():
                    df.loc[entry['ts'], which_eye + '_pup_diam'] = entry['pd'] if not isError else math.nan
                elif 'gd' in entry.keys():
                    df.loc[entry['ts'], which_eye + '_gaze_dir_x'] = entry['gd'][0] if not isError else math.nan
                    df.loc[entry['ts'], which_eye + '_gaze_dir_y'] = entry['gd'][1] if not isError else math.nan
                    df.loc[entry['ts'], which_eye + '_gaze_dir_z'] = entry['gd'][2] if not isError else math.nan

            # otherwise it contains gaze position data
            else:
                if 'gp' in entry.keys():
                    df.loc[entry['ts'], 'vid_gaze_pos_x'] = entry['gp'][0]*sceneVideoDimensions[0] if not isError else math.nan
                    df.loc[entry['ts'], 'vid_gaze_pos_y'] = entry['gp'][1]*sceneVideoDimensions[1] if not isError else math.nan
                elif 'gp3' in entry.keys():
                    df.loc[entry['ts'], '3d_gaze_pos_x'] = entry['gp3'][0] if not isError else math.nan
                    df.loc[entry['ts'], '3d_gaze_pos_y'] = entry['gp3'][1] if not isError else math.nan
                    df.loc[entry['ts'], '3d_gaze_pos_z'] = entry['gp3'][2] if not isError else math.nan
                    
            # ignore anything else

    # find out t0. Do the same as GlassesViewer so timestamps are compatible
    # that is t0 is at timestamp of last video start (scene or eye)
    vtsSync  = np.array( vtsSync)
    evtsSync = np.array(evtsSync)
    t0s = [vtsSync[vtsSync[:,1]==0,0]]
    if len(evtsSync)>0:
        t0s.append(evtsSync[evtsSync[:,1]==0,0])
    t0 = max(t0s)

    # get timestamp offset for scene video
    scene_video_ts_offset = (t0s[0]-t0) / 1000.0

    # convert timestamps from us to ms
    df.index = (df.index - t0) / 1000.0

    # json no longer needed, remove
    jsonFile.unlink(missing_ok=True)

    # return the dataframe
    return df, scene_video_ts_offset


if __name__ == '__main__':
    basePath = Path(__file__).resolve().parent / 'data'
    inBasePath = basePath / 'tobiiG2'
    outBasePath = basePath / 'preprocced'
    if not outBasePath.is_dir():
        outBasePath.mkdir()
    for d in inBasePath.iterdir():
        if d.is_dir():
            preprocessData(d,outBasePath)
