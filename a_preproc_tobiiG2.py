"""
Format raw Tobii data.

Tested with Python 3.8, open CV 3.2

Since the data originates on a SD card (or temp directory somewhere), a new output directory
will be created for each recording. The output directory will be created within the output root path specified by the user,
and named according to [mo-day-yr]/[hr-min-sec] of the original creation time format.

The output directory will contain:
    - frame_timestamps.tsv: frame number and corresponding timestamps for each frame in video
    - worldCamera.mp4: the video from the point-of-view scene camera on the glasses
    - gazeData_world.tsv: gaze data, where all gaze coordinates are represented w/r/t the world camera
"""

# python 2/3 compatibility
from __future__ import division
from __future__ import print_function

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
    getCameraFromTSLV(newDataDir)
    print('Prepping gaze data...')
    gazeDf, frame_timestamps = formatGazeData(newDataDir)

    # write the gaze data (world camera coords) to a csv file
    gazeDf.to_csv(str(newDataDir / 'gazeData_raw.tsv'), sep='\t', na_rep='nan', float_format="%.8f")

    # write standard subset of gaze data (world camera coords) to a csv file
    gazeDf[['frame_idx','video_timestamp','gaze_pos_x','gaze_pos_y','3d_gaze_pos_x','3d_gaze_pos_y','3d_gaze_pos_z']].to_csv(\
        str(newDataDir / 'gazeData.tsv'), sep='\t', na_rep='nan', float_format="%.8f")

    # also store frame_timestamps
    frame_timestamps.to_csv(str(newDataDir / 'frame_timestamps.tsv'), sep='\t', index=False)

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

    outputDir = outputDir / ('tobii_%s_%s' % (participant,recording))
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
        camera['tangentialDistortion'] = np.array(struct.unpack('3f',f.read(4*3))[:-1]) # last element always zero, there are only two tangential distortion parameters
        camera['sensorDimensions'] = np.array(struct.unpack('2h',f.read(2*2)))
    
    # turn into camera matrix and distCoeffs as used by OpenCV
    camera['cameraMatrix'] = np.identity(3)
    camera['cameraMatrix'][0,0] = camera['focalLength'][0]
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


def formatGazeData(inputDir):
    """
    load livedata.json, write to csv
    format to get the gaze coordinates w/r/t world camera, and timestamps for every frame of video

    Returns:
        - formatted dataframe with cols for timestamp, frame_idx, and normalized gaze data X & Y
        - np array of frame timestamps
    """

    # convert the json file to pandas dataframe
    df = json_to_df(str(inputDir / 'livedata.json'))

    # drop any row that precedes the start of the video timestamps
    df = df[df['video_timestamp'] >= df['video_timestamp'].min()]

    # read video file, create array of frame timestamps
    frame_timestamps = getVidFrameTimestamps(str(inputDir / 'worldCamera.mp4'))

    # use the frame timestamps to assign a frame number to each data point
    df['frame_idx'] = np.zeros(df.index.shape,dtype='int64')
    for i,vts in enumerate(df['video_timestamp']):

        # get index where this vts would be inserted into the frame_timestamp array
        idx = np.searchsorted(frame_timestamps['timestamp'], vts)
        if idx == 0:
            idx = 1
        # since idx points to frame timestamp for frame after the one during
        # which the vts ocurred, correct
        idx -= 1

        # set the frame index based on this index value
        df.loc[df.index[i], 'frame_idx'] = frame_timestamps['frame_idx'].iat[idx]

    # build the formatted dataframe
    df.index.name = 'timestamp'

    # return the gaze data df and frame time stamps array
    colOrder = [x for x in df.columns.tolist() if x!='frame_idx']
    colOrder.insert(0,'frame_idx')  # make frame_idx the first column
    return df[colOrder], frame_timestamps


def getVidFrameTimestamps(vid_file):
    """
    Pasre the supplied video, return an array of frame timestamps
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
    frame_ts = np.zeros(totalFrames)
    # first uncompress delta table
    idx = 0
    for count,dur in zip(df['sample_count'], df['sample_delta']):
        frame_ts[idx:idx+count] = dur
        idx = idx+count
    # turn into timestamps, first in time_base units
    frame_ts = np.roll(frame_ts,1)
    frame_ts[0] = 0.
    frame_ts = np.cumsum(frame_ts)
    # now into timestamps in ms
    frame_ts = frame_ts/time_base*1000

    ### convert the frame_timestamps to dataframe
    frameIdx = np.arange(0, len(frame_ts))
    frame_ts_df = pd.DataFrame({'frame_idx': frameIdx, 'timestamp': frame_ts})
    
    return frame_ts_df


def json_to_df(json_file):
    """
    convert the livedata.json file to a pandas dataframe
    """
    # dicts to store sync points
    vts_sync = {}			# RECORDED video timestamp sync
    df = pd.DataFrame()     # empty dataframe to write data to

    with open(json_file, 'rb') as j:

        # loop over all lines in json file, each line represents unique json object
        for line in j:
            entry = json.loads(line)
            
            # if non-zero status, skip packet
            if entry['s'] != 0:
                continue

            ### a number of different dictKeys are possible, respond accordingly
            if 'vts' in entry.keys(): # "vts" key signfies a video timestamp (first frame, first keyframe, and ~1/min afterwards)
                vts_sync[entry['ts']] = entry['vts']
                continue

            # if this json object contains "eye" data (e.g. pupil info)
            if 'eye' in entry.keys():
                which_eye = entry['eye'][:1]
                if 'pc' in entry.keys():
                    df.loc[entry['ts'], which_eye + '_pup_cent_x'] = entry['pc'][0]
                    df.loc[entry['ts'], which_eye + '_pup_cent_y'] = entry['pc'][1]
                    df.loc[entry['ts'], which_eye + '_pup_cent_z'] = entry['pc'][2]
                elif 'pd' in entry.keys():
                    df.loc[entry['ts'], which_eye + '_pup_diam'] = entry['pd']
                elif 'gd' in entry.keys():
                    df.loc[entry['ts'], which_eye + '_gaze_dir_x'] = entry['gd'][0]
                    df.loc[entry['ts'], which_eye + '_gaze_dir_y'] = entry['gd'][1]
                    df.loc[entry['ts'], which_eye + '_gaze_dir_z'] = entry['gd'][2]

            # otherwise it contains gaze position data
            else:
                if 'gp' in entry.keys():
                    df.loc[entry['ts'], 'gaze_pos_x'] = entry['gp'][0]
                    df.loc[entry['ts'], 'gaze_pos_y'] = entry['gp'][1]
                elif 'gp3' in entry.keys():
                    df.loc[entry['ts'], '3d_gaze_pos_x'] = entry['gp3'][0]
                    df.loc[entry['ts'], '3d_gaze_pos_y'] = entry['gp3'][1]
                    df.loc[entry['ts'], '3d_gaze_pos_z'] = entry['gp3'][2]
                    
            # ignore anything else

        
        # now we need to make a video timestamp column
        df['video_timestamp'] = np.array(df.index)	   # df.index is the gaze data timestamps
        df.loc[df.index < min(sorted(vts_sync.keys())), 'video_timestamp'] = np.nan		# set rows that occur before the first frame to nan

        # for each new vts sync package, reindex all of the rows after that timestamp
        for key in sorted(vts_sync.keys()):
            df.loc[df.index >= key, 'video_timestamp'] = np.array(df.index)[df.index >= key]   # necessary if there are more than 2 keys in the list (prior key changes need to be reset for higher vts syncs)
            df.loc[df.index >= key, 'video_timestamp'] = df['video_timestamp'] - key + vts_sync[key]

        # note: the vts column indicates, in microseconds, where a given datapoint would occur in the video timeline
        # the timestamps of the videoframes themselves need to be gotten from the video file instead

        # convert timestamps from us to ms
        df.index              = (df.index - df.index[0]) / 1000.0
        df['video_timestamp'] = df['video_timestamp'] / 1000.0

        # return the dataframe
        return df


if __name__ == '__main__':
    basePath = Path(__file__).resolve().parent / 'data'
    inBasePath = basePath / 'tobiiG2'
    outBasePath = basePath / 'preprocced'
    if not outBasePath.is_dir():
        outBasePath.mkdir()
    for d in inBasePath.iterdir():
        if d.is_dir():
            preprocessData(d,outBasePath)
