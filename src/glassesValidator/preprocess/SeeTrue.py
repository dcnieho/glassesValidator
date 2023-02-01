"""
Cast raw SeeTrue data into common format.

The output directory will contain:
    - frameTimestamps.tsv: frame number and corresponding timestamps for each frame in video
    - worldCamera.mp4: the video from the point-of-view scene camera on the glasses
    - gazeData.tsv: gaze data, where all 2D gaze coordinates are represented w/r/t the world camera
"""

import shutil
import pathlib
import os
import cv2
import pandas as pd
import numpy as np

from .. import utils


def preprocessData(output_dir, input_dir=None, rec_info=None, cam_cal_file=None):

    if shutil.which('ffmpeg') is None:
        RuntimeError('ffmpeg must be on path to prep SeeTrue recording for processing with GlassesValidator')

    """
    Run all preprocessing steps on SeeTrue data
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
        if rec_info.eye_tracker!=utils.EyeTracker.SeeTrue:
            raise ValueError(f'Provided rec_info is for a device ({rec_info.eye_tracker.value}) that is not an {utils.EyeTracker.SeeTrue.value}. Cannot use.')
        if not rec_info.proc_directory_name:
            rec_info.proc_directory_name = utils.make_fs_dirname(rec_info, output_dir)
        newDir = output_dir / rec_info.proc_directory_name
        if newDir.is_dir():
            raise RuntimeError(f'Output directory specified in rec_info ({rec_info.proc_directory_name}) already exists in the outputDir ({output_dir}). Cannot use.')


    print(f'processing: {input_dir.name}')


    ### check and copy needed files to the output directory
    print('Check and copy raw data...')
    if rec_info is not None:
        if not checkRecording(input_dir, rec_info):
            raise ValueError(f"A recording with the name \"{rec_info.name}\" was not found in the folder {input_dir}.")
        recInfos = [rec_info]
    else:
        recInfos = getRecordingInfo(input_dir)
        if recInfos is None:
            raise RuntimeError(f"The folder {input_dir} does not contain SeeTrue recordings.")

    # make output dirs
    for i in range(len(recInfos)):
        if recInfos[i].proc_directory_name is None or not recInfos[i].proc_directory_name:
            recInfos[i].proc_directory_name = utils.make_fs_dirname(recInfos[i], output_dir)
        newDataDir = output_dir / recInfos[i].proc_directory_name
        if not newDataDir.is_dir():
            newDataDir.mkdir()

        # store rec info
        recInfos[i].store_as_json(newDataDir)

        # make sure there is a processing status file, and update it
        utils.get_recording_status(newDataDir, create_if_missing=True)
        utils.update_recording_status(newDataDir, utils.Task.Imported, utils.Status.Running)


    #### prep the data
    for rec_info in recInfos:
        newDataDir = output_dir / rec_info.proc_directory_name
        print(f'{newDataDir.name}...')
        print('  Getting camera calibration...')
        if cam_cal_file is not None:
            shutil.copyfile(str(cam_cal_file), str(newDataDir / 'calibration.xml'))
        else:
            print('    !! No camera calibration provided!')

        # NB: gaze data and scene video prep are intertwined, status messages are output inside this function
        gazeDf, frameTimestamps = copySeeTrueRecording(input_dir, newDataDir, rec_info)

        # write the gaze data to a csv file
        gazeDf.to_csv(str(newDataDir / 'gazeData.tsv'), sep='\t', na_rep='nan', float_format="%.8f")

        # also store frame timestamps
        frameTimestamps.to_csv(str(newDataDir / 'frameTimestamps.tsv'), sep='\t')

        # indicate import finished
        utils.update_recording_status(newDataDir, utils.Task.Imported, utils.Status.Finished)


def getRecordingInfo(inputDir):
    # returns None if not a recording directory
    recInfos = []

    # NB: a SeeTrue directory may contain multiple recordings

    # get recordings. These are indicated by the sequence number in both EyeData.csv and ScenePics folder names
    for r in inputDir.glob('*.csv'):
        if not str(r.name).startswith('EyeData'):
            # print(f"file {r.name} not recognized as a recording (wrong name, should start with 'EyeData'), skipping")
            continue

        # get sequence number
        _,recording = r.stem.split('_')

        # check there is a matching scenevideo
        sceneVidDir = r.parent / ('ScenePics_' + recording)
        if not sceneVidDir.is_dir():
            # print(f"folder {sceneVidDir} not found, meaning there is no scene video for this recording, skipping")
            continue

        recInfos.append(utils.Recording(source_directory=inputDir, eye_tracker=utils.EyeTracker.SeeTrue))
        recInfos[-1].participant = inputDir.name
        recInfos[-1].name = recording

    # should return None if no valid recordings found
    return recInfos if recInfos else None


def checkRecording(inputDir, recInfo):
    """
    This checks that the folder is properly prepared
    (i.e. the required BeGaze exports were run)
    """
    # check we have an exported gaze data file
    file = f'EyeData_{recInfo.name}.csv'
    if not (inputDir / file).is_file():
        return False

    # check we have an exported scene video
    file = f'ScenePics_{recInfo.name}'
    if not (inputDir / file).is_dir():
        return False

    return True


def copySeeTrueRecording(inputDir, outputDir, recInfo):
    """
    Copy the relevant files from the specified input dir to the specified output dirs
    """
    # get scene video dimensions by interrogating a frame in sceneVidDir
    sceneVidDir = inputDir / ('ScenePics_' + recInfo.name)
    frame = next(sceneVidDir.glob('*.jpeg'))
    h,w,_ = cv2.imread(str(frame)).shape

    # prep gaze data and get video frame timestamps from it
    print('  Prepping gaze data...')
    file = f'EyeData_{recInfo.name}.csv'
    gazeDf, frameTimestamps = formatGazeData(inputDir / file, [w,h])

    # make scene video
    print('  Prepping scene video...')
    # 1. see if there are frames missing, insert black frames if so
    frames = []
    for f in sceneVidDir.glob('*.jpeg'):
        _,fr = f.stem.split('_')
        frames.append(int(fr))

    # 2. see if framenumbers are as expected from the gaze data file
    # get average ifi
    ifi = np.mean(np.diff(frameTimestamps.index))
    # 2.1 remove frame timestamps that are before the first frame for which we have an image
    frameTimestamps=frameTimestamps.drop(frameTimestamps[frameTimestamps.frame_idx < frames[ 0]].index)
    # 2.2 remove frame timestamps that are beyond last frame for which we have an image
    frameTimestamps=frameTimestamps.drop(frameTimestamps[frameTimestamps.frame_idx > frames[-1]].index)
    # 2.3 add frame timestamps for images we have before first eye data
    if frames[ 0] < frameTimestamps.iloc[ 0,:].to_numpy()[0]:
        nFrames = frameTimestamps.iloc[ 0,:].to_numpy()[0] - frames[ 0]
        t0 = frameTimestamps.index[0]
        f0 = frameTimestamps.iloc[ 0,:].to_numpy()[0]
        for f in range(-1,-(nFrames+1),-1):
            frameTimestamps.loc[t0+f*ifi] = f0+f
        frameTimestamps = frameTimestamps.sort_index()
    # 2.4 add frame timestamps for images we have after last eye data
    if frames[-1] > frameTimestamps.iloc[-1,:].to_numpy()[0]:
        nFrames = frames[-1] - frameTimestamps.iloc[-1,:].to_numpy()[0]
        t0 = frameTimestamps.index[-1]
        f0 = frameTimestamps.iloc[-1,:].to_numpy()[0]
        for f in range(1,nFrames+1):
            frameTimestamps.loc[t0+f*ifi] = f0+f
        frameTimestamps = frameTimestamps.sort_index()
    # 2.5 check if holes, fill
    blackFrames = []
    frameDelta = np.diff(frames)
    if np.any(frameDelta>1):
        # frames images missing, add them (NB: if timestamps also missing, thats dealt with below)
        idxGaps = np.argwhere(frameDelta>1).flatten()     # idxGaps is last idx before each gap
        frGaps  = np.array(frames)[idxGaps].flatten()
        nFrames = frameDelta[idxGaps].flatten()
        for b,x in zip(frGaps+1,nFrames):
            for y in range(x-1):
                blackFrames.append(b+y)

        # make black frame
        blackIm = np.zeros((h,w,3), np.uint8)   # black image
        for f in blackFrames:
            # store black frame to file
            cv2.imwrite(str(sceneVidDir / 'frame_{:d}.jpeg'.format(f)),blackIm)
            frames.append(f)
        frames = sorted(frames)

    frameTsDelta = np.diff(frameTimestamps.frame_idx)
    if np.any(frameTsDelta>1):
        # frames missing from frametimestamps
        err

    if len(frames) != frameTimestamps.shape[0]:
        raise RuntimeError('Number of frames ({}) isn''t equal to number of frame timestamps ({}) and this couldnt be repaired'.format(len(frames),frameTimestamps.shape[0]))

    # 3. make into video
    framerate = "{:.4f}".format(1000./ifi)
    cmd_str = ' '.join(['ffmpeg', '-y', '-f', 'image2', '-framerate', framerate, '-start_number', str(frames[0]), '-i', '"'+str(sceneVidDir / 'frame_%d.jpeg')+'"', '"'+str(outputDir / 'worldCamera.mp4')+'"'])
    os.system(cmd_str)

    # attempt 2 that should allow correct VFR video files, but doesn't work with current MediaWriter
    # due to what i think is a bug: https://github.com/matham/ffpyplayer/issues/129.
    ## get which pixel format
    #codec    = ffpyplayer.tools.get_format_codec(fmt='mp4')
    #pix_fmt  = ffpyplayer.tools.get_best_pix_fmt('bgr24',ffpyplayer.tools.get_supported_pixfmts(codec))
    #fpsFrac  = Fraction(1000./ifi).limit_denominator(10000).as_integer_ratio()
    #fpsFrac  = tuple([x*10 for x in fpsFrac])
    ## scene video
    #out_opts = {'pix_fmt_in':'bgr24', 'pix_fmt_out':pix_fmt, 'width_in':w, 'height_in':h,'frame_rate':fpsFrac}
    #vidOut   = MediaWriter(str(outputDir / 'worldCamera.mp4'), [out_opts], overwrite=True)
    #t0       = frameTimestamps.index[0]
    #for i,f in enumerate(frames):
    #    frame = cv2.imread(str(sceneVidDir / 'frame_{:5d}.jpeg'.format(f)))
    #    img   = Image(plane_buffers=[frame.flatten().tobytes()], pix_fmt='bgr24', size=(int(w), int(h)))
    #    t = (frameTimestamps.index[i]-t0)/1000
    #    print(t, t/(fpsFrac[1]/fpsFrac[0]))
    #    vidOut.write_frame(img=img, pts=t)

    # delete the black frames we added, if any
    for f in blackFrames:
        if (sceneVidDir / 'frame_{:d}.jpeg'.format(f)).is_file():
            (sceneVidDir / 'frame_{:d}.jpeg'.format(f)).unlink(missing_ok=True)

    # 4. write data to file
    # fix up frame idxs and timestamps
    firstFrame = frameTimestamps['frame_idx'].min()

    # write the gaze data to a csv file
    gazeDf['frame_idx'] -= firstFrame

    # also store frame timestamps
    # this is what it should be if we get VFR video file writing above to work
    #frameTimestamps['frame_idx'] -= firstFrame
    #frameTimestamps=frameTimestamps.reset_index().set_index('frame_idx')
    #frameTimestamps['timestamp'] -= frameTimestamps['timestamp'].min()
    # instead now, get actual ts for each frame in written video as that is what we
    # have to work with. Note that these do not match gaze data ts, but code nowhere
    # assumes they do
    frameTimestamps = utils.getFrameTimestampsFromVideo(outputDir / 'worldCamera.mp4')

    return gazeDf, frameTimestamps



def formatGazeData(inputFile, sceneVideoDimensions):
    """
    load gazedata file
    format to get the gaze coordinates w.r.t. world camera, and timestamps for
    every frame of video

    Returns:
        - formatted dataframe with cols for timestamp, frame_idx, and gaze data
        - np array of frame timestamps
    """

    # convert the json file to pandas dataframe
    df = gazedata2df(inputFile, sceneVideoDimensions)

    # get time stamps for scene picture numbers
    frameTimestamps = pd.DataFrame(df['frame_idx'])

    # return the gaze data df and frame time stamps array
    return df, frameTimestamps


def gazedata2df(textFile, sceneVideoDimensions):
    """
    convert the gazedata file to a pandas dataframe
    """

    df = pd.read_table(textFile,sep=';',index_col=False)
    df.columns=df.columns.str.strip()

    # remove unneeded columns
    rem = [x for x in df.columns if x not in ['Frame number','Timestamp','Gazepoint X','Gazepoint Y','Scene picture number']]
    df = df.drop(columns=rem)

    # rename and reorder columns
    lookup = {'Timestamp': 'timestamp',
              'Scene picture number': 'frame_idx',
              'Gazepoint X': 'vid_gaze_pos_x',
              'Gazepoint Y': 'vid_gaze_pos_y',}
    df=df.rename(columns=lookup)
    # reorder
    idx = [lookup[k] for k in lookup if lookup[k] in df.columns]
    idx.extend([x for x in df.columns if x not in idx])   # append columns not in lookup
    df = df[idx]

    # set timestamps as index
    df = df.set_index('timestamp')

    # turn gaze locations into pixel data with origin in top-left
    df['vid_gaze_pos_x'] *= sceneVideoDimensions[0]
    df['vid_gaze_pos_y'] *= sceneVideoDimensions[1]

    # return the dataframe
    return df