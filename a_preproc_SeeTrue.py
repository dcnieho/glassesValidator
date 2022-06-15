"""
Cast raw SeeTrue data into common format.

Tested with Python 3.8, open CV 4.0.1

The output directory will contain:
    - frameTimestamps.tsv: frame number and corresponding timestamps for each frame in video
    - worldCamera.mp4: the video from the point-of-view scene camera on the glasses
    - gazeData.tsv: gaze data, where all 2D gaze coordinates are represented w/r/t the world camera
"""

import shutil
from pathlib import Path
import os
import cv2
import pandas as pd
import numpy as np

import utils


def preprocessData(inputDir, outputDir):
    if shutil.which('ffmpeg') is None:
        RuntimeError('ffmpeg must be on path to prep SeeTrue recording for processing with GlassesValidator')

    """
    Run all preprocessing steps on SeeTrue data
    """
    print('processing: {}'.format(inputDir.name))
    ### copy the raw data to the output directory
    print('Copying raw data...')
    copySeeTrueRecordings(inputDir, outputDir)


def copySeeTrueRecordings(inputDir, outputDir):
    """
    Copy the relevant files from the specified input dir to the specified output dirs
    NB: a SeeTrue directory may contain multiple recordings
    """
    participant = inputDir.name
    
    # get recordings. These are indicated by the sequence number in both EyeData.csv and ScenePics folder names
    for r in inputDir.glob('*.csv'):
        if not str(r.name).startswith('EyeData'):
            print("file {} not recognized as a recording (wrong name, should start with 'EyeData'), skipping".format(str(r.name)))
            continue

        # get sequence number
        _,recording = r.stem.split('_')

        # check there is a matching scenevideo
        sceneVidDir = r.parent / ('ScenePics_' + recording)
        if not sceneVidDir.is_dir():
            print("folder {} not found, meaning there is no scene video for this recording, skipping".format(str(sceneVidDir)))
            continue

        # get scene video dimensions by interrogating a frame in sceneVidDir
        frame = next(sceneVidDir.glob('*.jpeg'))
        h,w,_ = cv2.imread(str(frame)).shape

        outputDir = outputDir / ('SeeTrue_%s_%s' % (participant,recording))
        if not outputDir.is_dir():
            outputDir.mkdir()
        print('Input data will be copied to: {}'.format(outputDir))

        # prep gaze data and get video frame timestamps from it
        print('  Prepping gaze data...')
        gazeDf, frameTimestamps = formatGazeData(outputDir, r, [w,h])

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
        gazeDf.to_csv(str(outputDir / 'gazeData.tsv'), sep='\t', na_rep='nan', float_format="%.8f")

        # also store frame timestamps
        # this is what it should be if we get VFR video file writing above to work
        #frameTimestamps['frame_idx'] -= firstFrame
        #frameTimestamps=frameTimestamps.reset_index().set_index('frame_idx')
        #frameTimestamps['timestamp'] -= frameTimestamps['timestamp'].min()
        # instead now, get actual ts for each frame in written video as that is what we
        # have to work with. Note that these do not match gaze data ts, but code nowhere
        # assumes they do
        frameTimestamps = utils.getFrameTimestampsFromVideo(outputDir / 'worldCamera.mp4')
        
        frameTimestamps.to_csv(str(outputDir / 'frameTimestamps.tsv'), sep='\t')



def formatGazeData(outputDir, inputFile, sceneVideoDimensions):
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


def gazedata2df(textFile,sceneVideoDimensions):
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


if __name__ == '__main__':
    basePath = Path(__file__).resolve().parent / 'data'
    inBasePath = basePath / 'SeeTrue'
    outBasePath = basePath / 'preprocced'
    if not outBasePath.is_dir():
        outBasePath.mkdir()
    for d in inBasePath.iterdir():
        if d.is_dir():
            preprocessData(d,outBasePath)
