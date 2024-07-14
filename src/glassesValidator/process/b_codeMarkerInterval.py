import pathlib
import numpy as np

import cv2
import csv
import threading

from ffpyplayer.player import MediaPlayer

import sys
isMacOS = sys.platform.startswith("darwin")
if isMacOS:
    import AppKit

from glassesTools import drawing, gaze_headref, gaze_worldref, ocv, plane, recording, timestamps
from glassesTools.video_gui import GUI, generic_tooltip_drawer

from .. import config
from .. import utils

# This script shows a video player that is used to indicate the interval(s)
# during which the poster should be found in the video and in later
# steps data quality computed. So this interval/these intervals would for
# instance be the exact interval during which the subject performs the
# validation task.
# This script can be run directly on recordings converted to the common format,
# but output from steps c_detectMarkers and d_gazeToPoster
# (which can be run before this script, they will just process the whole video)
# will also be shown if available.


stopAllProcessing = False
def process(working_dir, config_dir=None, show_poster=False):
    # if show_poster, also draw poster with gaze overlaid on it (if available)
    working_dir = pathlib.Path(working_dir)
    if config_dir is not None:
        config_dir = pathlib.Path(config_dir)

    print('processing: {}'.format(working_dir.name))

    # We run processing in a separate thread (GUI needs to be on the main thread for OSX, see https://github.com/pthom/hello_imgui/issues/33)
    gui = GUI(use_thread = False)
    key_tooltip = {
        "h": "Back 1 s, shift+H: back 10 s",
        "l": "Forward 1 s, shift+L: forward 10 s",
        "j": "Back 1 frame",
        "k": "Forward 1 frame",
        "p": "Pause or resume playback",
        "f": "Mark frame",
        "d": "Delete frame or current interval",
        "s": "Seek to start of next interval, shift+S seek to start of previous interval",
        "e": "Seek to end of next interval, shift+E seek to end of previous interval",
        "q": "Quit",
        'n': 'Next'
    }
    gui.set_interesting_keys(list(key_tooltip.keys()))
    gui.register_draw_callback('status',lambda: generic_tooltip_drawer(key_tooltip))
    main_win_id = gui.add_window(working_dir.name)

    proc_thread = threading.Thread(target=do_the_work, args=(working_dir, config_dir, gui, main_win_id, show_poster))
    proc_thread.start()
    gui.start()
    proc_thread.join()
    return stopAllProcessing


def do_the_work(working_dir, config_dir, gui, main_win_id, show_poster):
    global stopAllProcessing

    utils.update_recording_status(working_dir, utils.Task.Coded, utils.Status.Running)

    # get info about recording
    recInfo = recording.Recording.load_from_json(working_dir)

    # open file with information about Aruco marker and Gaze target locations
    validationSetup = config.get_validation_setup(config_dir)
    poster = config.poster.Poster(config_dir, validationSetup)

    # Read gaze data
    gazes = gaze_headref.read_dict_from_file(working_dir / 'gazeData.tsv')[0]

    # Read pose of poster, if available
    hasPosterPose = False
    if (working_dir / 'posterPose.tsv').is_file():
        try:
            poses = plane.read_dict_from_file(working_dir / 'posterPose.tsv')
            hasPosterPose = True
        except:
            # ignore when file can't be read or is empty
            pass

    # Read gaze on poster data, if available
    hasPosterGaze = False
    if (working_dir / 'gazePosterPos.tsv').is_file():
        try:
            gazesPoster = gaze_worldref.read_dict_from_file(working_dir / 'gazePosterPos.tsv')
            hasPosterGaze = True
        except:
            # ignore when file can't be read or is empty
            pass

    # get camera calibration info
    cameraParams= ocv.CameraParams.readFromFile(working_dir / "calibration.xml")
    hasCamCal   = cameraParams.has_intrinsics()

    # get interval coded to be analyzed, if available
    analyzeFrames = utils.readMarkerIntervalsFile(working_dir / "markerInterval.tsv")
    if analyzeFrames is None:
        analyzeFrames = []
    else:
        # flatten
        analyzeFrames = [i for iv in analyzeFrames for i in iv]

    # set up video playback
    # 1. window for scene video is already set up
    # 2. if wanted and available, second OpenCV window for poster with gaze on that plane
    show_poster &= hasPosterGaze  # no poster if we don't have poster gaze, it'd be empty and pointless
    if show_poster:
        poster_win_id = gui.add_window("poster")
    # 3. timestamp info for relating audio to video frames
    video_ts = timestamps.VideoTimestamps( working_dir / 'frameTimestamps.tsv' )
    # 4. mediaplayer for the actual video playback, with sound if available
    inVideo = recInfo.get_scene_video_path()
    ff_opts = {'volume': 1., 'sync': 'audio', 'framedrop': True}
    player = MediaPlayer(str(inVideo), ff_opts=ff_opts)

    # show
    subPixelFac = 8   # for sub-pixel positioning
    armLength = poster.marker_size/2 # arms of axis are half a marker long
    stopAllProcessing = False
    hasRequestedFocus = not isMacOS # False only if on Mac OS, else True since its a no-op
    while True:
        frame, val = player.get_frame(force_refresh=True)
        if val == 'eof':
            player.toggle_pause()
        if frame is not None:
            image, pts = frame
            width, height = image.get_size()
            frame = cv2.cvtColor(np.asarray(image.to_memoryview()[0]).reshape((height,width,3)), cv2.COLOR_RGB2BGR)
            del image

        if frame is not None:
            # the audio is my shepherd and nothing shall I lack :-)
            frame_idx = video_ts.find_frame(pts*1000)  # pts is in seconds, our frame timestamps are in ms
            if show_poster:
                refImg = poster.get_ref_image(400)

            # if we have poster pose, draw poster origin on video
            if hasPosterPose and frame_idx in poses and hasCamCal:
                drawing.openCVFrameAxis(frame, cameraParams.camera_mtx, cameraParams.distort_coeffs, poses[frame_idx].pose_R_vec, poses[frame_idx].pose_T_vec, armLength, 3, subPixelFac)

            # if have gaze for this frame, draw it
            # NB: usually have multiple gaze samples for a video frame, draw one
            if frame_idx in gazes:
                gazes[frame_idx][0].draw(frame, cameraParams, subPixelFac)

            # if have gaze in world info, draw it too (also only first)
            if hasPosterGaze and frame_idx in gazesPoster:
                if hasCamCal:
                    gazesPoster[frame_idx][0].drawOnWorldVideo(frame, cameraParams, subPixelFac)
                if show_poster:
                    gazesPoster[frame_idx][0].drawOnPlane(refImg, poster, subPixelFac)

            analysisIntervalIdx = None
            analysisLbl = ''
            for f in range(0,len(analyzeFrames)-1,2):   # -1 to make sure we don't try incomplete intervals
                if frame_idx>=analyzeFrames[f] and frame_idx<=analyzeFrames[f+1]:
                    analysisIntervalIdx = f
                if len(analysisLbl)>0:
                    analysisLbl += ', '
                analysisLbl += '{} -- {}'.format(*analyzeFrames[f:f+2])
            if len(analyzeFrames)%2:
                if len(analyzeFrames)==1:
                    analysisLbl +=   '{} -- xx'.format(analyzeFrames[-1])
                else:
                    analysisLbl += ', {} -- xx'.format(analyzeFrames[-1])

            # annotate what frame we're on
            frameClr = (0,0,255) if analysisIntervalIdx is not None else (0,0,0)
            # annotate analysis intervals
            textSize,baseline = cv2.getTextSize(analysisLbl,cv2.FONT_HERSHEY_PLAIN,2,2)
            cv2.rectangle(frame,(0,textSize[1]+baseline+5),(textSize[0]+5,0), frameClr, -1)
            cv2.putText(frame, (analysisLbl), (0, textSize[1]+baseline), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255),2)

            if frame is not None:
                gui.update_image(frame, pts, frame_idx, window_id = main_win_id)

            if show_poster:
                gui.update_image(refImg, pts, frame_idx, window_id = poster_win_id)

        if not hasRequestedFocus:
            AppKit.NSApplication.sharedApplication().activateIgnoringOtherApps_(1)
            hasRequestedFocus = True

        keys = gui.get_key_presses()
        # seek: don't ask me why, but relative seeking works best for backward,
        # and seeking to absolute pts best for forward seeking.
        if 'j' in keys:
            step = (video_ts.get_timestamp(frame_idx)-video_ts.get_timestamp(max(0,frame_idx-1)))/1000
            player.seek(-step)                              # back one frame
        if 'k' in keys:
            nextTs = video_ts.get_timestamp(frame_idx+1)
            if nextTs != -1.:
                step = (nextTs-video_ts.get_timestamp(max(0,frame_idx)))/1000
                player.seek(pts+step, relative=False)       # forward one frame
        if 'h' in keys or 'H' in keys:
            step = 1 if 'h' in keys else 10
            player.seek(-step)                              # back one or ten seconds
        if 'l' in keys or 'L' in keys:
            step = 1 if 'l' in keys else 10
            player.seek(pts+step, relative=False)           # forward one or ten seconds

        if 'p' in keys:
            player.toggle_pause()
            if not player.get_pause():
                player.seek(0)  # needed to get frames rolling in again, apparently, after seeking occurred while paused

        if 'f' in keys:
            if not frame_idx in analyzeFrames:
                analyzeFrames.append(frame_idx)
                analyzeFrames.sort()
        if 'd' in keys:
            if frame_idx in analyzeFrames:
                # delete this one marker from analysis frames
                analyzeFrames.remove(frame_idx)
            elif analysisIntervalIdx is not None:
                # delete current interval from analysis frames
                del analyzeFrames[analysisIntervalIdx:analysisIntervalIdx+2]

        if 's' in keys or 'S' in keys:
            if (analysisIntervalIdx is not None) and (frame_idx!=analyzeFrames[analysisIntervalIdx]):
                # seek to start of current interval
                ts = video_ts.get_timestamp(analyzeFrames[analysisIntervalIdx])
                player.seek(ts/1000, relative=False)
            else:
                # seek to start of next or previous analysis interval, if any
                forward = 's' in keys
                if forward:
                    idx = next((x for x in analyzeFrames[ 0:(len(analyzeFrames)//2)*2:2 ] if x>frame_idx), None) # slice gets starts of all whole intervals
                else:
                    idx = next((x for x in analyzeFrames[(len(analyzeFrames)//2)*2-2::-2] if x<frame_idx), None) # slice gets starts of all whole intervals in reverse order
                if idx is not None:
                    ts = video_ts.get_timestamp(idx)
                    player.seek(ts/1000, relative=False)
        if 'e' in keys or 'E' in keys:
            if (analysisIntervalIdx is not None) and (frame_idx!=analyzeFrames[analysisIntervalIdx+1]):
                # seek to end of current interval
                ts = video_ts.get_timestamp(analyzeFrames[analysisIntervalIdx+1])
                player.seek(ts/1000, relative=False)
            else:
                # seek to end of next or previous analysis interval, if any
                forward = 'e' in keys
                if forward:
                    idx = next((x for x in analyzeFrames[1:(len(analyzeFrames)//2)*2:2] if x>frame_idx), None) # slice gets ends of all whole intervals
                else:
                    idx = next((x for x in analyzeFrames[(len(analyzeFrames)//2)*2::-2] if x<frame_idx), None) # slice gets ends of all whole intervals in reverse order
                if idx is not None:
                    ts = video_ts.get_timestamp(idx)
                    player.seek(ts/1000, relative=False)

        if 'q' in keys:
            # quit fully
            stopAllProcessing = True
            break
        if 'n' in keys:
            # goto next
            break

        closed, = gui.get_state()
        if closed:
            stopAllProcessing = True
            break

    player.close_player()
    gui.stop()

    # store coded intervals to file
    utils.writeMarkerIntervalsFile(working_dir / 'markerInterval.tsv', analyzeFrames)

    utils.update_recording_status(working_dir, utils.Task.Coded, utils.Status.Finished)

    return stopAllProcessing
