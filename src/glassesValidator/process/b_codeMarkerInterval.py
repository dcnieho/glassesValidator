import pathlib
import numpy as np
from imgui_bundle import imgui
import cv2

from ffpyplayer.player import MediaPlayer

import sys
isMacOS = sys.platform.startswith("darwin")
if isMacOS:
    import AppKit

from glassesTools import annotation, gaze_headref, gaze_worldref, naming, ocv, plane, propagating_thread, recording, timestamps
from glassesTools.gui import video_player

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


def process(working_dir, config_dir=None, show_poster=False):
    # if show_poster, also draw poster with gaze overlaid on it (if available)
    working_dir = pathlib.Path(working_dir)
    if config_dir is not None:
        config_dir = pathlib.Path(config_dir)

    print('processing: {}'.format(working_dir.name))

    # We run processing in a separate thread (GUI needs to be on the main thread for OSX, see https://github.com/pthom/hello_imgui/issues/33)
    gui = video_player.GUI(use_thread = False)
    gui.add_window(working_dir.name)

    proc_thread = propagating_thread.PropagatingThread(target=do_the_work, args=(working_dir, config_dir, gui, show_poster), cleanup_fun=gui.stop)
    proc_thread.start()
    gui.start()
    proc_thread.join()


def do_the_work(working_dir, config_dir, gui: video_player.GUI, show_poster):
    utils.update_recording_status(working_dir, utils.Task.Coded, utils.Status.Running)

    # get info about recording
    recInfo = recording.Recording.load_from_json(working_dir)

    # open file with information about Aruco marker and Gaze target locations
    validationSetup = config.get_validation_setup(config_dir)
    poster = config.poster.Poster(config_dir, validationSetup)

    # Read gaze data
    gazes = gaze_headref.read_dict_from_file(working_dir / naming.gaze_data_fname)[0]

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
    cameraParams= ocv.CameraParams.read_from_file(working_dir / naming.scene_camera_calibration_fname)
    hasCamCal   = cameraParams.has_intrinsics()

    # get interval coded to be analyzed, if available
    analyzeFrames = utils.readMarkerIntervalsFile(working_dir / "markerInterval.tsv")
    if analyzeFrames is None:
        analyzeFrames = []
    else:
        # flatten
        analyzeFrames = [i for iv in analyzeFrames for i in iv]
    episodes = {annotation.Event.Validate: analyzeFrames}

    # set up video playback
    # 1. window for scene video is already set up
    # 2. if wanted and available, second OpenCV window for poster with gaze on that plane
    show_poster &= hasPosterGaze  # no poster if we don't have poster gaze, it'd be empty and pointless
    if show_poster:
        poster_win_id = gui.add_window("poster")
    # 3. timestamp info for relating audio to video frames
    video_ts = timestamps.VideoTimestamps( working_dir / naming.frame_timestamps_fname )
    # 4. mediaplayer for the actual video playback, with sound if available
    inVideo = recInfo.get_scene_video_path()
    ff_opts = {'volume': 1., 'sync': 'audio', 'framedrop': True}
    player = MediaPlayer(str(inVideo), ff_opts=ff_opts)
    gui.set_playing(True)

    # set up annotation GUI
    gui.set_allow_pause(True)
    gui.set_allow_seek(True)
    gui.set_allow_timeline_zoom(True)
    gui.set_show_controls(True, gui.main_window_id)
    gui.set_allow_annotate({annotation.Event.Validate}, {annotation.Event.Validate: imgui.Key.v})
    gui.set_show_timeline(True, video_ts, episodes, gui.main_window_id)
    gui.set_show_annotation_label(False, gui.main_window_id)
    gui.set_show_action_tooltip(True, gui.main_window_id)

    # show
    subPixelFac = 8   # for sub-pixel positioning
    armLength = poster.marker_size/2 # arms of axis are half a marker long
    hasRequestedFocus = not isMacOS # False only if on Mac OS, else True since its a no-op
    should_exit = False
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
                poses[frame_idx].draw_frame_axis(frame, cameraParams, armLength, 3, subPixelFac)

            # if have gaze for this frame, draw it
            # NB: usually have multiple gaze samples for a video frame, draw one
            if frame_idx in gazes:
                gazes[frame_idx][0].draw(frame, cameraParams, subPixelFac)

            # if have gaze in world info, draw it too (also only first)
            if hasPosterGaze and frame_idx in gazesPoster:
                gazesPoster[frame_idx][0].draw_on_world_video(frame, cameraParams, subPixelFac, None if not frame_idx in poses else poses[frame_idx])
                if show_poster:
                    gazesPoster[frame_idx][0].draw_on_plane(refImg, poster, subPixelFac)

            if frame is not None:
                gui.update_image(frame, pts, frame_idx, window_id=gui.main_window_id)

            if show_poster:
                gui.update_image(refImg, pts, frame_idx, window_id=poster_win_id)

        if not hasRequestedFocus:
            AppKit.NSApplication.sharedApplication().activateIgnoringOtherApps_(1)
            hasRequestedFocus = True

        requests = gui.get_requests()
        for r,p in requests:
            match r:
                case 'toggle_pause':
                    player.toggle_pause()
                    if not player.get_pause():
                        player.seek(0)  # needed to get frames rolling in again, apparently, after seeking occurred while paused
                    gui.set_playing(not player.get_pause())
                case 'seek':
                    player.seek(p, relative=False)
                case 'delta_frame':
                    new_ts = video_ts.get_timestamp(frame_idx+p)
                    if new_ts != -1.:
                        step = (new_ts-video_ts.get_timestamp(max(0,frame_idx)))/1000
                        player.seek(pts+step, relative=False)
                case 'delta_time':
                    player.seek(pts+p, relative=False)
                case 'add_coding':
                    event,frame_idx = p
                    if frame_idx not in episodes[event]:
                        episodes[event].append(frame_idx)
                        episodes[event].sort()
                        gui.notify_annotations_changed()
                case 'delete_coding':
                    event,frame_idxs = p
                    episodes[event] = [i for i in episodes[event] if i not in frame_idxs]
                    gui.notify_annotations_changed()
                case 'exit':
                    should_exit = True
                    break
        if should_exit:
            break

    player.close_player()
    gui.stop()

    # store coded intervals to file
    utils.writeMarkerIntervalsFile(working_dir / 'markerInterval.tsv', episodes[annotation.Event.Validate])

    utils.update_recording_status(working_dir, utils.Task.Coded, utils.Status.Finished)
