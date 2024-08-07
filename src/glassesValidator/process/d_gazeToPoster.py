import pathlib
import threading

from glassesTools import annotation, gaze_headref, gaze_worldref, ocv, plane, recording, worldgaze_gui
from glassesTools.video_gui import GUI

from .. import config
from .. import utils


def process(working_dir, config_dir=None, show_visualization=False, show_poster=True, show_only_intervals=True):
    # if show_visualization, each frame is shown in a viewer, overlaid with info about detected planes and projected gaze
    # if show_poster, gaze in poster space is also drawn in a separate window
    # if show_only_intervals, only the coded validation episodes (if available) are shown in the viewer while the rest of the scene video is skipped past
    working_dir  = pathlib.Path(working_dir)
    if config_dir is not None:
        config_dir = pathlib.Path(config_dir)

    print('processing: {}'.format(working_dir.name))

    # if we need gui, we run processing in a separate thread (GUI needs to be on the main thread for OSX, see https://github.com/pthom/hello_imgui/issues/33)
    if show_visualization:
        gui = GUI(use_thread = False)
        frame_win_id = gui.add_window(working_dir.name)
        gui.set_show_controls(True)
        gui.set_show_play_percentage(True)
        gui.set_show_annotation_label(False)

        proc_thread = threading.Thread(target=do_the_work, args=(working_dir, config_dir, gui, frame_win_id, show_poster, show_only_intervals))
        proc_thread.start()
        gui.start()
        proc_thread.join()
    else:
        do_the_work(working_dir, config_dir, None, None, False, False)


def do_the_work(working_dir, config_dir, gui, frame_win_id, show_poster, show_only_intervals):
    utils.update_recording_status(working_dir, utils.Task.Gaze_Tranformed_To_Poster, utils.Status.Running)

    # get camera calibration info
    cameraParams      = ocv.CameraParams.read_from_file(working_dir / "calibration.xml")

    # get interval coded to be analyzed, if any
    analyzeFrames   = utils.readMarkerIntervalsFile(working_dir / "markerInterval.tsv")

    # Read gaze data
    head_gazes  = gaze_headref.read_dict_from_file(working_dir / 'gazeData.tsv', episodes=analyzeFrames if not gui or show_only_intervals else None)[0]

    # Read camera pose w.r.t. poster
    poses       = plane.read_dict_from_file(working_dir / 'posterPose.tsv', episodes=analyzeFrames if not gui or show_only_intervals else None)

    # transform
    plane_gazes = gaze_worldref.from_head(poses, head_gazes, cameraParams)

    # store to file
    gaze_worldref.write_dict_to_file(plane_gazes, working_dir / 'gazePosterPos.tsv', skip_missing=True)

    utils.update_recording_status(working_dir, utils.Task.Gaze_Tranformed_To_Poster, utils.Status.Finished)


    # done if no visualization wanted
    if gui is None:
        return

    in_video = recording.Recording.load_from_json(working_dir).get_scene_video_path()
    validationSetup = config.get_validation_setup(config_dir)
    poster          = config.poster.Poster(config_dir, validationSetup)
    worldgaze_gui.show_visualization(
        in_video, working_dir / 'frameTimestamps.tsv', working_dir / "calibration.xml",
        {'poster': poster}, {'poster': poses}, head_gazes, {'poster': plane_gazes}, {annotation.Event.Validate: analyzeFrames},
        gui, frame_win_id, show_poster, show_only_intervals, 8
    )