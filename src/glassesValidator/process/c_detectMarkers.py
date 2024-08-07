import pathlib
import threading

from glassesTools import annotation, aruco, plane, recording
from glassesTools.video_gui import GUI

from .. import config
from .. import utils


def process(working_dir, config_dir=None, show_visualization=False, show_rejected_markers=False):
    # if show_visualization, each frame is shown in a viewer, overlaid with info about detected markers and poster
    # if show_rejected_markers, rejected ArUco marker candidates are also shown in the viewer. Possibly useful for debug
    working_dir  = pathlib.Path(working_dir)
    if config_dir is not None:
        config_dir = pathlib.Path(config_dir)

    print('processing: {}'.format(working_dir.name))

    # if we need gui, we run processing in a separate thread (GUI needs to be on the main thread for OSX, see https://github.com/pthom/hello_imgui/issues/33)
    if show_visualization:
        gui = GUI(use_thread = False)
        gui.add_window(working_dir.name)
        gui.set_show_controls(True)
        gui.set_show_play_percentage(True)
        gui.set_show_annotation_label(False)

        proc_thread = threading.Thread(target=do_the_work, args=(working_dir, config_dir, gui, show_rejected_markers))
        proc_thread.start()
        gui.start()
        proc_thread.join()
    else:
        do_the_work(working_dir, config_dir, None, False)


def do_the_work(working_dir, config_dir, gui, show_rejected_markers):
    utils.update_recording_status(working_dir, utils.Task.Markers_Detected, utils.Status.Running)

    # get info about recording
    recInfo = recording.Recording.load_from_json(working_dir)

    # open file with information about Aruco marker and Gaze target locations
    validationSetup = config.get_validation_setup(config_dir)
    # get info about markers on our poster
    poster          = config.poster.Poster(config_dir, validationSetup)

    # get interval(s) coded to be analyzed, if any
    analyzeFrames   = utils.readMarkerIntervalsFile(working_dir / "markerInterval.tsv")

    # get video file to process
    in_video = recInfo.get_scene_video_path()

    # set up pose estimator and run it
    estimator = aruco.PoseEstimator(in_video, working_dir / "frameTimestamps.tsv", working_dir / "calibration.xml")
    estimator.add_plane('validate',
                        {'plane': poster, 'aruco_params': {'markerBorderBits': validationSetup['markerBorderBits']}, 'min_num_markers': validationSetup['minNumMarkers']},
                        analyzeFrames)
    estimator.attach_gui(gui, {annotation.Event.Validate: [i for iv in analyzeFrames for i in iv]})
    if gui is not None:
        estimator.show_rejected_markers = show_rejected_markers
    poses, _, _ = estimator.process_video()

    plane.write_list_to_file(poses['validate'], working_dir/'posterPose.tsv', skip_failed=True)

    utils.update_recording_status(working_dir, utils.Task.Markers_Detected, utils.Status.Finished)