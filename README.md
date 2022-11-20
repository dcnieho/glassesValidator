﻿# GlassesValidator
Tool for automatic determination of data quality (accuracy and precision) of wearable eye tracker recordings.

Please cite:
Niehorster, D.C., Hessels, R.S., Benjamins, J.S., Nyström, M. and Hooge, I.T.C. (submitted).
GlassesValidator: A data quality tool for eye tracking glasses.

# How to acquire
The glassesValidator is available from `https://github.com/dcnieho/glassesValidator`, and supports Windows, MacOS and Linux.
For Windows ~~and MacOS users~~ who wish to use the glassesValidator GUI, the easiest way to acquire glassesValidator is to [download
a standalone executable](https://github.com/dcnieho/glassesValidator/releases/latest).

For users who wish to use glassesValidator in their Python code, ~~the package can be installed directly from Python using the command
`python -m pip install glassesValidator`. Should that fail,~~ this repository is pip-installable as well:
`python -m pip install git+https://github.com/dcnieho/glassesValidator.git#egg=glassesValidator`.

# Usage
The glassesValidator validation procedure consists of two parts, 1) a poster that is used during a recording, and 2) Python software
for offline processing of the recording to estimate data quality measures.

## glassesValidator projects
The glassesValidator GUI organizes recordings into a project folder. Each recording to be processed is imported into this project folder
and all further processing is done inside it. The source directories containing the original recordings remain untouched when running
glassesValidator. The glassesValidator project folder can furthermore contain a folder specifying the configuration of the project.
Such a configuration should be made if you used a poster different from the default (if no configuration folder is present, the default
settings are automatically used), and can be deployed with the `Deploy config` button in the GUI, or the
`glassesValidator.config.deploy_validation_config()` call from Python.


## The poster
The default poster is available 1) [here](/src/glassesValidator/config/markerBoard/board.pdf), 2) from the GUI with the `Get poster
pdf` button, and 3) can also be acquired from a Python script by calling
`glassesValidator.config.markerBoard.deploy_default_pdf()`.
The default poster should be printed at A2 size, as defined
in the pdf file, and is designed to cover a reasonable field of view when participants view it at armslength (i.e., 20 x 17.5 deg
at 60 cm). In order to check that the poster was printed at the correct scale, one should measure the sides of the ArUco markers.
We strongly recommend performing this check because printers may not be calibrated. In the case of the default glassesValidator
poster, each ArUco marker should have sides that are 4.19 cm long. If the poster was printed at the wrong scale, one must adapt
the glassesValidator configuration to match the size and position of the ArUco markers and fixation targets on your poster ([see
"Customizing the poster" below](#customizing-the-poster)).

### Customizing the poster
The poster pdf file is generated by the [LaTeX file in the same folder as the pdf](/src/glassesValidator/config/markerBoard/board.tex).
Its looks are defined in the files in the [config folder](/src/glassesValidator/config). As described above, this configuration can be
deployed and then edited. The edited configuration can both be used to generate a new poster with LaTeX and for performing the
data quality calculations.
Specifically, the files [`markerPositions.csv`](/src/glassesValidator/config/markerPositions.csv) and
[`targetPositions.csv`](/src/glassesValidator/config/targetPositions.csv) define where the ArUco markers and fixation targets
(respectively) are placed on the poster. Each coordinate in these files is for the center of the marker or gaze target and the origin
(0,0) is in the bottom left of the poster. The [`validationSetup.txt` configuration file](/src/glassesValidator/config/validationSetup.txt)
contains the following settings for the poster:

|setting|description|
| --- | --- |
|`distance`|viewing distance in cm, used to convert coordinates and sizes in degrees to cm. Only used when `mode` is `deg`.|
|`mode`|`cm` or `deg`. Sets the unit for the `markerSide` and `targetDiameter` below as well as for interpreting the coordinates in the marker and target position files.|
|`markerSide`|Size of ArUco markers. In cm or deg, see `mode` setting.|
|`markerPosFile`|File in the config folder where the markers to draw are specified, e.g. `markerPositions.csv`.|
|`targetPosFile`|File in the config folder where the targets to draw are specified, e.g. `targetPositions.csv`.|
|`targetType`|Type of targer to draw, can be `Tobii` (the calibration marker used by Tobii) or `Thaler` (layout ABC in the lower panel of Fig. 1 in [Thaler et al., 2013](https://doi.org/10.1016/j.visres.2012.10.012)).|
|`targetDiameter`|Diameter of the gaze target. In cm or deg, see `mode` setting. Ignored if `targetType` is `Tobii` and `useExactTobiiSize` is `1`.|
|`useExactTobiiSize`|`0` or `1`. If `1`, the gaze targets have the exact dimensions of a Tobii calibration marker (though possibly different colors). I.e., the `targetDiameter` parameter is ignored. Only used if `targetType` is `Tobii`.|
|`showGrid`|`0` or `1`. If `1`, a `gridCols` x `gridRows` grid is drawn behind the markers and gaze targets. The size of each grid cell is 1 cm if `mode` is cm, or 1 degree if `mode` is `deg`.|
|`gridCols`|Number of grid columns to draw if `showGrid` is `1`.|
|`gridRows`|Number of grid rows to draw if `showGrid` is `1`.|
|`showAnnotations`|`0` or `1`. If `1`, text annotations informing about the size of grid cells and markers is printed on the poster below the marker arrangement.|

To check your custom configuration, you can generate a poster pdf using [the steps below](#steps-for-making-your-own-poster). Furthermore,
a png image showing the poster will be generated in the configuration folder when any of glassesValidator's processing steps are run.

### Steps for making your own poster
1. Deploy the default configuration using the `Deploy config` button in the GUI, or the
   `glassesValidator.config.deploy_validation_config()` call from Python.
2. Edit the `validationSetup.txt` configuration file and the `markerPositions.csv` and `targetPositions.csv` files in the
   configuration folder to design the layout and look of the poster that you want.
3. Compile the `markerBoard/board.tex` LaTeX file with `pdfTex`, such as provided in the [TeX Live distribution](https://www.tug.org/texlive/).
4. Done, you should now have a pdf file with the poster as you defined.


## The GUI
![Glasses viewer screenshot](/.github/images/screenshot.png?raw=true)
Click the screenshot to see a full-size version.

The simplest way to use glassesValidator is by means of its GUI, see above. The full workflow can be performed in the GUI.
Specifically, it supports:
- GlassesValidator project management, such as making new projects or opening them
- Importing recordings into a glassesValidator project. Recordings can be found by glassesValidator by drag-dropping one or multiple directories onto
  the GUI or by clicking the `Add recordings` button (not shown) and selecting one or multiple directories. The selected directories and all their subdirectories are then searched for recordings,
  and the user can select which of the found recordings should be added to the glassesValidator import. Once listed in the GUI, the import action can then be started, which copies over data from the selected recording to the
  glassesValidator project folder, and transforms it to a common format.
- Showing a listing of the recordings in the project and several properties about them, where available. these properties currently are:
    - Eye tracker type (e.g. Pupil Invisible or Tobii Glasses 2)
    - Status, indicating whether a recording has been imported, coded, analyzed)
    - Recording name
    - Participant name
    - Project name
    - Recording duration
    - Recording start time
    - Working directory for the recording inside the glassesValidator project
    - Original source directory from which the recording was imported
    - Firmware version
    - Glasses serial number
    - Recording unit serial number
    - Recording software version
    - Scene camera serial number

  The GUI can be configured to show any combination of these columns in any order, and the listing can be sorted by any of these columns.
  The listing can furthermore be filtered to show only a subset of recordings by search string, eye tracker type and status.
- Annotating recordings to indicate the episode(s) in the recording that contain a validation using a separate GUI.
- Calculating data quality (accuracy and precision) for recordings.
- Exporting the data quality values of multiple recordings to a summary tab-separated spreadsheet file.
- Deploying the default configuration to a glassesValidator project so that it can be edited ([see "Customizing the poster" above](#customizing-the-poster)).

### Advanced settings
Several advanced settings can be made in the right bar of the glassesValidator GUI. For standard use, these are not of interest. Each setting is explained by means of a help text that pops up when hovering
over the setting. Nonetheless, the specific group of advanced settings for configuring what type of data quality to compute is discussed here in more detail.

glassesValidator can compute data quality in multiple ways. 
When deciding how to determine accuracy and precision, several decisions have to be made. By default the most appropriate decisions for standard use are selected and most users can skip this section, but another configuration might be more suited for some advanced use cases.
The following decisions are made:
1. Determining location of the participant:
   1. The researcher can assume a fixed viewing distance and provide this to glassesValidator in the project configuration.
   2. If the scene camera is calibrated (i.e. its properties such as focal length and distortion parameters have been estimated by a calibration procedure), it is possible to use the array of ArUco markers to estimate the position of the participant relative to the poster at each time point during a validation.

   When a scene camera calibration is available, glassesValidator will by default use mode ii, otherwise mode i will be used. Five of the six wearable eye trackers supported by glassesValidator provide the calibration of their scene camera, and glassesValidator will by default use this calibration for these eye
   trackers (mode ii). Currently, only the SeeTrue does not provide a camera calibration, and glassesValidator therefore by default uses an assumed fixed viewing distance for this eye tracker.

2) Transforming gaze positions from the scene camera reference frame to positions on the validation poster:
   1. Performed by means of homography.
   2. Performed using recovered camera pose and gaze direction vector, by means of intersection of gaze vector with the validation poster plane.

   Mode ii is used by default. However, like for decision 1, mode ii requires that a camera calibration is available. If a camera calibration is not available, mode i will be used instead.

3) Which data is used for determining gaze position on the validation poster:
   1. The gaze position in the scene camera image.
   2. Gaze direction vectors in a head reference frame.

   When operating in mode i, the eye tracker's estimate of the (binocular) gaze point in the scene camera image is used. This is the appropriate choice for most wearable eye tracking research, as it is this gaze point that is normally used for further analysis. However, in some settings and when the eye tracker provides gaze direction vectors for the individual eyes along with their origin, a different mode of operation may be more appropriate. Specifically, when using the wearable eye tracker's gaze vectors instead of the gaze point in the scene video in their analysis, the researcher should compute the accuracy and precision of these gaze vectors.

Altogether, combining these decisions, the following six types of data quality can be calculated using glassesValidator. NB: All API types are members of the `enum.Enum` `glassesValidator.process.DataQualityType`

|Name in GUI|Name in API|description|
| --- | --- | --- |
|Homography + view distance|`viewdist_vidpos_homography`|Use a homography tranformation to map gaze position from the scene video to the validation poster, , and use an assumed viewing distance (see the project's configuration) to compute data quality measures in degrees with respect to the scene camera. *Default mode when no camera calibration is available.*|
|Homography + pose|`pose_vidpos_homography`|Use a homography tranformation to map gaze position from the scene video to the validation poster, and use the determined pose of the scene camera (requires a calibrated camera) to compute data quality measures in degrees with respect to the scene camera.|
|Video ray + pose|`pose_vidpos_ray`|Use camera calibration to turn gaze position from the scene video into a direction vector, and determine gaze position on the validation poster by intersecting this vector with the poster. Then, use the determined pose of the scene camera (requires a calibrated camera) to compute data quality measures in degrees. If the eye tracker provides a 3D gaze point (e.g. the Tobii Pro Glasses 2 and 3), this 3D gaze point is used in lieu of a gaze direction vector derived from the gaze position in the scene camera with respect to the scene camera. *Default mode when a camera calibration is available.*|
|Left eye ray + pose|`pose_left_eye`|Use the gaze direction vector for the left eye provided by the eye tracker to determine gaze position on the validation poster by intersecting this vector with the poster. Then, use the determined pose of the scene camera (requires a camera calibration) to compute data quality measures in degrees with respect to the left eye.|
|Right eye ray + pose|`pose_right_eye`|Use the gaze direction vector for the right eye provided by the eye tracker to determine gaze position on the validation poster by intersecting this vector with the poster. Then, use the determined pose of the scene camera (requires a camera calibration) to compute data quality measures in degrees with respect to the right eye.|
|Average eye rays + pose|`pose_left_right_avg`|For each time point, take angular offset between the left and right gaze positions and the fixation target and average them to compute data quality measures in degrees. Requires 'Left eye ray + pose' and 'Right eye ray + pose' to be enabled|

In summary, by default for eye trackers for which a camera calibration is available the gaze position in the scene camera is transformed to a gaze position on the validation poster by means of intersecting a camera-relative gaze direction vector with the validation poster plane. Accuracy and precision are then computed using the angle between the vectors from the scene camera to the fixation target and to the gaze position on the poster. For eye trackers for which no camera calibration is available, an homography transformation is used to determine gaze position on the poster and accuracy and precision are computed using an assumed viewing distance configured in the glassesValidator project's configuration file. As discussed in the "Assuming a fixed viewing distance" section of the glassesValidator paper (Niehorster et al., in prep), differences in computed values between these two modes are generally small. Nonetheless, it is up to the researcher to decide whether the level of error introduced when operating without a camera calibration is acceptable and whether they should perform their own camera calibration.

### Matching gaze data to fixation targets
This section discusses how to decide which part of the gaze data constitutes a fixation on each of the fixation targets.
By default, to associate gaze data with fixation targets, glassesValidator first uses the [I2MC fixation classifier](https://github.com/dcnieho/I2MC_Python) to classify the fixations in the gaze position data on the poster.
Then, for each fixation target, the nearest fixation in poster-space that is at least 50 ms long is selected from all the classified fixations during the validation procedure. This matching is done in such a way that no fixation is matched to more than one fixation target.

The matching between fixations and fixation targets produced by this procedure is stored in a file `analysisInterval.tsv` in each recordings directory in the glassesValidator project. The advanced user can provide their own matching by changing the contents of this file.


## API
All of glassesValidator's functionality is exposed through its API. Below are all functions that are part of the
public API:
### glassesValidator.config
|function|description|
| --- | --- |
|`get_validation_setup()`|z|
|`get_targets()`|z|
|`get_markers()`|z|
|`deploy_validation_config()`|z|

#### glassesValidator.config.markerBoard
|function|description|
| --- | --- |
|`deploy_maker()`|z|
|`deploy_marker_images()`|z|
|`deploy_default_pdf()`|z|

### glassesValidator.GUI
|function|description|
| --- | --- |
|`run()`|z|

### glassesValidator.preprocess
|function|description|
| --- | --- |
|`get_recording_info()`|z|
|`do_import()`|z|
|`pupil_core()`|z|
|`pupil_invisible()`|z|
|`SeeTrue()`|z|
|`SMI_ETG()`|z|
|`tobii_G2()`|z|
|`tobii_G3()`|z|

### glassesValidator.process
|function|description|
| --- | --- |
|`code_marker_interval()`|z|
|`detect_markers()`|z|
|`gaze_to_board()`|z|
|`compute_offsets_to_targets()`|z|
|`determine_fixation_intervals()`|z|
|`calculate_data_quality()`|z|
|`do_coding()`|z|
|`do_process()`|z|
|`DataQualityType`|z|

### glassesValidator.utils
|function|description|
| --- | --- |
|`make_video()`|z|
|`EyeTracker`|z|
|`Recording`|z|
