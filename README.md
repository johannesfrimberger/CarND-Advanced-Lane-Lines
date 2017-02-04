**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./results/cam_calibration.png "Undistorted"
[image2]: ./results/img_process_step1.jpg "Road Transformed"
[image3]: ./results/img_process_step2.jpg "Binary Example"
[image4]: ./results/img_process_step3.png "Warp Example"
[image5]: ./results/img_process_step4.jpg "Fit Visual"
[image6]: ./results/img_process_step5.jpg "Output"
[video1]: ./results/project_video.mp4 "Video"

###Code Structure

The

| Name | Description |
|:-------------------------:|:---------:|
| main.py | Main method to read settings and run lane finding algo |
| AdvancedLaneFinding.py | Contains AdvancedLaneFinding (ALF) class |
| Line.py | Implementation of Line class storing lane information |
| Utils.py | Static methods used within ALF |
| config.yaml | Settings file to configure ALF |

To start the code you should run

`main.py -s config.yaml`

It will detect lanes for images and/or videos with the settings choosen in the `config.yaml` file.

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code used for camera calibration is located inside the `ALF` class and can be executed using
the `run_camera_calibration` method.

It takes the config dictionary read from the yaml section `CameraCalibration` which allows these settings:

| Name                      | Type      | Description |
|:-------------------------:|:---------:|:-----------:|
| Folder                    | String    | Folder to search for chessboard images |
| Pattern                   | String    | Pattern of images that should be used |
| UseStoredFile             | Boolean   | Use stored camera calibration file instead of calculcating from scratch |
| StorageFolder             | String    | Folder to store calibration file |
| ChessboardDimension       | Tuple     | Possible chessboard dimension to look for |
| StoreIntermediateResults  | Boolean   | Store intermediate results for debugging |

Default settings can be seen in `config.yaml`.

The `run_camera_calibration` method assumes that the chessboard is fixed at the (x,y) plane
with z=0 and that the same chessboard is used in all images.

It then iterates over all images and checks if one of the possible chessboard dimensions fits
the image. Multiple dimensions are used as in some images the chessboard corners are hidden.

The findings are stored within a single list and processed by the OpenCv `cv2.calibrateCamera()`
method returning the distortion correction. This can be applied to any image taken with the same
camera using the `cv2.undistort()`.

The results of distortion correction can be seen here:

![alt text][image1]

###Pipeline (single images)

Images from a selected folder can be processed using the `process_image_folder` method with the `ALF`
class.

It takes the config dictionary read from the yaml section `Image` which allows these settings:

| Name                      | Type      | Description |
|:-------------------------:|:---------:|:-----------:|
| Process                   | Boolean   | Activate/Deactivate function |
| InputFolder               | String    | Folder to search for images |
| Pattern                   | String    | Pattern of images that should be used |
| StoreIntermediateResults  | Boolean   | Store intermediate results for debugging |
| StorageFolder             | String    | Folder to store results |

Default settings can be seen in `config.yaml`.

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image2]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32([[190, 720], [583, 460], [705, 460], [1150, 720]])
dst = np.float32([
        [offset[0], img_size[1] - offset[1]],
        [offset[0], offset[1]],
        [img_size[0] - offset[0], offset[1]],
        [img_size[0] - offset[0], img_size[1] - offset[1]]
    ])

```
This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 190, 720      | 300, 720      |
| 583, 460      | 300, 0        |
| 705, 460      | 980, 0        |
| 1150, 720     | 980, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

To process a video you can use the `process_video` method with the `ALF` class.

It takes the config dictionary read from the yaml section `Video` which allows these settings:

| Name                      | Type            | Description |
|:-------------------------:|:---------------:|:-----------:|
| Process                   | Boolean         | Activate/Deactivate function |
| InputFile                 | List of Strings | Videos that should be processed |
| StorageFolder             | String          | Folder to store results |
| TrackLanes                | Boolean         | Track lanes between successive frames |

Default settings can be seen in `config.yaml`.

To detect lanes it uses the same methods as described above for single (independent) images.
Additional it keeps track of the detected lanes to improve stability.

####1. Averaging over multiple frames



####2. Adapt search window



####3. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./results/project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

