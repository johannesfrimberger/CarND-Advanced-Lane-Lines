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
[image3]: ./results/img_process_step2.png "Binary Example"
[image4]: ./results/img_process_step3.png "Warp Example"
[image5]: ./results/img_process_step4.png "Fit Visual"
[image6]: ./results/img_process_step5.jpg "Output"
[video1]: ./results/project_video.mp4 "Video"
[video2]: ./results/video_step_1.png "Video Search Window Adaption"

###Code Structure

File-Overview:

| Name | Description |
|:-------------------------:|:---------:|
| main.py | Main method to read settings and run lane finding algo |
| AdvancedLaneFinding.py | Contains AdvancedLaneFinding (ALF) class |
| Line.py | Implementation of Line class storing lane information |
| Utils.py | Static methods used within ALF |
| config.yaml | Settings file to configure ALF |

To start the code you should run

`main.py -s config.yaml`

It will detect lanes for images and/or videos with the settings chosen in the `config.yaml` file.

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

Default settings can be found in `config.yaml`.

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

Images from a selected folder can be processed using the `process_image_folder()` method within the `ALF`
class.

It takes the config dictionary read from the yaml section `Image` which allows these settings:

| Name                      | Type      | Description |
|:-------------------------:|:---------:|:-----------:|
| Process                   | Boolean   | Activate/Deactivate function |
| InputFolder               | String    | Folder to search for images |
| Pattern                   | String    | Pattern of images that should be used |
| StoreIntermediateResults  | Boolean   | Store intermediate results for debugging |
| StorageFolder             | String    | Folder to store results |

Default settings can be found in `config.yaml`.

####1. Provide an example of a distortion-corrected image.

For distortion correction the `ALF` class internally stores the calculated or loaded calibration and
distortion matrix.
Applying `cv2.undistort()` to the input matrix gives the undistorted image:

![alt text][image2]

This image is the input for all further processing steps.

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

To be able to detect lines we first need to create a binary image showing potentially relevant points.

The binary image is created by applying the sobel operator in x direction and thresholding the results with
a configurable lower and upper limit.

Additionally the RGB image is converted to HLS color space and the S color space is thresholded by an lower and
upper limit.

If one of these two binary criteria is fulfilled the pixel is considered as relevant for further processing.

To improve stability we applied two masks to the binary image.
The red mask reduces the view of view to a realistic area to search for lane line while the green mask
takes care of irregularities within the road.

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The perspective transform is implemented in the `transform_to_bev(image, src, offset=(0, 0))` method in the
`Utils.py` file.
It takes the image, the source points and an horizontal and vertical offset of the parallel points in the
transformed image.
The source points where chosen to be a fixed position on the image while the destination points are
adapted to the image size.

```
src = np.float32([[190, 720], [583, 460], [705, 460], [1150, 720]])
dst = np.float32([
        [offset[0], img_size[1] - offset[1]],
        [offset[0], offset[1]],
        [img_size[0] - offset[0], offset[1]],
        [img_size[0] - offset[0], img_size[1] - offset[1]]
    ])

```
This resulted in the following source and destination points for the given image size:

| Source        | Destination   |
|:-------------:|:-------------:|
| 190, 720      | 300, 720      |
| 583, 460      | 300, 0        |
| 705, 460      | 980, 0        |
| 1150, 720     | 980, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To find lanes the image is split horizontally into a left and right image plane.
Afterwards each plane is searched separately.
The top third of the birds eye view image is ignored.

Initially a histogram is taken to find the center of the points lying in the left and right plane.
Around this center a search window is placed and all points within this window are stored.

The search window is adapted in the next step by choosing the mean of the points detected within the current
search window. This makes the search window adapt to the shape of the curve.

The points detected for the left respectively right lane are fit to a polynomial of 2nd order.
The result can be seen as yellow line in the example image.

![alt text][image5]

These lanes are not detected from the image showed in the remaining documentation.
We chose a different image that has a smaller radius to show that the algorithm does not only process
straight lines.

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

To determine the position of the car we assumed the lane width to be 4 meters. According to US laws
the minimum lane width is 3.7m.

With this information we were able to determine the width of a pixel in meters. We took the left and right
position of the lanes in the bottom of the image in pixels and said this width is equivalent to 4 meters.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

All these steps are combined in the `find_lanes()` method. It takes the current image and allows some settings that
are made automatically if you use the `main.py` file with the config file.

It returns an annotated image as shown below.

![alt text][image6]

---

###Pipeline (video)

To process a video you can use the `process_video()` method within the `ALF` class.

It takes the config dictionary read from the yaml section `Video` which allows these settings:

| Name                      | Type            | Description |
|:-------------------------:|:---------------:|:-----------:|
| Process                   | Boolean         | Activate/Deactivate function |
| InputFile                 | List of Strings | Videos that should be processed |
| StorageFolder             | String          | Folder to store results |
| TrackLanes                | Boolean         | Track lanes between successive frames |

Default settings can be found in `config.yaml`.

To detect lanes it uses the same methods as described above for single (independent) images.
Additional it keeps track of the detected lanes to improve stability.

####1. Averaging over multiple frames

Contrary to the algorithm for a single image the video lane tracking algorithm low passes the results
of the current estimation.
This improves the stability of the findings as the lanes will only change continuously (while driving in the
same lane).

####2. Adapt search window

After a first of the curves is found the search window could be chosen more elegantly then using the brute force
method described above.

Knowing the radius of the last curvature measurement and assuming a continuous change in the curvature
you can place a search window around this curve.

![alt text][video2]

####3. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./results/project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

During the implementation I faced, among others, these issues and possible improvements I want to discuss:
- The algorithm is very sensitive to lighting conditions. Playing around with the thresholds works locally
but in general there will always be some situation the algorithm may fail. An adaptive algorithm (like Otsu) could
be used to improve the algorithm. Alternatively it would be possible to adapt the settings iteratively
based on the current fit quality and consistency of the lanes.
- Detection gets worse the further the lane is in front of the car. This problem could maybe be solved
by weighting the influence of the detected points on the "fit" based on the distance to the car. I want
to try this later on.
- Currently left and right lane are considered independently but they will/should have very equal curvature.
This could be used to improve the stability.
- Adapt the mask: The mask used after binary image is currently fixed. It could be adapted with the latest
curvature findings to improve detection on curvy roads.
- If the vehicle changes the lane the algorithm will take a lot of time to adapt. To prevent these an appropriate
reset mechanism has to be implemented.
- Low pass filtering the lanes smoothed the results but makes the system slower in recovering from possible
wrong detections. This could also benefit from a reset mechanism.
- Further the low pass filtering of all coefficients is not entirely correct as it might alter the shape of the curve
of the curve. I experimented with storing the found points for several frames and fitting the lane of this
measurement. This improved stability but tremendously increased the calculating time that's why I did not
proceed with this approach (for now).
