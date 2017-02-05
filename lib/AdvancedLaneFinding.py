# Import common libraries
import os
import glob
from tqdm import tqdm
import pickle

# Import everything needed to process and transform images
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Import everything needed to edit/save/watch video clips
from moviepy.editor import *

# Import local files
from lib.Line import Line
from lib.Utils import *


class AdvancedLaneFinding:
    """
    AdvancedLaneFinding (ALF) provides methods to detect and visualize lanes from a
    given video file or single images.
    For pre processing it contains methods to determine the parameters for image rectification.
    """

    def __init__(self, settings):
        """

        :param settings: Settings for lane finding
        """
        # Store settings for lane finding pipeline
        self.settings = settings

        # Camera calibration data
        self.calibration_available = False
        self.calibration_matrix = 0
        self.calibration_distortion = 0

        # Line objects for left and right lanes
        self.lane_data_valid = False
        self.lane_left = Line()
        self.lane_right = Line()

        # Parameters for masking the image
        self.mask_outer = [(0, 0), (0, 0), (0, 0), (0, 0)]
        self.mask_inner = [(0, 0), (0, 0), (0, 0), (0, 0)]

        # Parameters for lane tracking
        self.track_lanes = False

        # Set all parameters to default values
        self.reset_parameters()

    def reset_parameters(self):
        """

        """
        self.mask_outer = [(100, 710), (500, 480), (800, 480), (1180, 710)]
        self.mask_inner = [(420, 710), (590, 500), (720, 500), (900, 710)]

    def run_camera_calibration(self, settings):
        """
        Check if camera calibration can be read from storage or should/has to be done again.
        :param settings: Settings for camera calibration
        """
        # Set filename to store camera calibration information
        storage_file = os.path.join(settings["StorageFolder"], "camCalibration.p")

        # Check if intermediate resulsts should be stored
        store_results = settings["StoreIntermediateResults"]

        # Check if calibration should be determined or stored calibration should be used
        run_calibration = not(settings["UseStoredFile"])

        if not run_calibration:

            # Check if file exists
            if os.path.isfile(storage_file):
                print("Load camera calibration from {}".format(storage_file))
                calibration_data = pickle.load(open(storage_file, "rb"))

                # Update internal storage
                self.calibration_available = True
                self.calibration_matrix = calibration_data["mtx"]
                self.calibration_distortion = calibration_data["dist"]
            else:
                print("File {} does not exist --> Re-Run calibration algorithm".format(storage_file))
                run_calibration = True

        # If requested, run calibration and store the results
        if run_calibration:

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

            # Find all images in given folder
            all_images = glob.glob(os.path.join(settings["Folder"], "{}*".format(settings["Pattern"])))
            print("Start camera calibration on {} images in folder {}".format(len(all_images), settings["Folder"]))

            # Read all chessboard dimensions
            dims = eval(settings["ChessboardDimension"])

            # Arrays to store object points and image points from all the images.
            obj_points = []  # 3d points in real world space
            img_points = []  # 2d points in image plane.

            # Step through the list and search for chessboard corners
            for file_name in all_images:
                img = cv2.imread(file_name)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Find the chessboard corners
                ret = False

                # Iterate of all possible dimension
                for d in dims:
                    # Try this dimensions
                    ret, corners = cv2.findChessboardCorners(gray, d, None)

                    # If corners where found add them to list and break loop
                    if ret:

                        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
                        objp = np.zeros((d[1] * d[0], 3), np.float32)
                        objp[:, :2] = np.mgrid[0:d[0], 0:d[1]].T.reshape(-1, 2)
                        obj_points.append(objp)

                        # Improve accuracy
                        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                        img_points.append(corners)

                        if store_results:
                            img = cv2.drawChessboardCorners(img, d, corners, ret)
                            output_file = os.path.join(settings["StorageFolder"], "chess_" + os.path.basename(file_name))
                            os.makedirs(os.path.dirname(output_file), exist_ok=True)
                            cv2.imwrite(output_file, img)

                        break

                # If found, add object points, image points
                if not ret:
                    print("Discard {} for camera calibration".format(file_name))

            # Test undistortion on an image
            img = cv2.imread(all_images[0])
            img_size = (img.shape[1], img.shape[0])

            # Do camera calibration given object points and image points
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)

            # Store undistored images if requested
            if store_results:
                for file_name in all_images:
                    img = cv2.imread(file_name)
                    undist = cv2.undistort(img, mtx, dist, None, mtx)
                    output_file = os.path.join(settings["StorageFolder"], "undist_" + os.path.basename(file_name))
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    cv2.imwrite(output_file, undist)

            # Save the camera calibration result for later use
            dist_pickle = {}
            dist_pickle["mtx"] = mtx
            dist_pickle["dist"] = dist

            # Check if folder exists and create it if necessary

            os.makedirs(os.path.dirname(storage_file), exist_ok=True)
            pickle.dump(dist_pickle, open(storage_file, "wb"))

            # Update internal storage of calibration data
            self.calibration_available = True
            self.calibration_matrix = mtx
            self.calibration_distortion = dist

    def create_binary_image(self, image):
        """
        Apply threshold on HLS s channel and apply sobel to x direction
        :param image: Image that should be converted to binary
        :return: binary_image
        """
        # Pre process image data (e.g. convert to color spaces)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        s_channel = hls[:, :, 2]

        # Threshold on s channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= self.settings["HlsThreshLo"]) & (s_channel <= self.settings["HlsThreshHi"])] = 1

        # Sobel x
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
        abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # Threshold x gradient
        thresh_min = self.settings["SobelXThreshLo"]
        thresh_max = self.settings["SobelXThreshHi"]
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

        return combined_binary

    def fit_lane(self, image, tracking=False):

        if tracking and self.lane_left.detected and self.lane_right.detected:
            # Assume you now have a new warped binary image
            # from the next frame of video (also called "binary_warped")
            # It's now much easier to find line pixels!
            nonzero = image.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            margin = 100

            left_fit = self.lane_left.best_fit
            right_fit = self.lane_right.best_fit

            left_lane_inds = (
                (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
                    nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
            right_lane_inds = (
                (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
                    nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

            # Again, extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]

        else:

            # Assuming you have created a warped binary image called "binary_warped"
            # Take a histogram of the bottom half of the image
            histogram = np.sum(image[image.shape[0] / 2:, :], axis=0)

            # Create an output image to draw on and  visualize the result
            out_img = np.dstack((image, image, image)) * 255

            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            midpoint = np.int(histogram.shape[0] / 2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

            # Choose the number of sliding windows
            nwindows = 9
            # Set height of windows
            window_height = np.int(image.shape[0] / nwindows)
            # Identify the x and y positions of all nonzero pixels in the image
            nonzero = image.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            # Current positions to be updated for each window
            leftx_current = leftx_base
            rightx_current = rightx_base

            # Set the width of the windows +/- margin
            margin = 100
            # Set minimum number of pixels found to recenter window
            minpix = 50

            # Create empty lists to receive left and right lane pixel indices
            left_lane_inds = []
            right_lane_inds = []

            # Step through the windows one by one
            for window in range(nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = image.shape[0] - (window + 1) * window_height
                win_y_high = image.shape[0] - window * window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin

                # Identify the nonzero pixels in x and y within the window
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                nonzerox < win_xright_high)).nonzero()[0]
                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_left_inds) > minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

            # Concatenate the arrays of indices
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)

            # Extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]

        data_left = np.reshape(np.concatenate((lefty, leftx)), (2, -1)).transpose()
        data_right = np.reshape(np.concatenate((righty, rightx)), (2, -1)).transpose()

        return data_left, data_right

    def calc_curvature(self, lane_pixels):
        """

        :param lane_pixels:
        :return:
        """

        ind_left, ind_right = lane_pixels

        self.lane_left.update(ind_left[:, 0], ind_left[:, 1])
        self.lane_right.update(ind_right[:, 0], ind_right[:, 1])

    def undistort(self, image):
        """
        Undistort input image with internally stored camera calibration matrix
        :param image: Input image
        :return: Undistorted image
        """
        return cv2.undistort(image, self.calibration_matrix, self.calibration_distortion, None, self.calibration_matrix)

    def find_lanes(self, image, track_lanes, store_results=False, storage_folder="", file_name=""):
        """

        :param image:
        :param track_lanes:
        :param store_results:
        :param storage_folder:
        :param file_name:
        :return:
        """

        # 1. Apply the distortion correction to the raw image
        if self.calibration_available:
            undist = cv2.undistort(image, self.calibration_matrix, self.calibration_distortion, None, self.calibration_matrix)
            save_storage(store_results, storage_folder, file_name, "step1_", image)
        else:
            print("Camera calibration is not available")
            return image

        # 2. Use color transforms, gradients, etc., to create a thresholded binary image.
        binary = self.create_binary_image(undist)
        binary = apply_mask(binary, self.mask_outer)
        binary = apply_mask(binary, self.mask_inner, inverse=True)
        save_storage(store_results, storage_folder, file_name, "step2_", binary)

        # 3. Apply a perspective transform to rectify binary image ("birds-eye view")
        src = np.float32([[190, 720], [583, 460], [705, 460], [1150, 720]])
        bev, MInv = transform_to_bev(binary, src, offset=(300, 0))
        save_storage(store_results, storage_folder, file_name, "step3_", bev)

        # 4. Detect lane pixels and fit to find lane boundary
        lane_pixels = self.fit_lane(bev, track_lanes)

        # 5. Determine curvature of the lane and vehicle position with respect to center
        self.calc_curvature(lane_pixels)

        # 6.
        if track_lanes:
            left_lane, right_lane = self.lane_left.best_fit, self.lane_right.best_fit
        else:
            left_lane, right_lane = self.lane_left.current_fit, self.lane_right.current_fit

        # 7. Warp the detected lane boundaries back onto the original image
        radius, position = radius_and_position(left_lane, right_lane, 700, 640)
        warped_image = draw_lanes(left_lane, right_lane, image, bev, MInv, radius, position)

        return warped_image

    def find_lanes_image(self, image, store_results=False, storage_folder="", file_name=""):
        """
        Detect and visualize lane boundaries for a standalone image
        :param image: Image to process
        :param store_results: Store intermediate results
        :param storage_folder: Folder to store results and intermediate results
        :param file_name: Filename extension for intermediate results
        :return: Image with visualized lane boundaries
        """
        return self.find_lanes(image, False, store_results, storage_folder, file_name)

    def find_lanes_video(self, image):
        """
        Wrapper method to detect lanes with VideoFileClip methods
        :param image: Signle image from video file
        :return: Image with visualized lane boundaries
        """
        return self.find_lanes(image, self.track_lanes)

    def process_video(self, settings):
        """
        Process video defined in settings file and detect lane boundaries
        :param settings: Settings file for video processing
        """

        file_names = settings["InputFile"]
        storage_folder = settings["StorageFolder"]
        self.track_lanes = settings["TrackLanes"]

        for file_name in file_names:
            output_file = os.path.join(storage_folder, "proc_" + os.path.basename(file_name))
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            print("Start processing video {} and save it as {}".format(file_name, output_file))

            input = VideoFileClip(file_name)
            output = input.fl_image(self.find_lanes_video)
            output.write_videofile(output_file, audio=False)

    def process_image_folder(self, settings):
        """
        Read images in input folder and detect lane boundaries
        :param settings: Settings file for image processing
        """
        # Read settings
        input_folder = settings["InputFolder"]
        storage_folder = settings["StorageFolder"]
        pattern = settings["Pattern"]
        store_results = settings["StoreIntermediateResults"]

        # Find all images in given folder
        all_images = glob.glob(os.path.join(input_folder, "{}*.jpg".format(pattern)))

        print("Start processing images {} in folder {} with pattern {}".format(len(all_images), input_folder, pattern))

        # Iterate over all images
        for file_name in tqdm(all_images, unit="Image"):
            output_file = os.path.join(storage_folder, "proc_" + os.path.basename(file_name))
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            img = mpimg.imread(file_name)
            mpimg.imsave(output_file, self.find_lanes_image(cv2.resize(img, (1280, 720)), store_results,
                                                            storage_folder, file_name))
