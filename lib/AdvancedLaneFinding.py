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
        self.mask_outer = [(100, 660), (500, 480), (800, 480), (1130, 660)]
        self.mask_inner = [(370, 660), (530, 550), (780, 550), (990, 660)]

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

    def apply_mask(self, image, mask, extend=False, inverse=False):
        """
        Apply internally stored mask to image
        :param image:
        :param mask:
        :param extend:
        :return:
        """
        mask_bl = mask[0]
        mask_tl = mask[1]
        mask_tr = mask[2]
        mask_br = mask[3]

        if extend:
            # TODO: Check that boundaries are not exceeded
            mask_bl = (mask_bl[0] - 50, mask_bl[1] + 50)
            mask_tl = (mask_tl[0] - 50, mask_tl[1] - 50)
            mask_tr = (mask_tr[0] + 50, mask_tr[1] - 50)
            mask_br = (mask_br[0] + 50, mask_br[1] + 50)

        vertices = np.array([[mask_bl, mask_tl, mask_tr, mask_br]],
                        dtype=np.int32)

        return region_of_interest(image, vertices, inverse)

    def create_binary_image(self, image):
        """
        Apply threhshold on HLS s channel and sobel x direction
        :param image: Image that should be thresholded
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
        #combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

        return combined_binary

    def fit_lane(self, image):
        """

        :param image:
        :return:
        """
        # Split image plane into left and right side
        # ToDo: Adapt this to previously found lanes
        left_image = image[:, 0:(image.shape[1]/2)]
        right_image = image[:, (image.shape[1] / 2):]

        # Calculate histogram of points along x-axis and
        left_histogram = np.sum(left_image[left_image.shape[0] / 2:, :], axis=0)
        start_ind_left = np.argmax(left_histogram)

        right_histogram = np.sum(right_image[right_image.shape[0] / 2:, :], axis=0)
        start_ind_right = np.argmax(right_histogram) + (image.shape[1] / 2)

        search_window = 200 / 2

        and_image = np.zeros_like(image)
        and_image[:, (start_ind_left-search_window):(start_ind_left+search_window)] = 1
        and_image[:, (start_ind_right - search_window):(start_ind_right + search_window)] = 1

        slices = 90

        ind_left = np.zeros(shape=(0, 2))
        ind_right = np.zeros(shape=(0, 2))

        for bottom in range(image.shape[0], 0, -slices):
            image_slice = image[(bottom-slices):bottom, :]

            l = np.array(np.where(
                image_slice[:, (start_ind_left - search_window):(start_ind_left + search_window)] > 0)).transpose()
            r = np.array(np.where(
                image_slice[:, (start_ind_right - search_window):(start_ind_right + search_window)] > 0)).transpose()

            l = l + ((bottom-slices), (start_ind_left - search_window))
            r = r + ((bottom - slices), (start_ind_right - search_window))

            if l.shape[0] > 0:
                ind_left = np.concatenate((ind_left, l), axis=0)
            if r.shape[0] > 0:
                ind_right = np.concatenate((ind_right, r), axis=0)

        return ind_left, ind_right

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

        :param image:
        :return:
        """
        return cv2.undistort(image, self.calibration_matrix, self.calibration_distortion, None, self.calibration_matrix)

    def findLanes(self, image, track_lanes, store_results=False, storage_folder="", file_name=""):
        """

        :param image: Image that should be processed
        :param track_lanes: Use information from previous frames to improve tracking accuracy
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
        masked = self.apply_mask(undist, self.mask_outer, extend=True)
        binary = self.create_binary_image(masked)
        binary = self.apply_mask(binary, self.mask_outer)
        binary = self.apply_mask(binary, self.mask_inner, inverse=True)
        save_storage(store_results, storage_folder, file_name, "step2_", binary)

        # 3. Apply a perspective transform to rectify binary image ("birds-eye view")
        src = np.float32([[190, 720], [583, 460], [705, 460], [1150, 720]])
        bev, MInv = transform_to_bev(binary, src, offset=(300, 0))
        save_storage(store_results, storage_folder, file_name, "step3_", bev)

        # 4. Detect lane pixels and fit to find lane boundary
        lane_pixels = self.fit_lane(bev)

        # 5. Determine curvature of the lane and vehicle position with respect to center
        self.calc_curvature(lane_pixels)

        # 6. Warp the detected lane boundaries back onto the original image
        if track_lanes:
            left_lane, right_lane = self.lane_left.best_fit, self.lane_right.best_fit
        else:
            left_lane, right_lane = self.lane_left.current_fit, self.lane_right.current_fit

        warped_image = draw_lanes(left_lane, right_lane, image, bev, MInv)

        return warped_image

    def findLanesImage(self, image, store_results=False, storage_folder="", file_name=""):
        """

        :param image:
        :param keep_memory:
        :return:
        """
        return self.findLanes(image, False, store_results, storage_folder, file_name)

    def findLanesVideo(self, image):
        """

        :param image:
        :return:
        """
        return self.findLanes(image, self.track_lanes)

    def process_video(self, settings):
        """

        :param settings:
        :return:
        """

        file_names = settings["InputFile"]
        storage_folder = settings["StorageFolder"]
        self.track_lanes = settings["TrackLanes"]

        for file_name in file_names:
            output_file = os.path.join(storage_folder, "proc_" + os.path.basename(file_name))
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            print("Start processing video {} and save it as {}".format(file_name, output_file))

            input = VideoFileClip(file_name)
            output = input.fl_image(self.findLanesVideo)
            output.write_videofile(output_file, audio=False)

    def process_image_folder(self, settings):
        """

        :param settings:
        :return:
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
            mpimg.imsave(output_file, self.findLanesImage(cv2.resize(img, (1280, 720)), store_results,
                                                          storage_folder, file_name))
