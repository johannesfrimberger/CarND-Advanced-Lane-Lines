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
from Line import Line
from Utils import region_of_interest, transform_to_bev

class AdvancedLaneFinding:
    """
    AdvancedLaneFinding (alf) provides methods to detect and visualize lanes from a
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
        self.left_lane = Line()
        self.right_lane = Line()

        # Parameters for masking the image
        self.mask_outer = [(0, 0), (0, 0), (0, 0), (0, 0)]
        self.mask_inner = [(0, 0), (0, 0), (0, 0), (0, 0)]

        # Set all parameters to default values
        self.reset_pameters()

    def reset_pameters(self):
        """

        """
        self.mask_outer = [(100, 660), (500, 480), (800, 480), (1130, 660)]
        self.mask_inner = [(370, 660), (530, 550), (780, 550), (990, 660)]

    def runCameraCalibration(self, settings):
        """
        Check if camera calibration can be read from storage or should/has to be done again.
        :param settings: Settings for camera calibration
        """

        runCalibration = not(settings["UseStoredFile"])
        if settings["UseStoredFile"]:
            fileName = settings["StorageFile"]
            # Check if file exists
            if os.path.isfile(fileName):
                print("Load camera calibration from {}".format(fileName))
                calibrationData = pickle.load(open(fileName, "rb"))
                self.calibration_available = True
                self.calibration_matrix = calibrationData["mtx"]
                self.calibration_distortion = calibrationData["dist"]
            else:
                print("File {} does not exist --> Re-Run calibration algorithm".format(fileName))
                runCalibration = True

        if runCalibration:

            # Find all images in given folder
            allImages = glob.glob(os.path.join(settings["Folder"], "{}*".format(settings["Pattern"])))

            print("Start camera calibration on {} images in folder {}".format(len(allImages), settings["Folder"]))

            dim = eval(settings["ChessboardDimension"])

            # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
            objp = np.zeros((dim[1] * dim[0], 3), np.float32)
            objp[:, :2] = np.mgrid[0:dim[0], 0:dim[1]].T.reshape(-1, 2)

            # Arrays to store object points and image points from all the images.
            objpoints = []  # 3d points in real world space
            imgpoints = []  # 2d points in image plane.

            # Step through the list and search for chessboard corners
            for filename in allImages:
                img = cv2.imread(filename)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Find the chessboard corners
                ret, corners = cv2.findChessboardCorners(gray, dim, None)

                # If found, add object points, image points
                if ret == True:
                    objpoints.append(objp)
                    imgpoints.append(corners)
                else:
                    print("Discard {} for camera calibration".format(filename))

            # Test undistortion on an image
            img = cv2.imread(allImages[0])
            img_size = (img.shape[1], img.shape[0])

            # Do camera calibration given object points and image points
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

            # Save the camera calibration result for later use
            dist_pickle = {}
            dist_pickle["mtx"] = mtx
            dist_pickle["dist"] = dist
            pickle.dump(dist_pickle, open(settings["StorageFile"], "wb"))

            # Update internal storage of calibration data
            self.calibration_available = True
            self.calibration_matrix = mtx
            self.calibration_distortion = dist

    def apply_mask(self, image, mask, extend = False, inverse = False):
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
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

        #color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))

        return combined_binary

    def fit_lane(self, image, Minv):
        slice = 100
        windows = 100
        center = 0

        colorBev = np.dstack((image, image, image)) * 255

        leftLane = np.copy(image)
        rightLane = np.copy(image)

        xCoordLeft = []
        xCoordRight = []
        yCoordLeft = []
        yCoordRight = []

        # print(colorBev.shape)
        for n in reversed(range(7)):
            imageSlice = image[(center + (n * slice)):(center + ((n + 1) * slice)), :]
            histogram = np.sum(imageSlice, axis=0)
            indLeft = histogram[0:(len(histogram) / 2)].argmax()
            indRight = histogram[(len(histogram) / 2):].argmax() + int(len(histogram) / 2)

            xCoord = float(center) + ((2*float(n)) + 1) * slice / 2.

            if indLeft > 0:
                xCoordLeft.append(xCoord)
                yCoordLeft.append(float(indLeft))

            if indRight > 0:
                xCoordRight.append(xCoord)
                yCoordRight.append(float(indRight))

            #colorBev[(center + (n * slice)):(center + ((n + 1) * slice)), indLeft - windows:indLeft + windows, 0:2] = 0
            #colorBev[(center + (n * slice)):(center + ((n + 1) * slice)), indRight - windows:indRight + windows, 1:3] = 0

        left_fit = np.polyfit(xCoordLeft, yCoordLeft, 2)
        right_fit = np.polyfit(xCoordRight, yCoordRight, 2)
        yvals = np.linspace(0, 100, num=101) * 7.2
        left_fitx = left_fit[0] * yvals ** 2 + left_fit[1] * yvals + left_fit[2]
        right_fitx = right_fit[0] * yvals ** 2 + right_fit[1] * yvals + right_fit[2]

        # Plot up the fake data
        #plt.plot(leftx, yvals, 'o', color='red')
        #plt.plot(rightx, yvals, 'o', color='blue')
        plt.xlim(0, 1280)
        plt.ylim(0, 720)
        plt.plot(left_fitx, yvals, color='green', linewidth=3)
        plt.plot(right_fitx, yvals, color='red', linewidth=3)
        #plt.plot(right_fitx, yvals, color='green', linewidth=3)
        plt.gca().invert_yaxis()  # to visualize as we do the images
        #plt.show()

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(image).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))

        cv2.imshow("Test", newwarp)
        cv2.waitKey()

    def findLanes(self, image, keep_memory):
        """

        :param image: Image that should be processed
        :param keep_memory: Use information from previous frames to improve tracking accuracy
        :return:
        """

        # 1. Apply the distortion correction to the raw image
        if self.calibration_available:
            undist = cv2.undistort(image, self.calibration_matrix, self.calibration_distortion, None, self.calibration_matrix)
        else:
            print("Camera calibration is not available")
            return image

        # 2. Use color transforms, gradients, etc., to create a thresholded binary image.
        undist = self.apply_mask(undist, self.mask_outer, extend=True)
        binary = self.create_binary_image(undist)
        binary = self.apply_mask(binary, self.mask_outer)
        binary = self.apply_mask(binary, self.mask_inner, inverse=True)

        # 3. Apply a perspective transform to rectify binary image ("birds-eye view")
        bev, MInv = transform_to_bev(binary)

        # 4. Detect lane pixels and fit to find lane boundary
        lane = self.fit_lane(bev, MInv)

        # 5. Determine curvature of the lane and vehicle position with respect to center

        # 6. Warp the detected lane boundaries back onto the original image

        #return binary
        return bev

    def findLanesImage(self, image, keep_memory=False):
        """

        :param image:
        :param keep_memory:
        :return:
        """
        #return self.findLanes(image, keep_memory)

        return image

    def findLanesVideo(self, image):
        """

        :param image:
        :return:
        """
        return self.findLanes(image, True)

    def processVideo(self, file):
        """

        :param file:
        :return:
        """
        output_file = file.split(".")
        output_file = output_file[0] + "_Processed." + output_file[1]

        print("Start processing video {} and save it as {}".format(file, output_file))

        input = VideoFileClip(file)
        output = input.fl_image(self.findLanesVideo)
        output.write_videofile(output_file, audio=False)

        return 0

    def processImageFolder(self, folder, pattern):
        """

        :param folder:
        :param pattern:
        :return:
        """
        # Find all images in given folder
        allImages = glob.glob(os.path.join(folder, "{}*.jpg".format(pattern)))

        print("Start processing images {} in folder {} with pattern {}".format(len(allImages), folder, pattern))

        # Iterate over all images
        for file in tqdm(allImages, unit="Image"):
            outfile = file.split("/")
            outfile = os.path.join(os.path.join(outfile[0], "processed"), outfile[1])
            mpimg.imsave(outfile, self.findLanesImage(mpimg.imread(file)))
        return 0
