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

    def findLanes(self, image, keep_memory):
        """

        :param image:
        :param keep_memory:
        :return:
        """
        return image

    def findLanesImage(self, image):
        """

        :param image:
        :return:
        """
        return self.findLanes(image, False)

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
