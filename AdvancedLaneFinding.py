# Import common libraries
import os
import glob
from tqdm import tqdm

# Import everything needed to process and transform images
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Import everything needed to edit/save/watch video clips
from moviepy.editor import *

class AdvancedLaneFinding:
    """

    """

    def __init__(self, settings):
        """

        :param settings:
        """
        # Initially store settings for lane finding pipeline
        self.settings = settings

    def runCameraCalibration(self, settings):
        """

        :param settings:
        :return:
        """
        # Find all images in given folder
        allImages = glob.glob(os.path.join(settings["Folder"], "{}*.jpg".format(settings["Pattern"])))

        print("Start camera calibration on {} images in folder {}".format(len(allImages), settings["Folder"]))

        nCorners = (6, 9)

        objpoints = []
        imgpoints = []

        objp = np.zeros((6*9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Iterate over all images
        for file in tqdm(allImages, unit="Image"):
            outfile = file.split("/")
            outfile = os.path.join(os.path.join(outfile[0], "processed"), outfile[1])

            img = mpimg.imread(file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, nCorners, None)
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, nCorners, corners, ret)
                #mpimg.imsave(outfile, img)
            else:
                print("Discard image {} for calibration".format(file))
                #mpimg.imsave(outfile, img)

        img = mpimg.imread(allImages[0])
        img_size = (img.shape[1], img.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

        for file in tqdm(allImages, unit="Image"):
            outfile = file.split("/")
            outfile = os.path.join(os.path.join(outfile[0], "processed"), outfile[1])

            img = mpimg.imread(file)
            undist = cv2.undistort(img, mtx, dist, None, mtx)
            mpimg.imsave(outfile, undist)

    def findLanes(self, image, keepMemory):
        """

        :param image:
        :param keepMemory:
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
