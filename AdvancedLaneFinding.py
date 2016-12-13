# Import common libraries
import os
import glob
from tqdm import tqdm

# Import everything needed to process and transform images
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

        # Iterate over all images
        for file in tqdm(allImages, unit="Image"):
            outfile = file.split(".")
            outfile = outfile[0] + "_processed." + outfile[1]
            mpimg.imsave(outfile, self.findLanesImage(mpimg.imread(file)))
        return 0
