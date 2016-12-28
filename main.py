import yaml
import argparse

from lib.AdvancedLaneFinding import AdvancedLaneFinding

def main():
    """
    Read settings file and run AdvancedLaneFinding class with this settings
    """

    # Set parser for inputs
    parser = argparse.ArgumentParser(description="Processing input arguments")
    parser.add_argument("-s", "--settings_file", help="Set yaml settings file", required=True)
    args = parser.parse_args()

    # All config parameters are written down in separate yaml file
    with open(args.settings_file) as fi:
        settings = yaml.load(fi)

    # Create instance of alf
    alf = AdvancedLaneFinding(settings["LaneFindingParameters"])

    # Run camera calibration
    alf.run_camera_calibration(settings["CameraCalibration"])

    # Check if video should be processed
    if settings["Video"]["Process"]:
        alf.processVideo(settings["Video"]["File"])

    # Check if all images in a folder should be processed
    if settings["Image"]["Process"]:
        alf.processImageFolder(settings["Image"]["Folder"], settings["Image"]["Pattern"])

if __name__ == "__main__":
    main()
