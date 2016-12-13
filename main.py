import yaml
import argparse

from AdvancedLaneFinding import AdvancedLaneFinding

def main():

    # Set parser for
    parser = argparse.ArgumentParser(description="Processing input arguments")
    parser.add_argument("-s", "--settings_file", help="Set yaml settings file", required=True)
    args = parser.parse_args()

    # Move all config parameters to separate yaml file
    with open(args.settings_file) as fi:
        settings = yaml.load(fi)

    # Check if video should be processed
    if settings["Video"]["Process"]:
        alf = AdvancedLaneFinding(settings["LaneFindingParameters"])
        alf.processVideo(settings["Video"]["File"])

    # Check if image folder should be processed
    if settings["Image"]["Process"]:
        alf = AdvancedLaneFinding(settings["LaneFindingParameters"])
        alf.processImageFolder(settings["Image"]["Folder"], settings["Image"]["Pattern"])

if __name__ == "__main__":
    main()