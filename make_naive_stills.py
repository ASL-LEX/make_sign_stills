import math
import os
import plac
import logging

import cv2

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(name)s - %(asctime)-14s %(levelname)-8s: %(message)s',
                              "%m-%d %H:%M:%S")

ch.setFormatter(formatter)

logger.handlers = []  # in case module is reload()'ed, start with no handlers
logger.addHandler(ch)


def save_median_frame(video, outputdir):
    """Saves the median frame of the video as a JPEG

    Args:
        video (str): path to video file

    Returns:
        (str) path to saved JPEG file
    """
    cap = cv2.VideoCapture(video)
    f = math.floor(get_num_frames(video)/2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f-1)
    ret, frame = cap.read()
    filename = os.path.splitext(os.path.basename(video))[0]+"_median_frame.jpg"
    cv2.imwrite(os.path.join(outputdir, filename), frame)
    cap.release()
    return(os.path.join(outputdir, filename))
    
    

def get_num_frames(video):
    """Returns the number of frames in a video"""
    cap = cv2.VideoCapture(video)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return(num_frames)



@plac.annotations(inputdir=('path to the directory containing video files to process '
                            '(in .m4v, .mov, or .mp4 format).',
                            'positional', None, str),
                  outputdir=('path to directory into which to place results', 'positional',
                             None, str),
                  clean=('clean outputdir of existing .jpg  and .gif files', 'flag'))
def main(inputdir="./videos", outputdir=None, clean=False):
    """
    Iterates over files in directory and creates a still image from the median frame of the video

    Processes all .m4v, .mov, and .mp4 videos in a given input
    directory

    """
    if not outputdir:
        outputdir = inputdir
    if not os.path.isdir(outputdir):
        os.makedirs(outputdir)
    if clean:
        jpg_files = [os.path.join(outputdir, f) for
                     f in os.listdir(outputdir) if f.endswith(('.jpg', '.gif'))]
        if jpg_files:
            logger.info("removing existing .jpg and .gif files")
        for f in jpg_files:
            logger.debug("removing {}".format(f))
            os.remove(f)

    vid_files = [os.path.join(inputdir, f) for
                 f in os.listdir(inputdir) if f.endswith(('.m4v', '.mov', '.mp4'))]
    for f in vid_files:
        logger.debug("file: %s" % f)
        logger.debug("%s: num_frames %s" % (f, get_num_frames(f)))
        outfile = save_median_frame(f, outputdir)
        logger.info("wrote %s" % outfile)


if __name__ == "__main__":
    plac.call(main)
