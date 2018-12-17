import cv2
# import sys
import logging
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import plac
from scipy import signal
# from PIL import Image

from wand.image import Image, COMPOSITE_OPERATORS
# from wand.drawing import Drawing
# from wand.display import display
from wand.compat import nested


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(name)s - %(asctime)-14s %(levelname)-8s: %(message)s',
                              "%m-%d %H:%M:%S")

ch.setFormatter(formatter)

logger.handlers = []  # in case module is reload()'ed, start with no handlers
logger.addHandler(ch)


def hist(img):
    """
    Returns a histogram analysis of a single image (in this case, a video frame)
    """
    return cv2.calcHist([img], [0], None, [256], [0, 256])


def read_frames(video):
    """
    Reads a video file and returns a list of the histogram data for each frame
    """
    v = cv2.VideoCapture(video)
    frames = []
    success, image = v.read()
    while success:
        success, image = v.read()
        if success:
            frames.append(hist(image))
    return frames


def get_frame_difference(video):
    """
    Goes through the histograms of video frames pairwise and returns a
    list of frame indices (x) and histogram differences (y)

    """
    frames = read_frames(video)
    x = []
    y = []
    for n, f in enumerate(frames):
        if n != len(frames)-1:
            x.append(n)
            y.append(
                1-(cv2.compareHist(hist(f), hist(frames[n+1]), cv2.HISTCMP_CORREL)))
    return x, y


def plot_changes(video, outputdir):
    """
    Returns a plot of the frame histogram differences over video frames
    (NB: not necessary for the analysis)
    """
    plotname = os.path.splitext(os.path.basename(video))[0]+"_plot_frames.png"
    x, y = get_frame_difference(video)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel='Frame', ylabel='Difference',
           title='Frame differences over time')
    ax.grid()

    fig.savefig(os.path.join(outputdir, plotname))
    # plt.show()


def get_key_frames(video, ignore_tail=0.5, show=False):
    """Reads through the frame differences of a video, assumes the first
    peak to be the start of the sign, then returns the negative peaks
    (i.e. estimated holds) of the remaining frames

    Args:
        video (str): path to video file
        ignore_tail (float): fraction of peaks to ignore (i.e. not
        include in key frames) from the tail end of the
        video. `ignore_tail=0` means include all peaks,
        `ignore_tail=0.4` means ignore the last 40% of peaks. Rounding
        is applied in a conservative manner (i.e. include more peaks
        rather than less)

    """
    x, y = get_frame_difference(video)
    # diff = list(zip(y, x))

    # These are hardcoded figures, you may need to adjust (e.g. 15,25)
    # peaks = signal.find_peaks_cwt(y, np.arange(1, 15)) # Calle's defaults
    peaks = signal.find_peaks_cwt(y, np.arange(1, 15))
    first = peaks[0]
    msg = 'peak frames: {}'.format(peaks)
    logger.debug(msg)
    neg = [1-n for n in y[first:]]
    # These are hardcoded figures, you may need to adjust (e.g. 15,45)
    # peaks2 = signal.find_peaks_cwt(neg, np.arange(1.5, 8)) # Calle's defaults
    # peaks2 = signal.find_peaks_cwt(neg, np.arange(1, 8))
    peaks2 = signal.find_peaks_cwt(neg, np.arange(1, 8))
    msg = 'negative peak frames: {}'.format([peak+first for peak in peaks2])
    logger.debug(msg)
    frames = [peak+first for peak in peaks2]
    tail_frames = frames[math.floor(len(frames) * (1-ignore_tail)):]
    logger.debug('tail frames: {}'.format(tail_frames))
    key_frames = [f for f in frames if f not in tail_frames]
    logger.debug('key frames: {}'.format(key_frames))
    if show:
        play_video(video, peaks, key_frames, tail_frames)
    return key_frames


def save_key_frames(video, outputdir, ignore_tail, show=False):
    """
    Saves the frames that are estimated holds as image files and
    returns a list of their names (NB: only frames in the first half
    of the list of key frames are saved, as later frames are assumed
    to constitute final rest position)

    """
    # looks like the issue might be that the videos are in another directory
    outfile = os.path.splitext(os.path.basename(video))[0]
    all_frames = get_key_frames(video, ignore_tail, show)
    frames = all_frames  # Uncomment if you want all key frames to be included
    # Comment out if you want all key frames included
    # frames = all_frames[:math.ceil(len(all_frames)/2)]
    count = 1
    filenames = []
    for f in frames:
        v = cv2.VideoCapture(video)
        v.set(1, f-1)
        ret, frame = v.read()
        filename = outfile+"_frame"+str(count)+".jpg"
        cv2.imwrite(os.path.join(outputdir, filename), frame)
        filenames.append(filename)
        count += 1
    return filenames


def make_overlay(a, b, outname):
    """Makes an overlay image of two key frames

    Using ImageMagick and the Wand python bindings
    """
    logger.debug("entering make_overlay with a: {},"
                 " b: {}, outname: {}".format(a, b, outname))
    bot = Image(filename=a)
    top = Image(filename=b)

    with nested(bot, top) as (b, t):
        t.transparentize(0.5)
        b.composite_channel("all_channels", t, "dissolve",
                            math.floor(b.width/2) - math.floor(t.width/2),
                            math.floor(b.height/2) - math.floor(t.height/2))
        b.save(filename=outname)
    logger.debug("leaving make_overlay")


def make_gif(images, outname, delay=30):
    """Makes a .gif animation from frames of a video

    Args:
        images (list of str): paths to image files
        delay (int): length of time to display each image (in 1/100sec)
        outname (str): path to write animation, including .gif extension
    """
    with Image() as wand:
        for f in images:
            with Image(filename=f) as img:
                logger.debug("appending %s to wand.sequence" % f)
                wand.sequence.append(img)
        for i in range(len(wand.sequence)):
            with wand.sequence[i] as frame:
                frame.delay = delay
        wand.type = 'optimize'
        logger.debug("saving .gif with filename %s" % outname)
        wand.save(filename=outname)


def make_images(video, outputdir, ignore_tail, show=False, gif=False, debug=False):
    """Creates overlay images of relevant key frames generated from
    videos and deletes individual frames

    """
    imgs = save_key_frames(video, outputdir, ignore_tail, show)
    imgs = [os.path.join(outputdir, f) for f in imgs]
    if gif:
        logger.debug("calling make_gif(%s, %s)" % (imgs, imgs[0].split("_")[0]+".gif"))
        make_gif(imgs, imgs[0].split("_")[0]+".gif")
    outname = imgs[0].split("_")[0]+"_still.jpg"
    logger.debug("from make_images, outname: %s" % outname)
    logger.debug("from make_images, len(imgs): %s, imgs %s" % (len(imgs), imgs))
    if len(imgs) == 1:
        os.system("mv %s %s" % (imgs[0], outname))
    elif len(imgs) == 2:
        make_overlay(imgs[0], imgs[1], outname)
    elif len(imgs) >= 3:
        ims = imgs[:3]
        out1 = (os.path.splitext(outname)[0]
                + "_A" + os.path.splitext(outname)[1])
        logger.debug("from make_images, ims[0]: %s" % ims[0])
        logger.debug("from make_images, ims[1]: %s" % ims[1])
        logger.debug("from make_images, out1: %s" % out1)
        make_overlay(ims[0], ims[1], out1)
        make_overlay(out1, ims[2], outname)
    if not debug:
        if len(imgs) > 1:
            for img in imgs:
                os.system("rm "+img)
        for f in [os.path.join(outputdir, f) for f in os.listdir(outputdir)]:
            if f.endswith("_A.jpg") or f.endswith("_B.jpg"):
                os.system("rm "+f)


def play_video(video, green_frames=None, yellow_frames=None, red_frames=None):
    """plays a video, pausing and color-coding the detected key frames

    Args:
        video (str): path to video file
        green_frames, yellow_frames, red_frames (list of int): index of
            frames belonging to each class. These frames will be rendered
            in green, yellow, and red, respectively.

    """
    if green_frames is None:
        green_frames = []
    if yellow_frames is None:
        yellow_frames = []
    if red_frames is None:
        red_frames = []

    vid_name = os.path.basename(video)
    # play video at full speed
    cap = cv2.VideoCapture(video)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(vid_name, frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    # play video while signaling key frames
    cap = cv2.VideoCapture(video)
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        if i in green_frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blank = np.zeros_like(gray)
            green_frame = cv2.merge([blank, gray, blank])
            cv2.imshow(vid_name, green_frame)
            if cv2.waitKey(200) & 0xFF == ord('q'):
                break
        if i in yellow_frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blank = np.zeros_like(gray)
            yellow_frame = cv2.merge([blank, gray, gray])
            cv2.imshow(vid_name, yellow_frame)
            if cv2.waitKey(500) & 0xFF == ord('q'):
                break
        if i in red_frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blank = np.zeros_like(gray)
            red_frame = cv2.merge([blank, blank, gray])
            cv2.imshow(vid_name, red_frame)
            if cv2.waitKey(200) & 0xFF == ord('q'):
                break
        else:
            cv2.imshow(vid_name, frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        i = i + 1
    cap.release()
    cv2.destroyAllWindows()

    # play back just the selected frames
    cap = cv2.VideoCapture(video)
    for f in yellow_frames:
        cap.set(1, f-1)
        ret, frame = cap.read()  # this seems to not be reading the key frame
        if not ret:
            break
        cv2.imshow(vid_name, frame)
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


@plac.annotations(inputdir=('path to the directory containing video files to process '
                            '(in .m4v, .mov, or .mp4 format).',
                            'positional', None, str),
                  outputdir=('path to directory into which to place results', 'positional',
                             None, str),
                  ignoretail=('fraction of key frames to exclude from tail', 'option'),
                  gif=('make .gif animation from key frames', 'flag'),
                  clean=('clean outputdir of existing .jpg  and .gif files', 'flag'),
                  show=('display videos, highlighting key frames as they are calculated', 'flag'),
                  debug=('verbose console output', 'flag'))
def main(inputdir="./videos", outputdir="./output", ignoretail=0.25, gif=False,
         clean=False, show=False, debug=False):
    """Estimates sign holds for sign videos and creates overlay stills of the sign

    Processes all .m4v, .mov, and .mp4 videos in a given input
    directory, creating composite overlay images of key hold frames,
    and optionally creating .gif animations of the hold frames.

    Output files are named according to the filename of the input
    video, with composite images named [vidname]_still.jpg, and .gif
    animations named [vidname].gif. Individual still images are named
    [vidname]_frame[n].jpg, where n is the 1-based index of the image
    in the sequence

    """
    if not outputdir:
        outputdir = inputdir
    if debug:
        ch.setLevel(logging.DEBUG)
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

    vid_files = [os.path.join(inputdir, f) for f in os.listdir(inputdir)
                 if f.endswith(('.m4v', '.mov', '.mp4'))]
    for f in vid_files:
        logger.info("file: %s" % f)
        plot_changes(f, outputdir)  # Uncomment to create plots of changes in-between frames
        make_images(f, outputdir, ignoretail, show, gif, debug)


if __name__ == "__main__":
    plac.call(main)
