import math
import os
import plac

import cv2


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
    return(os.path.join(outputdir,filename))
    
    

def get_num_frames(video):
    """Returns the number of frames in a video"""
    cap = cv2.VideoCapture(video)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return(num_frames)



@plac.annotations(inputdir=('path to the directory containing video files to process.'
                            ' Output files will be written to this same directory.',
                            'positional'),
                  outputdir=('path to directory into which to place results', 'positional'))
def main(inputdir="./videos", outputdir=None):
    """
    Iterates over files in directory and creates overlay images of key frames for each .mp4 file
    """
    if not outputdir:
        outputdir = inputdir
    vid_files = [os.path.join(inputdir, f) for f in os.listdir(inputdir) if f.endswith('.m4v')]
    for f in vid_files:
        print("file: %s" % f)
        print("%s: num_frames %s" % (f, get_num_frames(f)))
        outfile = save_median_frame(f, outputdir)
        print("wrote %s" % outfile)


if __name__ == "__main__":
    plac.call(main)
