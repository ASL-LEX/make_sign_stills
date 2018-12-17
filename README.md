# Make sign stills
Cloned from Carl Borstell's [Make Sign
Stills](https://github.com/borstell/make_sign_stills).

## Borstell's intro
Estimates sign holds for sign videos and outputs overlay stills of the sign

With this script, it is possible to input videos of individual signs for any sign language (in theory), and it outputs an overlay image of the sign that is representative of the sign movements. 

The analysis is a rather crude way of estimating hold phases in the sign. It makes use of the [OpenCV](https://opencv.org) library for analyzing video frames, the [SciPy](https://www.scipy.org) library for identifying peaks in changes between frames, and the [ImageMagick](https://www.imagemagick.org) library for generating overlay stills.

As a sign video is given to the script, each frame is analyzed and compared pairwise for changes. The first peak (i.e. a lot of changes between frames) is assumed to be the initial transport movement before the sign starts. The script then looks for negative peaks (i.e. small changes ≈ hold phases) and saves these frames as representative phases of the sign. 

## Our work
Mostly housekeeping on the script (to turn it into a
command-line-application using plac), but also notably:

- Added a video play method that highlights the detected key frames
  during playback, which can help in tuning the parameters of the
  algorithm.
- Added .gif creation


## Getting Started
1. Clone the repo
2. Install [ImageMagick
   v6](https://legacy.imagemagick.org/script/index.php) (there are
   reports that v7 does nto work with the Wand python bindings
3. `cd` to the repo directory, and run `pipenv install` (or `pipenv
   install --skip-lock` to be less strict with version requirements)
