# Company Seal Recogniser

Authur: Kefeng Huang

## Overview

This is a python project to detect company seals from the contract, recognise the characters (mainly Chinese characters) and compare it with the previous ones to tell if it is a fake one. The detection comes with the Ellipse and Line Segment Detector, with Continous Validation ([ELSDc][https://github.com/viorik/ELSDc][1]) method by V. Patraucean and P. Gurdjos and R. Grompone von Gioi.

## Pre-request

The ELSDc method requires the library `lapack`. This repo includes these libraries for Mac  in the `src/ELSDC/lib/`. If other versions are downloaded, remember to change the name or path in the `src/ELSDC/Makefile`.

The python part requires `cv2` and `numpy`. Install them using `pip3 install opencv-python numpy`.

## Usage

### Compile the ELSDC

Make sure the library path in `src/ELSDC/Makefile` is correct. Run `make` at repo `src/ELSDC`. Leave the ouput `elsdc` file at `src/ELSDC/elsdc` or change the `elsdc_path` at `src/python/ellipse_detecetor.py` line 26.

### Run Python

Change the name of the input file and output path at `src/python/main.py` when calling the function. Then,  under the project repo, run `python3 src/python/main.py`.

## Output

The outputs at `images/ouput/` contains to kinds of images:

* *\*number.png* : This is an area cut from the orginal image contains the seal detected. The edge length is 1.5 times larger than the major length of the ellipse.
* *label.png*: This is an resized image with suitable size to show the ellipse detected with a blue circle.