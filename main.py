#-*- coding:utf-8 -*-
"""
Created on Wed January 4 2016

@author: Weisen pan

@function:
         face detection, face alignement
	 one key proprecess

@attention:
	1. only work on linux 
	2. use the dlib to do the face detection and key point detection
	3. 
"""

import sys

import dlib
from skimage import io

import numpy as np
import cv2

detector = dlib.get_frontal_face_detector()

def dlib_face_detection(dlib_rgb_img):
	# The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    #dets = detector(dlib_rgb_img, 1)

    # Finally, if you really want to you can ask the detector to tell you the score
	# for each detection.  The score is bigger for more confident detections.
	# The third argument to run is an optional adjustment to the detection threshold,
	# where a negative value will return more detections and a positive value fewer.
	# Also, the idx tells you which of the face sub-detectors matched.  This can be
	# used to broadly identify faces in different orientations.
    dets, scores, idx = detector.run(dlib_rgb_img, 1, -1)
    max_score = 0
    print("Number of faces detected: {}".format(len(dets)))
    for i, d in enumerate(dets):
        #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        #    i, d.left(), d.top(), d.right(), d.bottom()))
        #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}, score: {}, face_type:{}".format(
        #    i, d.left(), d.top(), d.right(), d.bottom(), scores[i], idx[i]))
        offset = 10;
        if scores[i] > max_score:
        	max_score = scores[i]
	        max_x = d.left() - offset
	        max_y = d.top() - offset
	        max_w = d.right() - d.left() + 2 * offset
	        max_h = d.bottom() - d.top() + 2 * offset
    dlib_rgb_face_roi = dlib_rgb_img[max_y:max_y+max_h, max_x:max_x+max_w]
    #cv2.rectangle(dlib_rgb_img, (max_x,max_y), (max_x+max_w,max_y+max_h), (255, 0, 255), 2)
    print("Max face rectangle: Left: {} Top: {} Width: {} Height: {}, Max Score: {}".format(
            max_x, max_y, max_w, max_h, max_score))
    return dlib_rgb_face_roi



if __name__ == "__main__":

	for f in sys.argv[1:]:
	    print("Processing file: {}".format(f))
	    cv2_bgr_img = cv2.imread(f)
	    dlib_rgb_img = cv2.cvtColor(cv2_bgr_img, cv2.COLOR_BGR2RGB)
	    dlib_rgb_face_roi = dlib_face_detection(dlib_rgb_img)

	    cv2_bgr_face_roi = cv2.cvtColor(dlib_rgb_face_roi, cv2.COLOR_RGB2BGR)
	    cv2.imshow('dlib_rgb_img',dlib_rgb_img)
	    cv2.imshow('cv2_bgr_face_roi',cv2_bgr_face_roi)
	    cv2.imwrite('test.jpg', cv2_bgr_face_roi)
	    cv2.waitKey(0)
	    cv2.destroyAllWindows()

# # For test the dlib face detection
# if __name__ == "__main__":

# 	for f in sys.argv[1:]:
# 	    print("Processing file: {}".format(f))
# 	    cv2_bgr_img = cv2.imread(f)
# 	    dlib_rgb_img = cv2.cvtColor(cv2_bgr_img, cv2.COLOR_BGR2RGB)
# 	    dlib_rgb_face_roi = dlib_face_detection(dlib_rgb_img)

# 	    cv2_bgr_face_roi = cv2.cvtColor(dlib_rgb_face_roi, cv2.COLOR_RGB2BGR)
# 	    cv2.imshow('dlib_rgb_img',dlib_rgb_img)
# 	    cv2.imshow('cv2_bgr_face_roi',cv2_bgr_face_roi)
# 	    cv2.imwrite('test.jpg', cv2_bgr_face_roi)
# 	    cv2.waitKey(0)
# 	    cv2.destroyAllWindows()