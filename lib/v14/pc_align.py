#!/usr/bin/env python
# -*- coding:utf8 -*-
# 參考 https://github.com/wangpoet/tools_for_face_dect

import dlib
import numpy
from skimage import io
import cv2,time
from imp import reload
import lib.v14.pc_config as pc; reload(pc)

SCALE_FACTOR      = 1 
FEATHER_AMOUNT    = 11
FACE_POINTS       = list(range(17, 68))
MOUTH_POINTS      = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS  = list(range(22, 27))
RIGHT_EYE_POINTS  = list(range(36, 42))
LEFT_EYE_POINTS   = list(range(42, 48))
NOSE_POINTS       = list(range(27, 35))
JAW_POINTS        = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
							   RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
OVERLAY_POINTS = [
	LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
	NOSE_POINTS + MOUTH_POINTS,
]

# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.6


def get_landmarks(img,index):
	rects = detector(img, 0)
	if len(rects)>0:
		return numpy.matrix([[p.x, p.y] for p in predictor(img, rects[index]).parts()])
	else:
		return []

def annotate_landmarks(im, landmarks):
	im = im.copy()
	for idx, point in enumerate(landmarks):
		pos = (point[0, 0], point[0, 1])
		cv2.putText(im, str(idx), pos,
					fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
					fontScale=0.4,
					color=(0, 0, 255))
		cv2.circle(im, pos, 3, color=(0, 255, 255))
	return im

def draw_convex_hull(im, points, color):
	points = cv2.convexHull(points)
	cv2.fillConvexPoly(im, points, color=color)

	
def transformation_from_points(points1, points2):
	points1 = points1.astype(numpy.float64)
	points2 = points2.astype(numpy.float64)

	c1 = numpy.mean(points1, axis=0)
	c2 = numpy.mean(points2, axis=0)
	points1 -= c1
	points2 -= c2

	s1 = numpy.std(points1)
	s2 = numpy.std(points2)
	points1 /= s1
	points2 /= s2

	U, S, Vt = numpy.linalg.svd(points1.T * points2)

	R = (U * Vt).T

	return numpy.vstack([numpy.hstack(((s2 / s1) * R,
									   c2.T - (s2 / s1) * R * c1.T)),
						 numpy.matrix([0., 0., 1.])])



def warp_im(im, M, dshape):
	output_im = numpy.zeros(dshape, dtype=im.dtype)
	cv2.warpAffine(im,
				   M[:2],
				   (dshape[1], dshape[0]),
				   dst=output_im,
				   borderMode=cv2.BORDER_TRANSPARENT,
				   flags=cv2.WARP_INVERSE_MAP)
	return output_im

	
def face_alignment(img,index=0,tow=100,toh=100):
	tt=time.time()
	landmarks2 = get_landmarks(img,index)
	if len(landmarks2)==0:
		print("align fail.. not a good face")
		return None
	M = transformation_from_points(landmark_model[ALIGN_POINTS],
							   landmarks2[ALIGN_POINTS])
	warped_mask = warp_im(img, M, model.shape)
	h, w, c = warped_mask.shape
	new = cv2.resize(warped_mask[0:h,0:w],(tow,toh),interpolation=cv2.INTER_LINEAR)
	pc.TIME_FACE_ALIGN+=(time.time()-tt)
	return new



detector          = dlib.get_frontal_face_detector()
predictor         = dlib.shape_predictor("lib/model/shape_predictor_68_face_landmarks.dat")
model             = io.imread("lib/model/model.jpg") #使用skimage的io讀取圖片
landmark_model    = get_landmarks(model,0)
	
	