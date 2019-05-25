# -*- coding: utf-8 -*-

import numpy as np
import cv2,dlib,time
from imp import reload
import lib.v14.pc_config as pc; reload(pc)
import lib.v14.pc_util as pu; reload(pu)


detector = dlib.get_frontal_face_detector()
faceCascade = None
cvdnn = None

def DlibDetBodyFaces(img,oimg,minsize=0,maxsize=0):   #輸出有身體的大頭照
	faces=DlibDetFace(img,img)
	ratio=oimg.shape[0]//img.shape[0]
	imgs,faces=GenImglistFromFace(faces,img,oimg,ratio)
	#for gg in imgs:
	#	ShowImgIfWinOS(gg)
	return imgs,faces

def DlibDetFace(img,oimg,level=1):   #level=1 精準 , 0:嚴謹 (img=縮圖,oimg=原圖)
	#使用dlib自帶的frontal_face_detector作為特徵提取器

	dets = detector(img, level)
	faces=[]
	for i, d in enumerate(dets):
		x=d.left()
		y=d.top()
		w=d.right()-x
		h=d.bottom()-y
		faces.append((x,y,w,h))
	return faces	

def DlibGetBestFace(imgs,score=1,level=0,debug=pc.IS_DEBUG):
	tt=time.time()
	#獲取比較全面的資訊，如獲取人臉與detector的匹配程度

	best_score=score
	best_img=None
	have_face=False
	best_idx=0
	
	scores=[]
	idxs=[]
	goodFace=[]
	for i in range(len(imgs)):
		det, sco, idx = detector.run(imgs[i], level)
		if len(det)==0 : continue  #沒有臉要排除
		if len(det)>1 : continue   #一張圖有2張臉要排除
		if idx[0]==3 : continue    #idx=3代表人臉角度太斜要排除
		if sco[0]<score: continue  #低於門檻要排除
		if debug: print("face: {}, score: {}, face_type:{}".format(len(det),sco[0], idx[0]))
		scores.append(sco[0])
		idxs.append(idx[0])
		goodFace.append(imgs[i])
	
	GoodQuility=False
	for i in range(len(scores)):
		if scores[i]>1 and idxs[i]==0: GoodQuility=True

	for i in range(len(scores)):
		if (scores[i]>best_score) and ((GoodQuility==False) or (GoodQuility==True and idxs[i]==0)):
				best_img=goodFace[i]
				best_score=scores[i]
				best_idx=idxs[i]
				have_face=True		
				
	if debug:
		if have_face:
			print("best score:",best_score," ,best idx:",best_idx)
			pu.ShowImgIfWinOS(best_img)
		else:
			pu.ShowImgIfWinOS(imgs[0])
	
	pc.TIME_BEST_FACE+=(time.time()-tt)
	return have_face,best_img,best_score
	
def IsPerson(img,level=0,score=0.5):
	tt=time.time()
	#faces=DlibDetFace(img,img,level)
	#pc.TIME_FACE_VERIFY+=(time.time()-tt)
	#return len(faces)>0
#'''

	det, sco, idx = detector.run(img, level)
	pc.TIME_FACE_VERIFY+=(time.time()-tt)
	ret=False
	if len(det)==0 : return ret  #沒有臉要排除
	if len(det)>1 : return ret    #一張圖有2張臉要排除
	#if idx[0]==3 : return ret     #idx=3代表人臉角度太斜要排除
	if sco[0]<score: return ret   #低於門檻要排除	
	
	return True
#'''

def CvDetBodyFaces(img,oimg,CVscaleFactor=1.1,CVminNeighbors=1,minsize=0,maxsize=0):   #輸出有身體的大頭照
	tt=time.time()
	faces=CvDetFace(img,img,CVscaleFactor,CVminNeighbors,minsize,maxsize)
	ratio=oimg.shape[0]//img.shape[0]
	imgs,faces=GenImglistFromFace(faces,img,oimg,ratio)
	#for gg in imgs:
	#	ShowImgIfWinOS(gg)
	pc.TIME_FACE_DET+=(time.time()-tt)
	return imgs,faces

	
def CvDetFace(img,oimg,CVscaleFactor=1.1,CVminNeighbors=1,minsize=0,maxsize=0):
	global faceCascade
	cascPath="./lib/model/lbpcascade_frontalface.xml"  #目前速度最快
	#cascPath="./lib/model/haarcascade_frontalface_alt2.xml"  #折衷
	#cascPath="./lib/model/haarcascade_frontalface_alt.xml"  #目前最準
	if faceCascade==None:
		faceCascade = cv2.CascadeClassifier(cascPath)
	ratio=oimg.shape[0]//img.shape[0]
	cvMinSize = pc.MIN_FACE_SIZE//ratio if minsize==0 else minsize
	cvMaxSize = pc.MAX_FACE_SIZE//ratio if maxsize==0 else maxsize
	faces = faceCascade.detectMultiScale(
		img,
		scaleFactor=CVscaleFactor,
		minNeighbors=CVminNeighbors,
		minSize=(cvMinSize,cvMinSize),
		maxSize=(cvMaxSize,cvMaxSize),
		flags= 0
			#| cv2.CASCADE_FIND_BIGGEST_OBJECT
			#| cv2.CASCADE_SCALE_IMAGE 
			| cv2.CASCADE_DO_ROUGH_SEARCH 
			| cv2.CASCADE_DO_CANNY_PRUNING 
		#CASCADE_SCALE_IMAGE CASCADE_DO_CANNY_PRUNING CASCADE_DO_ROUGH_SEARCH
	)
	return faces


def CvDNNDetFace(img,oimg,CVscaleFactor=1.1,CVminNeighbors=1,minsize=0,maxsize=0):
	global cvdnn
	modelFile = "./lib/model/opencv_face_detector_uint8.pb"
	configFile = "./lib/model/opencv_face_detector.pbtxt"
	if cvdnn==None:
		cvdnn = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
	conf_threshold = 0.7
	
	frameOpencvDnn = img.copy()
	#frameOpencvDnn = cv2.cvtColor(frameOpencvDnn,cv2.COLOR_BGR2GRAY)
	frameHeight = frameOpencvDnn.shape[0]
	frameWidth = frameOpencvDnn.shape[1]
	blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.5, (256, 144), [104, 117, 123], False, False)

	cvdnn.setInput(blob)
	detections = cvdnn.forward()
	faces = []
	for i in range(detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > conf_threshold:
			x1 = int(detections[0, 0, i, 3] * frameWidth)
			y1 = int(detections[0, 0, i, 4] * frameHeight)
			x2 = int(detections[0, 0, i, 5] * frameWidth)
			y2 = int(detections[0, 0, i, 6] * frameHeight)
			faces.append([x1, y1, x2, y2])
			#cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)

	conf_threshold = 0.7
	return faces


def DeepDetBodyFaces(img,oimg,minsize=0,maxsize=0):   #輸出有身體的大頭照
	faces=DeepDetFace(img,img)
	ratio=oimg.shape[0]//img.shape[0]
	imgs,faces=GenImglistFromFace(faces,img,oimg,ratio)
	#for gg in imgs:
	#	ShowImgIfWinOS(gg)
	return imgs,faces
	
	
def DeepDetFace(img,oimg):
	import face_recognition as fr
	dets = fr.face_locations(img)
	faces=[]
	for (t,l,r,b) in dets:
		x=l
		y=t
		w=r-x
		h=b-y
		faces.append((x,y,w,h))	
	return faces
	
	
def DrawFace(img,faces,color="R",lsize=5,ratio=1,offset=(0,0)):
	tt=time.time()
	ox,oy=offset
	if color=="B":c=(50,50,255)
	if color=="R":c=(255,50,50)
	if color=="Y":c=(50,255,255)
	r=ratio
	
	for (x, y, w, h) in faces:
		cv2.rectangle(img, (ox+x*r, oy+y*r), (ox+(x+w)*r, oy+(y+h)*r), c, lsize)
	pc.TIME_DRAW_INFO+=(time.time()-tt)
	return img	
	
	
def GenImglistFromFace(faces,det_img,org_img,ratio,debug=pc.IS_DEBUG,clip=False):
	W=org_img.shape[1]
	H=org_img.shape[0]
	filted_face=[]
	filted_img=[]
	rt=ratio
	
	for (x,y,w,h) in faces:
		if (x < 0) or (y < 0) or (w < 0) or (h < 0):
			continue

		#''' 標準大頭照
		fl = (x*rt)-(w*rt)//2
		ft = (y*rt)-(h*rt)//2
		fw = (w*2*rt)
		fh = (h*3*rt)
		#'''
		''' 臉部特寫照
		fl = (x*rt)-(w*rt)//3
		ft = (y*rt)-(h*rt)//3
		fw = (w*1.66*rt)
		fh = (h*2.5*rt)
		#'''
		if (fl <= 0) or (ft <= 0) or (fl+fw >= W-1) or (ft+fh >= H-1):
			continue
		#if debug:print(W,H,x,y,w,h,rt,fl,ft,fw,fh)
		#face = cv2.resize(org_img[f.y:f.y + f.h, f.x:f.x + f.w],(100,100),interpolation=cv2.INTER_CUBIC)
		body = cv2.resize(org_img[ft:ft+fh,fl:fl+fw],(pc.SYSTEM_IMG_WIDTH,pc.SYSTEM_IMG_HEIGHT),interpolation=cv2.INTER_LINEAR) #  INTER_CUBIC INTER_LINEAR INTER_AREA
		filted_img.append(body)
		filted_face.append((x,y,w,h))
		
	return filted_img,filted_face	

	
def FilterFrame(org_frame,first_frame,w,h):
	tt=time.time()

	firstframe=cv2.resize(first_frame,(w,h),interpolation=cv2.INTER_AREA)
	firstframe_gray=cv2.cvtColor(firstframe, cv2.COLOR_BGR2GRAY)
	frame = cv2.resize(org_frame,(w,h),interpolation=cv2.INTER_AREA)
	gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	imgDelta = cv2.absdiff(gray_img,firstframe_gray) #//底圖差值 cv2.absdiff(firstframe,img)
	imgDelta = cv2.GaussianBlur(imgDelta,(5,5),0) #高斯模糊

	thresh = cv2.threshold(imgDelta, 30, 255, cv2.THRESH_BINARY)[1]  #二值化
	thresh = cv2.erode(thresh, None, iterations=2)  #消蝕
	thresh = cv2.dilate(thresh, None, iterations=6)  #膨脹
	thresh = cv2.dilate(thresh, None, iterations=6)  #膨脹
	thresh = cv2.dilate(thresh, None, iterations=6)  #膨脹
	
	filted_gimg = thresh & gray_img                      #重建去背後的前景(縮圖)	
	threshc=cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
	filted_cimg = threshc & frame                        #重建去背後的前景(縮圖)	
	
	pc.TIME_FILTER_FRAME+=(time.time()-tt)
	return filted_gimg,filted_cimg

	
def SeetaDetFace(img,oimg,minsize=0,maxsize=0):   #level=1 精準 , 0:嚴謹 (img=縮圖,oimg=原圖)
	import lib.pyseeta.detector as pd
	detector = pd.Detector(model_path="lib/pyseeta/model/seeta_fd_frontal_v1.0.bin")
	ratio=oimg.shape[0]//img.shape[0]
	MinSize = pc.MIN_FACE_SIZE//ratio if minsize==0 else minsize
	MaxSize = pc.MAX_FACE_SIZE//ratio if maxsize==0 else maxsize
	detector.set_min_face_size(MinSize)
	#detector.set_max_face_size(MaxSize)  #沒這個參數可用
	dets = detector.detect(img)
	scores=[]
	faces=[]
	for i, d in enumerate(dets):
		x=d.left
		y=d.top
		w=d.right-x
		h=d.bottom-y
		faces.append((x,y,w,h))
		scores.append(d.score)
		
	detector.release()
	return faces,scores
	

def SeetaDetBodyFaces(img,oimg,minsize=0,maxsize=0):   #輸出有身體的大頭照	
	faces,scores=SeetaDetFace(img,img,minsize,maxsize)
	ratio=oimg.shape[0]//img.shape[0]
	imgs,faces=GenImglistFromFace(faces,img,oimg,ratio)
	#for gg in imgs:
	#	ShowImgIfWinOS(gg)
	return imgs,faces,scores