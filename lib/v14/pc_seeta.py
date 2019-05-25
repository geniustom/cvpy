# -*- coding: utf-8 -*-
from imp import reload
import lib.v14.pc_det as pd; reload(pd)
import lib.v14.pc_config as pc; reload(pc)


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
	imgs,faces=pd.GenImglistFromFace(faces,img,oimg,ratio)
	#for gg in imgs:
	#	ShowImgIfWinOS(gg)
	return imgs,faces,scores