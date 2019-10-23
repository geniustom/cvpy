 # -*- coding: utf-8 -*-

from __future__ import division
import cv2,math
import face_recognition as fr
import lib.json_log as log
import sys,os,time,platform
from imp import reload
import lib.v14.pc_det as pd; reload(pd)
import json


P1_DEFAULT="./FaceDB/"


def Json_print(obj,sort=True):
	print(json.dumps(obj,sort_keys=sort,indent=4,ensure_ascii=False))


def clip_best_face(img): #輸出最適比例的大頭照
	try:
		ah,aw=img.shape[0],img.shape[1]
		[(t,r,b,l)] = fr.face_locations(img)
		w,h=r-l,b-t
		cx,cy=round(l+(w/2)) , round(t+(h/2))
		ss=round(min(cx,(aw-cx),cy,(ah-cy)/1.5))
		nimg = img[cy-ss:round(cy+ss*1.5),cx-ss:round(cx+ss)]
		nimg = cv2.resize( nimg, (200,250),interpolation=cv2.INTER_LINEAR)
	except:
		nimg=img
	return nimg

def get_face_score(img):
	img=clip_best_face(img)
	det, score, idx = pd.detector.run(img,0,-1)
	print(det,score,idx)
	if len(det)==0 : return 0  #沒有臉要排除
	if len(det)>1 : 
		cnt=0
		for s in score: #score<0的不是真正的臉
			if s>0: cnt+=1
		if cnt>1: return 0   #一張圖有2張臉要排除
	#if idx[0]==3 : return 0    #idx=3代表人臉角度太斜要排除
	return round(score[0],3)


def api(P1):
	start_ts=time.time()
	files = next(os.walk(P1))[2]
	result={}
	result_img=[]
	for f in files:
		r={}
		img=fr.load_image_file(P1+f)
		r['filename']= f
		r['score']= get_face_score(img)
		result_img.append(r)
	result['facescores']=result_img
	result['process_time']=round(time.time()-start_ts,3)
	Json_print(result)
	#print(result)

def main():
	if len(sys.argv)==2:
		api(sys.argv[1])
	else:
		api(P1_DEFAULT)


if __name__ == '__main__':	
	main()