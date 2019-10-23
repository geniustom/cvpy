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

def get_face_score(img):
	det, score, idx = pd.detector.run(img,0,-1)
	if len(det)==0 : return 0  #沒有臉要排除
	if len(det)>1 : return 0   #一張圖有2張臉要排除
	if idx[0]==3 : return 0    #idx=3代表人臉角度太斜要排除
	return score[0]


def api(P1):
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
	Json_print(result)
	#print(result)

def main():
	if len(sys.argv)==2:
		api(sys.argv[1])
	else:
		api(P1_DEFAULT)


if __name__ == '__main__':	
	main()