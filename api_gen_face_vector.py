 # -*- coding: utf-8 -*-
import face_recognition as fr
import lib.json_log as log
import numpy as np
import sys,os,time

'''
  輸入 
  python gen_face_vector.py [P1]
  - [P1] = image local path  // 待產生特徵向量的圖片

  輸出
  {
	"error_msg": "none",
	"face_vector": "[-0.10333242267370224, ... , -0.0052691707387566566]",
	"process_time": 0.413
  }

'''

l={}
P1_DEFAULT="./FaceDB/Tommy3.jpg"

def main():
	if len(sys.argv)==2:
		api(sys.argv[1])
	else:
		api(P1_DEFAULT)


def api(P1):
	vector=""
	errmsg=""
	start_ts=time.time()
	try:
		img=fr.load_image_file(P1)
		img_enc=fr.face_encodings(img)
		if len(img_enc)>0:
			vector=str(list(img_enc[0]))
		else:
			errmsg="no face in image"
	except Exception as e: 
		errmsg=str(e)
	log.Json_log(l,"face_vector",vector)
	log.Json_log(l,"error_msg",errmsg)
	log.Json_log(l,"process_time",round(time.time()-start_ts,3))
	log.Json_print(l)


if __name__ == '__main__':	
	main()