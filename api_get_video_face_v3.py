 # -*- coding: utf-8 -*-
from __future__ import division
import cv2
import face_recognition as fr
import lib.json_log as log
import numpy as np
import sys,os,time,platform
from imp import reload
import lib.v14.pc_det as pd; reload(pd)
import lib.v14.pc_util as pu; reload(pu)
import lib.v14.pc_config as pc; reload(pc)
import face_recognition as fr

'''
Note: 
	置信度拉高較不會有雜圖,並可減少時間
	若效果不好可用大圖來換精度
'''

##################### 可變動參數區 #####################
#192,108 #320,180 #144,81 #96,54 #120,68 #160,90 #1280,720 #960,540 #640,360 #480,270
pw,ph=	96,54	#做motion block的大小	 
#sw,sh= 640,360	#做FACE DETECTION的大小
sw,sh= 1280,720	#做FACE DETECTION的大小
#sw,sh= 1920,1080	#做FACE DETECTION的大小
face_score=0.1	#最低可允許的人臉分數
motion_size=0.05 	#0~1 浮點數,代表有效的motion block面積占比, 0.1 代表 1/10以上面積的motion才會偵測
frame_step = 3
########################################################

P1_DEFAULT="./test_video/red1.mp4"  #"./test_video/red1.mp4" "rtmp://104.155.222.173:1936/live/SWB000k6RnxS" #"SWC002s9DYhh_20190307_0292.flv" #"SWC002s9DYhh_20181129_0618.flv" #"SWC002s9DYhh_20190123_0578.flv" "SWC002s9DYhh_20190124_0208"
P2_DEFAULT="./queue_folder/home/Output.jpg"
P3_DEFAULT="500"
l={}

def main():
	if len(sys.argv)==4:
		api(sys.argv[1],sys.argv[2],sys.argv[3])
	else:
		api(P1_DEFAULT,P2_DEFAULT,P3_DEFAULT)


def ClipBestFace(img): #輸出最適比例的大頭照
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




def api(P1,P2,P3):
	imgs=[]
	totalimgs=[]
	errmsg=""
	have_face=False
	best_score=0
	det_time=0
	fpsOpencvDnn=0
	first_frame=[]
	background_frame=[]
	
	try:
		cap = cv2.VideoCapture(P1)
		hasFrame, frame = cap.read()
		#vid_writer = cv2.VideoWriter('output-dnn-{}.avi'.format(str(source).split(".")[0]),cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame.shape[1],frame.shape[0]))
		frame_count = 0
		tt_opencvDnn = 0
		start_ts=time.time()
		while(frame_count<int(P3)):
			tt=time.time()
			hasFrame, frame = cap.read()		
			pc.TIME_WAIT_FRAME+=(time.time()-tt)
			if not hasFrame: break
			if len(first_frame)==0:	first_frame=frame
	#			if frame_count%30==0: 
	#				background_frame=first_frame #background 每 30 frames refresh一次
	#				first_frame=frame
			frame_count += 1
			if frame_count%frame_step!=0: continue
	
			t = time.time()
			# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
			#sframe = cv2.resize(frame,(pw,ph),interpolation=cv2.INTER_AREA)
			#sframe=frame
			#sframe = sframe[:, :, ::-1]
			#sframe = cv2.cvtColor(sframe,cv2.COLOR_BGR2GRAY)
			#------- 去除背景
			cut_frame = cv2.resize(frame,(sw,sh),interpolation=cv2.INTER_AREA)
			gframe,cframe=pd.FilterFrame(frame,first_frame,pw,ph)
			cframe,motion_rect=pd.FindMotionRect(cut_frame,cframe,motion_size)
			#motion_rect=[cut_frame]
			#-------
			motion_bboxes=[]
			motion_imgs=[]
			#if len(motion_rect)==0: continue
			for m in motion_rect:
				#pu.ShowImgIfWinOS(m)
				#sframe,imgs, bboxes = pd.CVDnnDetBodyFaces(m,m)	#dnn 作法
				imgs,bboxes=pd.CvDetBodyFaces(m,m,CVminNeighbors=1,minsize=25,maxsize=200)	#haar 作法 face+body
				#imgs,bboxes=pd.DeepDetBodyFaces(m,m)	#haar 作法 face+body
				#bboxes=pd.CvDetFace(sframe,frame)	#haar 作法 face
				motion_bboxes=motion_bboxes+bboxes
				motion_imgs=motion_imgs+imgs
				#if len(motion_bboxes)>0: print(motion_bboxes)
			
			tt_opencvDnn += time.time() - t
			fpsOpencvDnn = frame_count / tt_opencvDnn
			label = "FPS : {:.2f}".format(fpsOpencvDnn)
			cv2.putText(cframe, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
			pu.ShowCVVideoIfWinOS(cframe)
			for img in motion_imgs:	
				#pu.ShowImgIfWinOS(img)
				totalimgs.append(img)
			#cv2.imshow("Face Detection Comparison", frame)
			#vid_writer.write(frame)
			if frame_count == 1: tt_opencvDnn = 0
	
	
		det_time=time.time()
		have_face,best_img,best_score = pd.DlibGetBestFace(totalimgs,score=face_score,debug=False)
		if have_face:
			pu.ShowImgIfWinOS(best_img)
			nimg=ClipBestFace(best_img)
			pu.ShowImgIfWinOS(nimg)
			pu.SaveImg(nimg,P2)
	except Exception as e: 
		errmsg=str(e)	
		
	log.Json_log(l,"p1",P1)
	log.Json_log(l,"p2",P2)
	log.Json_log(l,"p3",P3)
	log.Json_log(l,"fps",fpsOpencvDnn)
	log.Json_log(l,"processed_frames",frame_count)
	log.Json_log(l,"face_detected",have_face)
	log.Json_log(l,"face_score",best_score)
	log.Json_log(l,"error_msg",errmsg)
	
	log.Json_log(l,"0.process_time_wait_frame",round(pc.TIME_WAIT_FRAME,3))
	log.Json_log(l,"1.process_time_filter_frame",round(pc.TIME_FILTER_FRAME,3))
	log.Json_log(l,"2.process_time_motion_block",round(pc.TIME_MOTION_DET,3))
	log.Json_log(l,"3.process_time_detface",round(pc.TIME_FACE_DET,3))
	log.Json_log(l,"4.process_time_bestface",round(time.time()-det_time,3))
	log.Json_log(l,"5.process_time_ui_shown",round(pc.TIME_UI_SHOWN,3))
	log.Json_log(l,"9.process_time_total",round(time.time()-start_ts,3))
	log.Json_print(l)
	cv2.destroyAllWindows()






if __name__ == '__main__':	
	main()
