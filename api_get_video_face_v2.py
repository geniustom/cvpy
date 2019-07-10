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

l={}
frame_step = 1
pw,ph=	480,270	 #192,108 #320,180 #144,81 #96,54 #120,68 #160,90 #1280,720 #640,360 #480,270
conf_threshold = 0.99
face_score=0.5
modelFile = "lib/model/opencv_face_detector_uint8.pb"
configFile = "lib/model/opencv_face_detector.pbtxt"
net=None
P1_DEFAULT="./test_video/t11.flv"  #"SWC002s9DYhh_20190307_0292.flv" #"SWC002s9DYhh_20181129_0618.flv" #"SWC002s9DYhh_20190123_0578.flv" "SWC002s9DYhh_20190124_0208"
P2_DEFAULT="./queue_folder/home/Output.jpg"
P3_DEFAULT="6000"


def main():
	if len(sys.argv)==4:
		api(sys.argv[1],sys.argv[2],sys.argv[3])
	else:
		api(P1_DEFAULT,P2_DEFAULT,P3_DEFAULT)


def ClipBestFace(img): #輸出最適比例的大頭照
	ah,aw=img.shape[0],img.shape[1]
	[(t,r,b,l)] = fr.face_locations(img)
	w,h=r-l,b-t
	cx,cy=round(l+(w/2)) , round(t+(h/2))
	ss=round(min(cx,(aw-cx),cy,(ah-cy)/1.5))
	nimg = img[cy-ss:round(cy+ss*1.5),cx-ss:round(cx+ss)]
	nimg = cv2.resize( nimg, (200,250),interpolation=cv2.INTER_LINEAR)

	return nimg

def DetBodyFaces(img,oimg,minsize=0,maxsize=0):   #輸出有身體的大頭照
	imgs=[]
	faces=[]
	frame,f=detectFaceOpenCVDnn(img)
	rt=oimg.shape[0]//img.shape[0]
	oh,ow=oimg.shape[0],oimg.shape[1]
	#''' 標準大頭照
	
	for (x,y,w,h) in f:
		fl = (x*rt)-(w*rt)//2
		ft = (y*rt)-(h*rt)//2
		#print(f,fl,ft,ow,oh)
		#if fl<0 or fl>ow or ft<0 or ft>oh: continue #多這行會loss很多frame
		fw = (w*2*rt)
		fh = (h*3*rt)
		
		#body = cv2.resize(oimg[ft:ft+fh,fl:fl+fw],(pc.SYSTEM_IMG_WIDTH,pc.SYSTEM_IMG_HEIGHT),interpolation=cv2.INTER_LINEAR) #  INTER_CUBIC INTER_LINEAR INTER_AREA
		#body = cv2.resize(oimg[ft:ft+fh,fl:fl+fw],(200,round(fw/fh*300)),interpolation=cv2.INTER_LINEAR) #  INTER_CUBIC INTER_LINEAR INTER_AREA
		#body = oimg[ft:ft+fh,fl:fl+fw] 
		body = oimg[max(0,ft):min(oh,ft+fh),max(0,fl):min(ow,fl+fw)] 
		if body.shape[1]>400:
			body = cv2.resize( body, (round(body.shape[1]/2), round(body.shape[0]/2)),interpolation=cv2.INTER_LINEAR)
		body = body[:, :, ::-1]
		#body=oimg[ft:ft+fh,fl:fl+fw]
		imgs.append(body)
		faces.append(f)

	return frame,imgs,faces

def detectFaceOpenCVDnn(frame):
	global net,conf_threshold
	if net==None: net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
	frameOpencvDnn = frame.copy()
	#frameOpencvDnn = cv2.cvtColor(frameOpencvDnn,cv2.COLOR_BGR2GRAY)
	frameHeight = frameOpencvDnn.shape[0]
	frameWidth = frameOpencvDnn.shape[1]
	blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (pw, ph), [104, 117, 123], False, False)

	net.setInput(blob)
	detections = net.forward()
	bboxes = []
	for i in range(detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > conf_threshold:
			x1 = int(detections[0, 0, i, 3] * frameWidth)
			y1 = int(detections[0, 0, i, 4] * frameHeight)
			x2 = int(detections[0, 0, i, 5] * frameWidth)
			y2 = int(detections[0, 0, i, 6] * frameHeight)
			bboxes.append([x1, y1, x2-x1, y2-y1])
			# for debug
			cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)
	return frameOpencvDnn, bboxes


def api(P1,P2,P3):
	imgs=[]
	totalimgs=[]
	errmsg=""
	start_ts=time.time()
	have_face=False
	best_score=0
	det_time=0
	fpsOpencvDnn=0
	try:
		cap = cv2.VideoCapture(P1)
		hasFrame, frame = cap.read()
	
		#vid_writer = cv2.VideoWriter('output-dnn-{}.avi'.format(str(source).split(".")[0]),cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame.shape[1],frame.shape[0]))
		frame_count = 0
		tt_opencvDnn = 0
		while(frame_count<int(P3)):
			hasFrame, frame = cap.read()
			if not hasFrame: break
			frame_count += 1
			if frame_count%frame_step!=0: continue
	
			t = time.time()
			# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
			#sframe = cv2.resize(frame,(pw,ph),interpolation=cv2.INTER_AREA)
			#sframe=frame
			#sframe = sframe[:, :, ::-1]
			#sframe = cv2.cvtColor(sframe,cv2.COLOR_BGR2GRAY)
	
			sframe,imgs, bboxes = DetBodyFaces(frame,frame)	#dnn 作法
			#imgs,bboxes=pd.CvDetBodyFaces(sframe,frame)	#haar 作法 face+body
			#imgs,bboxes=pd.DeepDetBodyFaces(sframe,frame)	#haar 作法 face+body
			#bboxes=pd.CvDetFace(sframe,frame)	#haar 作法 face
			
			tt_opencvDnn += time.time() - t
			fpsOpencvDnn = frame_count / tt_opencvDnn
			label = "FPS : {:.2f}".format(fpsOpencvDnn)
			#print(label)
			pu.ShowCVVideoIfWinOS(sframe)
			for img in imgs:	
				#pu.ShowImgIfWinOS(img)
				totalimgs.append(img)
			#pu.ShowImgIfWinOS(best_img)
			#cv2.putText(outOpencvDnn, label, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3, cv2.LINE_AA)
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
	log.Json_log(l,"process_time_detface",round(det_time-start_ts,3))
	log.Json_log(l,"process_time_bestface",round(time.time()-det_time,3))
	log.Json_print(l)
	cv2.destroyAllWindows()






if __name__ == '__main__':	
	main()
