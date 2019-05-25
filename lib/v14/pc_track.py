# -*- coding: utf-8 -*-
import numpy as np
import cv2,dlib,time
from imp import reload
import lib.v14.pc_det as pdd; reload(pdd)
import lib.v14.pc_util as puu; reload(puu)
import lib.v14.pc_config as pcc; reload(pcc)


'''
class face_info:
	def __init__(self,x,y,t,fimg,score):
		self.x = x
		self.y = y
		self.t = t
		self.score=score
		self.fimg = fimg	
'''
	
class score_face:
	def __init__(self,face,f_index):
		self.face_cnt = 0
		self.last_face_xy=face
		self.last_det_t=f_index
		self.fimg  = []
		self.score = []

class Person:
	def __init__(self):
		self.person_cnt = 0
		self.total_face_cnt = 0
		self.sf = []
		self.tracker=[]
		self.track_box=[]	

	def get_bbox(self,face):
		fx,fy,fw,fh=face
		#b=tuple(face)
		#b=tuple([int(fx-fw/2),int(fy-fh/2),fw*2,fh*3])
		#b=tuple([fx-0.2*fw,fy-0.2*fh,fw*1.4,fh*2])
		#b=tuple([int(fx-0.1*fw),int(fy-0.1*fh),int(fw*1.2),int(fh*2.5)])
		b=tuple([fx,fy,fw,int(fh*pcc.TRACK_BODY_HEIGHT)])
		return b
		
	def get_face_from_bbox(self,bbox):
		bx,by,bw,bh=bbox
		face=(bx,by,bw,int(bh/pcc.TRACK_BODY_HEIGHT))
		return face
		
	def new_tracker(self,frame,face,f_index):
		tt=time.time()
		b=self.get_bbox(face)
		if pcc.TRACK_ALGORITHM=="DLIB":
			t=dlib.correlation_tracker()
			dx,dy,dr,db=b[0],b[1],b[0]+b[2],b[1]+b[3]
			t.start_track(frame, dlib.rectangle(dx,dy,dr,db))
		else:
			t=cv2.Tracker_create(pcc.TRACK_ALGORITHM)
			t.init(frame, b)
		
		sf=score_face(face,f_index)
		pcc.TIME_BODY_TRACK+=(time.time()-tt)
		return t,b,sf

	def new_person(self,frame,face,f_index,fimg,score):
		t,b,sf=self.new_tracker(frame,face,f_index)
		sf.last_face_xy=face
		sf.last_det_t=f_index
		sf.fimg.append(fimg)
		sf.score.append(score)
		#print (t,b)
		self.tracker.append(t)
		self.track_box.append(b)
		self.sf.append(sf)
		self.person_cnt+=1
		
	def is_same_face(self,face,track_box):
		fx,fy,fw,fh=face
		tx,ty,tw,th=track_box
		#print(fx,fy,fw,fh,tx,ty,tw,th)
		fc=np.array([ int(fx+fw/2) , int(fy+fh/2) ])
		r=int(fw/2)
		#result= fc[0]>=tx-r and fc[0]<=tx+tw+r and fc[1]>=ty-r and fc[1]<=ty+th+r
		result= fc[0]>=tx-r and fc[0]<=tx+tw+r and fc[1]>=ty-r and fc[1]<=ty+th
		return result
			
	def add_to_person(self,frame,face,fimg,f_index,score):
		#print('----------------')
		for i in range(self.person_cnt):
			if self.tracker[i] is None : continue
			if self.is_same_face(face,self.sf[i].last_face_xy):
				t,b,_=self.new_tracker(frame,face,f_index)
				self.tracker[i]=t
				self.track_box[i]=b
				self.sf[i].last_face_xy=face
				self.sf[i].last_det_t=f_index
				self.sf[i].fimg.append(fimg)
				self.sf[i].score.append(score)
				puu.BeepIfWinOS()
				#print("face:",face,"in Person",i+1,"box:",self.sf[i].last_face_xy,"to",b)
				return True
		
		if pdd.IsPerson(fimg,score=pcc.DETECT_SCORE):
			self.new_person(frame,face,f_index,fimg,score)
			puu.ShowImgIfWinOS(fimg)
			puu.BeepIfWinOS(stype=1)
			print("Detect new people, now cnt:",self.person_cnt,face)
			return True
		else:
			return False

	def clear_tracker(self,i,reason):
		print("Delete",i+1,reason)
		self.tracker[i]=None
		self.track_box[i]=None
		self.sf[i].last_face_xy=(0,0,0,0)
				
	def check_bbox_vaild(self,i,frame,f_index):
		b=self.track_box[i]
		(bx,by,bw,bh)= (int(b[0]),int(b[1]),int(b[2]),int(b[3]))
		h,w,_=frame.shape

		# 追蹤人體的眶若多少個frame沒有臉就停止track
		if f_index-self.sf[i].last_det_t>pcc.TRACK_FACE_TIMEOUT:self.clear_tracker(i,"face track timeout");return		
		# 當超出邊界的時候停止追蹤
		if bx<=0 or by<=0 or by+bh>=h or bx+bw>=w :self.clear_tracker(i,"out bound");return
		# 當框的大小不符合臉的範圍時停止追蹤
		#if bw<(pcc.MIN_FACE_SIZE/1.2) or bw>(pcc.MAX_FACE_SIZE*1.2): self.clear_tracker(i,"window not match face");return
		# 當下方都是黑的時候停止追蹤
		#if frame[by+bh,bx].sum()==0 and frame[by+bh,bx+bw].sum()==0: self.clear_tracker(i,"track buttom range black");return
		# 當四個角都是黑的時候停止追蹤
		if frame[by,bx].sum()==0 and frame[by+bh,bx].sum()==0 and frame[by,bx+bw].sum()==0 and frame[by+bh,bx+bw].sum()==0: self.clear_tracker(i,"track all range black");return

		#print("Box",i+1,": [",bx,by,"]",frame[by,bx],frame[by+bh,bx],frame[by,bx+bw],frame[by+bh,bx+bw])
		
		if self.is_same_face(self.sf[i].last_face_xy,(bx,by,bw,bh)):
			#print("face:",self.sf[i].last_face_xy,"in Person",i+1,"refresh from",self.get_face_from_bbox(bbox))
			self.sf[i].last_face_xy=self.get_face_from_bbox((bx,by,bw,bh))
	
		
	def AddFace(self,frame,face,fimg,f_index,debug=pcc.IS_DEBUG,ratio=1,score=0):
		r=ratio
		fx,fy,fw,fh=face
		org_face=(int(fx*r),int(fy*r),int(fw*r),int(fh*r))
		return self.add_to_person(frame,org_face,fimg,f_index,score)


		
	def RefreshTracker(self,frame,f_index):
		tt=time.time()
		for i in range(self.person_cnt):
			if self.tracker[i] is None : continue
			if pcc.TRACK_ALGORITHM=="DLIB":
				self.tracker[i].update(frame)
				d=self.tracker[i].get_position()
				self.track_box[i]=(d.left(),d.top(),d.right()-d.left(),d.bottom()-d.top())
			else:
				ok, self.track_box[i] = self.tracker[i].update(frame) # Update tracker
				
			self.check_bbox_vaild(i,frame,f_index) # Check tracker vaild
			#print(self.track_box[i])
		#print("------------------")
		pcc.TIME_BODY_TRACK+=(time.time()-tt)
		
	def DrawTrackFrame(self,frame,ratio=1):
		tt=time.time()
		r=ratio
		for i in range(self.person_cnt):
			if self.tracker[i] is None : continue
			b=self.track_box[i]
			bbox=(int(b[0]),int(b[1]),int(b[2]),int(b[3]))		
			p1 = (bbox[0]*r,bbox[1]*r)
			p2 = (bbox[0]*r + bbox[2]*r,bbox[1]*r + bbox[3]*r)		
			font=cv2.FONT_HERSHEY_SIMPLEX #使用默认字体
			frame=cv2.putText(frame,str(i+1),(int(bbox[0]*r),int(bbox[1]*r)-5),font,0.6,(100, 255,100 ),1)
			#添加文字，1.2表示字体大小，（0,40）是初始的位置，(255,255,255)表示颜色，2表示粗细
			cv2.rectangle(frame, p1, p2, (0, 255,0 ),2)
		pcc.TIME_DRAW_INFO+=(time.time()-tt)
		return frame
		
		

def CheckRealPeople(p=Person(),sco=0,debug=pcc.IS_DEBUG):
	realface=[]
	for i in range(p.person_cnt):
		#先產生此回合所有第一階段排出相同的人臉
		checkimg=[]
		for j in range(len(p.sf[i].fimg)):
			checkimg.append(p.sf[i].fimg[j])
		#取出品質最好的一張
		if debug:print("------ face ",i,"------")
		have_face,img,score=pdd.DlibGetBestFace(checkimg,score=sco,level=0)
		if have_face==True:
			realface.append(img)
	
	if debug:print(len(realface))
	return realface

		
		
'''
	def RefreshTracker(self,frame,f_index):
		for i in range(self.person_cnt):
			if f_index-self.sf[i].last_det_t>TRACK_FACE_TIMEOUT:
				self.tracker[i]=None
				self.track_box[i]=None
				self.sf[i].last_face_xy=(0,0,0,0)
				continue
			# Update tracker
			ok, self.track_box[i] = self.tracker[i].update(frame)
			# Draw bounding box
			if ok:
				b=self.track_box[i]
				(bx,by,bw,bh)= (int(b[0]),int(b[1]),int(b[2]),int(b[3]))
				h,w,_=frame.shape
				
				if bx>0 and by>0 and by+bh<h and bx+bw<w :		#確認在範圍內			
					if self.is_same_face(self.sf[i].last_face_xy,(bx,by,bw,bh)):
						#print("face:",self.sf[i].last_face_xy,"in Person",i+1,"refresh from",self.get_face_from_bbox(bbox))
						self.sf[i].last_face_xy=self.get_face_from_bbox((bx,by,bw,bh))
'''		
		