# -*- coding: utf-8 -*-
import numpy as np
import cv2,time
from imp import reload
import imageio
import lib.v14.pc_config as pc; reload(pc)

class VideoSource:
	def __init__(self,fn,width=1280,height=720):
		tt=time.time()
		self.frame_cnt=0
		self.frame_index=0
		self.cam=None
		self.first_frame=None
		
		if fn==0 or fn==1 :
			self.vtype = "cam"
			self.cam = cv2.VideoCapture(fn)
			self.cam.isOpened()
			self.cam.set(3,width) # 修改解析度 寬
			self.cam.set(4,height) # 修改解析度 高
		else:
			self.vtype = "file"
			self.cam=imageio.get_reader(fn,'ffmpeg')
			try:
				self.frame_cnt=len(self.cam)
			except RuntimeError:
				self.frame_cnt=1000
				pass
		pc.TIME_GET_FRAME+=(time.time()-tt)
		
				
	def GetFrame(self):
		tt=time.time()
		ret=True
		frame=None
		if self.vtype == "file":
			if self.frame_index < self.frame_cnt:
				try:
					frame = self.cam.get_data(self.frame_index)
				except RuntimeError:
					ret=False
					pass
		else:
			ret, bgrframe = self.cam.read()
			if ret==True: frame = cv2.cvtColor(bgrframe, cv2.COLOR_BGR2RGB)
			
		if self.frame_index==0 : self.first_frame=frame
		self.frame_index+=1
		
		if frame is None:
			ret=False
			#self.cam.close()
		pc.TIME_GET_FRAME+=(time.time()-tt)
		return ret,frame
    
def GetAvgFrame(fn):
	import imageio
	ims = []
	cam=imageio.get_reader(fn,'ffmpeg')
	try:
		for im in enumerate(cam):
			ims.append(im[1])
	except RuntimeError:
		pass
	return ims	
		
		
def GetVideoFrame(fn): #很吃記憶體
	import imageio
	ims = []
	cam=imageio.get_reader(fn,'ffmpeg')
	try:
		for im in enumerate(cam):
			ims.append(im[1])
	except RuntimeError:
		pass
	return ims

def ScaleTo720p(img):
	return cv2.resize(img,(1280,720),interpolation=cv2.INTER_LINEAR)
	
	
def ShowImgIfWinOS(img):
	import platform
	from skimage import io
	if platform.system()=="Windows":
		#cv2.imshow("face", img) #show在外視窗
		tt=time.time()
		io.imshow(img)
		io.show()
		pc.TIME_UI_SHOWN+=(time.time()-tt)

		
def ShowCVVideoIfWinOS(img,tittle="Video"):
	import platform
	if platform.system()=="Windows":
		tt=time.time()
		cv2.imshow(tittle, img) #show在外視窗			
		cv2.waitKey(1)
		pc.TIME_UI_SHOWN+=(time.time()-tt)		
		
		
def ShowVideoIfWinOS(img,tittle="Video"):
	import platform
	if platform.system()=="Windows":
		tt=time.time()
		if len(img.shape) > 2:
			destRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			cv2.imshow(tittle, destRGB) #show在外視窗
		else:
			cv2.imshow(tittle, img) #show在外視窗			
		cv2.waitKey(1)
		pc.TIME_UI_SHOWN+=(time.time()-tt)

		
def BeepIfWinOS(stype=0):
	import platform
	if platform.system()=="Windows":
		import winsound
		if stype==0:	
			winsound.Beep(600,50)
		else:
			winsound.Beep(1000,300)


def ReadImg(fn):
	from skimage import io
	img = io.imread(fn) #使用skimage的io讀取圖片
	return img

	
def SaveImg(img,fn):
	#io.imsave(img,fn)
	destRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	cv2.imwrite(fn,destRGB, [cv2.IMWRITE_JPEG_QUALITY, 100])


def ResizeImg(img,tow,toh):
	if img.shape==2:
		h, w = img.shape
	else:
		h, w, c = img.shape
	new = cv2.resize(img[0:h,0:w],(tow,toh),interpolation=cv2.INTER_LINEAR) #  INTER_CUBIC INTER_LINEAR INTER_AREA
	return new
	
	
def ToGrayImg(img):
	return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)			
						
			
def ImgAdd(src,img,type="right"):
	from PIL import Image,ImageDraw
	if src is None : return img
	if img is None : return src
	h, w, c = src.shape
	nh,nw,nc = img.shape
	if type=="right":
		target = Image.new('RGB', (w+nw, max(h,nh)))
		target.paste(Image.fromarray(src), (0, 0, w, h))
		target.paste(Image.fromarray(img), (w, 0, w+nw, nh))
	elif type=="buttom":
		target = Image.new('RGB', (max(w,nw), h+nh))
		target.paste(Image.fromarray(src), (0, 0, w, h))
		target.paste(Image.fromarray(img), (0, h, nw, nh+h))		
	return np.array(target, dtype=np.uint8)

def ShowAllImg(imgs):
	img_total=None
	for i in range(len(imgs)):
		img_total=ImgAdd(img_total,imgs[i],type="right")
	return img_total

	
def ImgsToImgArray(imgs): #將以拚好的圖還原成array
	h, w, c = imgs.shape
	cnt=w//pc.SYSTEM_IMG_WIDTH
	ret=[]
	for i in range(cnt):
		uni = imgs[0:pc.SYSTEM_IMG_HEIGHT,pc.SYSTEM_IMG_WIDTH*i:pc.SYSTEM_IMG_WIDTH*(i+1)]
		ret.append(np.array(uni, dtype=np.uint8))
	return ret


def TextOut(frame,text,locat=(0,0),size=1.5,color=(255, 255,100),bold=2)	:
	frame=cv2.putText(frame,text,locat,cv2.FONT_HERSHEY_SIMPLEX,size,color,bold)
	return frame
	

def GenerateResultFrame(final_faces,col=6,w=1280,h=720):
	from PIL import Image,ImageDraw
	import numpy as np
	img_cnt=len(final_faces)
	row= img_cnt//col
	allimg=None
	for i in range(0,row+1):
		rowimg=None
		for j in range(0,col):
			if i*col+j>=img_cnt:
				break
			rowimg=ImgAdd(rowimg,final_faces[i*col+j],type="right")
		allimg=ImgAdd(allimg,rowimg,type="buttom")

	
	target = Image.new('RGB', (w, h))
	if not(allimg is None):
		h, w, c = allimg.shape
		target.paste(Image.fromarray(allimg), (0, 0, w, h))

	return np.array(target, dtype=np.uint8)
	
	
def ShowImgToSlack(img,chan="#another_people_cnt",topic="",comment="",body=""):
	import time,os
	from slacker import Slacker
	token_astra="xoxp-58923446293-73927282641-126459541009-fe5c783a4c35952d96f730e4d311d561"
	token_dbgmiii="xoxp-6046862866-6046758996-117696788384-ca72ac38459d83316826c1e029ec80cd"
	slack = Slacker(token_dbgmiii)
	timestamp = time.time()
	fn=str(timestamp)+".jpg"
	SaveImg(img,fn)
	ret=slack.files.upload(fn,channels=chan,title=topic,initial_comment=comment)
	if body!="": slack.chat.post_message(chan, "```"+body+"```")
	os.remove(fn)	
	
	
def MkDir(path):
	import os                # 引入模块
	path=path.strip()        # 去除首位空格
	path=path.rstrip("\\")   # 去除尾部 \ 符号
	isExists=os.path.exists(path)
	if not isExists: os.makedirs(path)

def CloseAllIfWinOS():	
	import platform	
	if platform.system()=="Windows":	
		cv2.destroyAllWindows()

		

	
	
	
	
	
	