 # -*- coding: utf-8 -*-
import face_recognition as fr
import lib.json_log as log
import numpy as np
import sys,os,time,dlib,cv2
#import redis
import json
from imp import reload
import lib.v14.pc_util as pu; reload(pu)

'''
  輸入 
  python api_face_recogn.py [P1] [P2] [P3]
  
  - P1: 人臉向量的Json檔 (一個group一檔)
  - P2: 輸入圖檔的path
  - P3: 輸出圖檔的path (人臉加框,若有辨識到,在左上角顯示人名)
  
  輸出
  {
    "error_msg": "none",
    "detected_name": [
            "Tommy"
    ],
    "process_time": 1.115
  }
'''	

P1_DEFAULT="face_cache/home.json"
P2_DEFAULT="queue_folder/home/multi_person.jpg"
P3_DEFAULT="queue_folder/home/Output.jpg"
 
l={}

def add_person(person_list,face_name,face_key,age=0,gender=0,emotion=0):
	d={}
	d["face_key"]=face_key
	d["gender"]=gender
	d["age"]=age
	d["emotion"]=emotion
	person_list[face_name]=d
	return person_list


def stru(s):
	return str(s,encoding = "utf-8")


def get_face_db_from_redis(redis_info,location_info):
	face_names=[]
	face_vectors=[]
	face_id_lists=[]
	pool=redis.ConnectionPool(host=redis_info.split(":")[0], port=redis_info.split(":")[1])
	r=redis.Redis(connection_pool=pool)
	flist=stru(r.get(location_info)).split(",")
	for fid in flist:
		face_names.append(stru(r.hget(REDIS_FACE_ID_PREFIX+fid,"name")))
		face_vectors.append(eval(stru(r.hget(REDIS_FACE_ID_PREFIX+fid,"face_vector"))))
		face_id_lists.append(fid)
	#print (face_names)
	#print (face_vectors)
	return face_names,face_vectors,face_id_lists


def get_face_db_from_json(file_path):
	face_names=[]
	face_vectors=[]
	face_id_lists=[]
	with open(file_path, 'r',encoding="utf-8") as f:
		flist = json.load(f)	
	for fid in flist['database']:
		face_names.append(fid['name'])
		face_vectors.append(eval(fid['face_vector']))
		face_id_lists.append(fid)	
	#print (face_names)
	#print (face_vectors)
	return face_names,face_vectors,face_id_lists


'''
	locations 格式
	(top,right,bottom,left)
'''
def get_face_vector(img_path):
	img_size=64
	face_imgs=[]
	img=fr.load_image_file(img_path)
	locations = fr.face_locations(img)
	encodings = fr.face_encodings(img, locations)
	if len(locations)!=len(encodings):return
	img_h, img_w, _ = np.shape(img)

	for (top,right,bottom,left) in locations:
		x1, y1, x2, y2, w, h = left, top, right+1, bottom+1, right-left, bottom-top
		xw1 = max(int(x1 - 0.4 * w), 0)
		yw1 = max(int(y1 - 0.4 * h), 0)
		xw2 = min(int(x2 + 0.4 * w), img_w - 1)
		yw2 = min(int(y2 + 0.4 * h), img_h - 1)
		face = np.empty( (1, img_size, img_size, 3) )
		face[0,:,:,:] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
		face_imgs.append(face)

	

	return face_imgs,locations,encodings


'''
	face_names: 目前db已有的人名
	face_vectors: 目前db已有的人名對應的向量
	face_id_lists: 目前db已有的人名對應的 face_id
	test_locations: 在待測圖片中找到的一個或多個face座標
	test_encodings: 在待測圖片中找到的一個或多個face座標對應的向量
'''
def get_face_profile(face_names,face_vectors,face_id_lists,test_face_imgs,test_locations,test_encodings):
	result_person=[] #{}
	result_locat=[]
	fid=""
	for i in range(len(test_encodings)):
		# See if the face is a match for the known face(s)
		match = fr.compare_faces(face_vectors, test_encodings[i], tolerance=0.4)  #越小越嚴格 0.5在人多容易重複
		cnt=0
		for j in range(len(match)):
			name="?"
			if match[j]==True: 
				name=face_names[j]
				fid=j	#face_id_lists[j]
				cnt+=1
				break
		result_person.append(name)
	return result_person,result_locat


def puttext(img,string,x,y,color):
	from PIL import Image, ImageDraw, ImageFont
	cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2和PIL中颜色的hex码的储存顺序不同
	pilimg = Image.fromarray(cv2img)

	# PIL图片上打印汉字
	draw = ImageDraw.Draw(pilimg) # 图片上打印
	abspath=os.path.split(os.path.realpath(__file__))[0]+"/font.ttc"
	#print(abspath)
	font = ImageFont.truetype(abspath, 20, encoding="utf-8") # 参数1：字体文件路径，参数2：字体大小
	y-=22
	draw.text((x+1, y+1), string, (0, 0, 0), font=font) # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
	draw.text((x+1, y-1), string, (0, 0, 0), font=font) # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
	draw.text((x-1, y+1), string, (0, 0, 0), font=font) # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
	draw.text((x-1, y-1), string, (0, 0, 0), font=font) # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
	draw.text((x, y), string, color, font=font) # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体

	# PIL图片转cv2 图片
	cv2charimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
	return cv2charimg



def mark_face(img_path,locations,names):
	oimg=fr.load_image_file(img_path)
	h,w,c=oimg.shape
	if h>=w:
		ratio=w/300
		img=cv2.resize( oimg, (300, round(h/ratio)),interpolation=cv2.INTER_LINEAR)
	else:
		ratio=h/300
		img=cv2.resize( oimg, (round(w/ratio), 300),interpolation=cv2.INTER_LINEAR)

	for i in range(len(locations)):
		y1,x1,y2,x2=locations[i]
		cv2.rectangle(img, (round(x1/ratio), round(y1/ratio)), (round(x2/ratio), round(y2/ratio)), (0, 0, 0), 3, 1)
		if names[i]=='?':
			cv2.rectangle(img, (round(x1/ratio), round(y1/ratio)), (round(x2/ratio), round(y2/ratio)), (255, 0, 0), 1, 1)
			img=puttext(img,names[i],round(x2/ratio),round(y1/ratio),(0, 0, 255))
		else:
			cv2.rectangle(img, (round(x1/ratio), round(y1/ratio)), (round(x2/ratio), round(y2/ratio)), (0, 255, 0), 1, 1)
			img=puttext(img,names[i],round(x2/ratio),round(y1/ratio),(0, 255, 0))
		#cv2.putText(img, names[i], (round(x2/ratio),round(y1/ratio)-5), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 2, 1)
		#cv2.putText(img, names[i], (round(x2/ratio),round(y1/ratio)-5), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1, 1)

	return img


def api(P1,P2,P3):
	errmsg=""
	detected_person=[] #{}
	opt_person=[]
	start_ts=time.time()

	try:
		#face_names,face_vectors,face_id_lists=get_face_db_from_redis(P1,P2)
		face_names,face_vectors,face_id_lists=get_face_db_from_json(P1)
		#print(face_names)
		#print(face_vectors)
		if os.path.isfile(P2):
			test_face_imgs,test_locations,test_encodings=get_face_vector(P2)
			if len(test_encodings)>0:
				detected_person,_=get_face_profile(face_names,face_vectors,face_id_lists,test_face_imgs,test_locations,test_encodings)
				img=mark_face(P2,test_locations,detected_person)
				pu.SaveImg(img,P3)
				pu.ShowImgIfWinOS(img)
		else:
			errmsg="image not exist"
	except Exception as e: 
		errmsg=str(e)

	for s in detected_person:
		if s!='?': opt_person.append(s)
		
	log.Json_log(l,"error_msg",errmsg)
	log.Json_log(l,"detected",opt_person)
	log.Json_log(l,"process_time",round(time.time()-start_ts,3))
	log.Json_print(l)


def main():
	if len(sys.argv)==4:
		api(sys.argv[1],sys.argv[2],sys.argv[3])
	else:
		api(P1_DEFAULT,P2_DEFAULT,P3_DEFAULT)


if __name__ == '__main__':	
	main()




'''
#redis.Redis(host='192.168.56.100', port=6379)
# P1_DEFAULT="127.0.0.1:6379"
# P2_DEFAULT="cv:face_id_list:TEST_ID"
# P3_DEFAULT="/home/tommy/vca-facerecog/src/cvpy/FaceDB/tommy3.jpg"
# REDIS_FACE_ID_PREFIX="cv:face_id:"

def main():
	l={}
	detected=add_person({},"Chris","6e2756d1-2511-4f3d-a3af-5dfe6ee483e0:1513671476","male","37","hungry")
	detected=add_person(detected,"Tommy","a4d277c3-6739-4bc3-b996-ca842607210d:1513671476","male","35","happy")
	l["detected"]=detected
	l["error_msg"]=""
	l["process_time"]=0.5
	log.Json_print(l)

def main_test():
	l={}
	#'{"detected": ["chris": {"face_key": "0ec431b3-171b-4602-8948-2c4bfcd136fa:1513671476","gender": "male","age": "37","emotion": "hungry"},],"error_msg": "","process_time": 0.775}'
	user={"Chris":{
		"face_key":"6e2756d1-2511-4f3d-a3af-5dfe6ee483e0:1513671476",
		"gender": "male",
		"age": "37",
		"emotion": "hungry"},
		"Tommy":{
		"face_key":"a4d277c3-6739-4bc3-b996-ca842607210d:1513671476",
		"gender": "male",
		"age": "30",
		"emotion": "happy"}}
	detected=user
	log.Json_log(l,"detected",detected)
	log.Json_log(l,"error_msg","")
	log.Json_log(l,"process_time",0.5)	
	log.Json_print(l)
'''
