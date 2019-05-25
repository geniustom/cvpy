# -*- coding: utf-8 -*-
from imp import reload
import face_recognition as fr
import numpy as np

######################################## DB ########################################
def savePeopleFeature(did, timepoint_h, timepoint_d, timepoint_w, fn_list, vector,dbhost):
	import pymysql
	import datetime
	auth,server=dbhost.split("@")
	usr,pwd=auth.split(":")
	dbh,dbp,dbn=server.split(":")
	conn = pymysql.connect(host=dbh, port=int(dbp), user=usr, passwd=pwd, db=dbn)
	cur = conn.cursor()
	created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	try:
		for i in range(len(vector)):
			v=vector[i]
			fn=fn_list[i]
			sql = "INSERT INTO `people_feature` (`did`, `timepoint_h`, `timepoint_d`, `timepoint_w`, `filename`, `vector`, `created_at`) VALUES (%s, %s, %s, %s, %s, %s, %s)"
			cur.execute(sql, (did, timepoint_h, timepoint_d, timepoint_w, fn, v, created_at)) 
		conn.commit()
	except:
	    conn.rollback()
	    raise 
	finally:
	    cur.close()
	    conn.close()


def fetchPeopleFeature(did, period, timepoint,dbhost):
	import pymysql
	auth,server=dbhost.split("@")
	usr,pwd=auth.split(":")
	dbh,dbp,dbn=server.split(":")
	conn = pymysql.connect(host=dbh, port=int(dbp), user=usr, passwd=pwd, db=dbn)
	cur = conn.cursor()
	result = [];
	try: 
		if period.upper() == "H":
			sql = "SELECT filename,vector FROM `people_feature` WHERE `did`=%s AND `timepoint_h`=%s"
		elif period.upper() == "D":
			sql = "SELECT filename,vector FROM `people_feature` WHERE `did`=%s AND `timepoint_d`=%s"
		elif period.upper() == "W":
			sql = "SELECT filename,vector FROM `people_feature` WHERE `did`=%s AND `timepoint_w`=%s"
		cur.execute(sql, (did, timepoint))
		result = cur.fetchall()
	except:
		raise 
	finally:
	    cur.close()
	    conn.close()
	return result

	
def GetFaceVector(imgs):
	vector=[]
	for img in imgs:
		face_locations = fr.face_locations(img)
		face_encoding = fr.face_encodings(img,face_locations)
		if len(face_encoding)>0:
			v=str(list(face_encoding[0]))
			vector.append(v)
	return vector

def SaveVectorsToDB(vectors,did,tp_h,tp_d,tp_w,fn_list,dbhost):
	savePeopleFeature(did, tp_h, tp_d, tp_w, fn_list, vectors,dbhost=dbhost)


def LoadVectorFromDB(did,period,timepoint,dbhost):
	dbresult=fetchPeopleFeature(did, period, timepoint,dbhost=dbhost)
	vectors=[]
	files=[]
	for rec in dbresult:
		files.append(rec[0])
		vectors.append(eval(rec[1]))
	return files,np.array(vectors)
	
####################################################################################	

def face_distance_euc(v1, v2):
    if len(v1) == 0:
        return np.empty((0))
    return np.linalg.norm(v1 - v2)

def cal_distance(vectors):
	cnt=len(vectors)
	sim=np.zeros((cnt,cnt))
	for i in range(cnt):
		for j in range(cnt):
			if j>i:
				dist=1-face_distance_euc(vectors[i],vectors[j])
				sim[i,j]=dist
			else:
				sim[i,j]=sim[j,i]
	return sim	


def CalGroupOuterDist(sim,src,des):
	smin=1
	smax=0
	savg=0
	cnt=0
	for s in src:
		for d in des:
			smin=min(smin,sim[s,d])
			smax=max(smax,sim[s,d])
			savg+=(sim[s,d])**2
			cnt+=1
			
	savg=np.sqrt(savg/cnt)	
	return savg,smin,smax
	
	
def CalGroupInnerDist(sim,src):
	smin=9999
	smax=0
	savg=0
	avgcnt=0
	cnt=len(src)
		
	for i in range(cnt):
		for j in range(cnt):
			if j>i:
				smin=min(smin,sim[src[i],src[j]])
				smax=max(smax,sim[src[i],src[j]])
				savg+=(sim[src[i],src[j]])**2
				avgcnt+=1
	savg=np.sqrt(savg/avgcnt)	
		
	return savg,smin,smax
	

def IsSameGroup(sim,src,des,th,IS_DEBUG=False):
	savg,smin,smax=CalGroupOuterDist(sim,src,des)
	if IS_DEBUG: print (src,des,"distance:",savg,smin)
	if savg>th[0] and smin>th[1]:
		return True
	else:
		return False
			

def CalSimClustering(sim,unique,th,IS_DEBUG=False):
	runcnt=0
	
	while True:
		if len(unique)<=2:break
		span=0
		src=unique[0]
		des=unique[1]
		
		if IsSameGroup(sim,src,des,th)==True:
			unique.remove(des);	unique.remove(src)
			new=sorted(src+des); unique.append(new)
			span+=1
			runcnt=0
			if IS_DEBUG: print (unique)
			continue
		else:
			unique.remove(src);	unique.append(src)
		runcnt+=1	

		if span==0 and runcnt>len(unique) : break  #代表無法繼續組合了

	return unique
	
def CalTotalGroupDist(sim,unique,th):
	in_group_avg=0
	in_group_min=0  #1
	in_group_max=0
	out_group_avg=0
	out_group_min=0 #1
	out_group_max=0
	cnt=0
	for u in unique:
		if len(u)>1 :
			savg,smin,smax=CalGroupInnerDist(sim,u)
			cnt+=1
			in_group_avg+=savg**2
			in_group_min+=smin**2 #in_group_min=min(in_group_min,smin)  #in_group_min+=smin**2
			in_group_max+=smax**2 #in_group_max=max(in_group_max,smax)  #in_group_max+=smax**2

	in_group_avg=np.sqrt(in_group_avg/cnt) if cnt>0 else th[0]
	in_group_min=np.sqrt(in_group_min/cnt) if cnt>0 else th[0]
	in_group_max=np.sqrt(in_group_max/cnt) if cnt>0 else th[0]
	
	cnt=0
	for i in range(len(unique)):
		for j in range(len(unique)):
			if j>i:
				savg,smin,smax=CalGroupOuterDist(sim,unique[i],unique[j])	
				cnt+=1
				out_group_avg+=savg**2
				out_group_min+=smin**2 #out_group_min=min(out_group_min,smin) #out_group_min+=smin**2
				out_group_max+=smax**2 #out_group_max=max(out_group_max,smax) #out_group_max+=smax**2
	if cnt!=0:
		out_group_avg=np.sqrt(out_group_avg/cnt)
		out_group_min=np.sqrt(out_group_min/cnt)
		out_group_max=np.sqrt(out_group_max/cnt)
	return in_group_avg,in_group_min,in_group_max,out_group_avg,out_group_min,out_group_max
	
	
def FastSimClustering(sim,IS_DEBUG=False):
	cnt=len(sim)
	unique=[]
	
	step_tag=[]
	step_unique_list=[]
	step_unique=[]
	step_igavg=[]
	step_igmin=[]
	step_igmax=[]
	step_ogavg=[]
	step_ogmin=[]
	step_ogmax=[]
	for i in range(cnt):unique.append([i])

	print("      sim |Oa/Im |Om/Im | Oma  |  Oa  |  Ia  |  Im  | Omin | Rate |")
	for i in range(66,30,-1):
		step_unique_list.append(list.copy(unique))
		lastlen=len(unique)
		unique=CalSimClustering(sim,unique,th=[i/100,0.0])
		igavg,igmin,igmax,ogavg,ogmin,ogmax=CalTotalGroupDist(sim,unique,th=[i/100,0.0])

		step_tag.append(i)		
		step_unique.append(len(unique))
		step_igavg.append(igavg)
		step_igmin.append(igmin)
		step_igmax.append(igmax)
		step_ogavg.append(ogavg)
		step_ogmin.append(ogmin)
		step_ogmax.append(ogmax)
		if IS_DEBUG:print(unique)
		print("Dist:",format(i/100,"1.2f"),format(ogavg/igmin,"1.4f"),
				format(ogmax/igmin,"1.4f"),format(ogmax,"1.4f"),format(ogavg,"1.4f"),format(igavg,"1.4f"),
				format(igmin,"1.4f"),format(ogmin,"1.4f"),format(lastlen/len(unique),"1.4f"),
				" Unique:",len(unique))
	
	return step_unique_list,step_tag,step_igavg,step_igmin,step_igmax,step_ogavg,step_ogmin,step_ogmax,step_unique


def ShowResultIfWinOS(tag,igavg,igmin,igmax,ogavg,ogmin,ogmax,unique):
	import platform
	if platform.system()!="Windows": return

	import matplotlib.pyplot as plt	
	# 作圖並印出績效
	plt.subplot(311)
	plt.bar(tag,unique)
	plt.grid()
	plt.subplot(312)
	plt.plot(tag,igavg,'r',label='Iavg')
	plt.plot(tag,igmin,'g',label='Imin')
	plt.plot(tag,igmax,'b',label='Imax')
	plt.plot(tag,ogavg,'y',label='Oavg')
	plt.plot(tag,ogmin,'c',label='Omin')
	plt.plot(tag,ogmax,'k',label='Omax')
	plt.grid()
	plt.legend(loc='lower right')
	plt.subplot(313)
	plt.plot(tag,np.array(ogavg)/np.array(igmin),'b',label='Oa/Imi')
	plt.plot(tag,np.array(ogmax)/np.array(igmin),'r',label='Oma/Imi')
	plt.plot(tag,igmin,'g',label='Imin')
	plt.plot(tag,ogavg,'y',label='Oavg')
	plt.grid()
	plt.legend(loc='upper right')
	plt.show()
	print(unique)
	

def GetUnique(uniquelist,ogmax,ogavg,igmin):
	varlist=[]
	ulist=[]
	lastlen=0
	for u in uniquelist:
		rate=lastlen/len(u)
		varlist.append(rate)
		ulist.append(len(u))
		lastlen=len(u)
		
	ucnt=0
	nowmax=1
	lowerbound=False
	for i in range(1,len(uniquelist)):
		#print (ogavg[i]/igmin[i],varlist[i],ulist[i])
		if ogmax[i]/igmin[i]<1 and ogmax[i-1]/igmin[i-1]<1:continue
		if ogavg[i]/igmin[i]>1 and ogavg[i-1]/igmin[i-1]>1:
			lowerbound=True
			continue
		if lowerbound==True: continue
		if varlist[i]<2 and varlist[i]>nowmax:
			nowmax=varlist[i]
			ucnt=ulist[i]
	if len(uniquelist)>0 and ucnt==0: ucnt=len(uniquelist[26])
	return ucnt
	
	
def CalUniquePeople(vectors):
	sim=cal_distance(vectors)	
	unique_list,tag,igavg,igmin,igmax,ogavg,ogmin,ogmax,unique=FastSimClustering(sim)
	ShowResultIfWinOS(tag,igavg,igmin,igmax,ogavg,ogmin,ogmax,unique)
	ucnt=GetUnique(unique_list,ogmax,ogavg,igmin)
	return ucnt,unique_list

	


'''
Hourly unique的作法
	1. OA必須大於0.4
	2. 找Om/Im 往上穿插 >1 -> <1 的

筆記
	1. 從45~55

觀察:
	1. Omax 必須小於 Imin
	2. Oavg 必須小於 Imin
	3. 滿足此二條件後儘量靠左取
	4. Oavg/Imin 由左往右從>1穿過<1時
   
	5. 由左往右找第一次出現 Iavg,Imax 皆大於 Omax,Oavg

'''

'''
	tt=time.time()
	sim=cal_distance(vectors)
	unique_list,tag,igavg,igmin,igmax,ogavg,ogmin,ogmax,unique=FastSimClustering(sim)
	ShowResultIfWinOS(tag,igavg,igmin,igmax,ogavg,ogmin,ogmax,unique)
'''




