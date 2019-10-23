import os
import sys
import json
import subprocess
import csv
import time
import math
from optparse import OptionParser
from shutil import copyfile



def CliCommand(cmd,cb=None):

    print(cmd)
    try:
        # stdout = subprocess.PIPE lets you redirect the output
        res = subprocess.Popen(['/bin/bash', '-c', cmd],stdout=subprocess.PIPE)
    except OSError:
        print( "error: popen")
        exit(1) # if the subprocess call failed, there's not much point in continuing

    ret = bytearray()

    while 1:
        line = res.stdout.readline()
        if not line: break
#        sys.stdout.write(line.decode('utf-8'))
        if cb:
            cb(line)

        ret+=line
    
    res.wait() # wait for process to finish; this also sets the returncode variable inside 'res'

    if res.returncode != 0:
        print("  os.wait:exit status != 0\n")
        sys.exit(1)
    else:
        print ("os.wait:({},{})".format(res.pid, res.returncode))

    
    return ret.decode('utf-8')




MSG_USAGE = "migrate"
optParser = OptionParser(MSG_USAGE)
optParser.add_option("-s","--sn", action = "store", type = "string", dest = "sn")
optParser.add_option("-f","--from", action = "store", type = "int", dest = "fromtime",default=0)
optParser.add_option("-t","--to", action = "store", type = "int", dest = "totime",default=0)
optParser.add_option("-c","--comment", action = "store", type = "string", dest = "comment")
optParser.add_option("-p","--path", action = "store", type = "string", dest = "path")
optParser.add_option("-i","--state", action = "store", type = "string", dest = "state")  # getface,recogface
(options, args) = optParser.parse_args()
print( options)


fromtime=0
state=options.state
sn=options.sn
fromtime=options.fromtime
totime=options.totime
comment=options.comment
path=options.path
csvpath=path+comment

if not os.path.exists(path+"negtive"):
    os.mkdir(path+"negtive")
if not os.path.exists(path+"positive"):
    os.mkdir(path+"positive")

with open(csvpath, 'a', newline='') as csvfile:
    writer=csv.writer(csvfile,quotechar = "'")
    writer.writerow(['url','time','face_score','got_face_cost','got_face','got_id_cost','got_id'])
    
for dirPath, dirNames, fileNames in os.walk(path):

    print ("%s"%dirPath)
    for f in sorted(fileNames):
        filename, file_extension = os.path.splitext(f)
        fsplit = filename.split("-")
        get_face_start_time=time.time()
        if fsplit[0] != sn:
            continue
        ev_time=int(fsplit[1])

        if (fromtime > 0 and ev_time<fromtime) or (totime>0 and ev_time>totime):
           continue

        named_tuple = time.localtime(ev_time+28800)
        time_string = time.strftime("%Y_%m_%d_%H:%M:%S", named_tuple)

        if state != "recogface":
            if file_extension == ".flv":
                
                dstjpg="%s-%s-%s.jpg"%(fsplit[0],fsplit[1],time_string)
                dstjpgPath="%s/%s"%(path,dstjpg)
                srcflv="%s/%s"%(path,f)
                cli = CliCommand("python ./api_get_video_face_v4.py %s %s 6000"%(srcflv,dstjpgPath))
                get_face_end_time=time.time()
                print(cli)
    
                facejpgUrl="=IMAGE(\"http://35.201.225.82:8080/image/%s\")"%(dstjpg)
    
                got_face=0
                getface = json.loads(cli)
                if getface['face_detected'] is True:
                    print("%s got face"%(f))
                else:
                    with open(csvpath, 'a', newline='') as csvfile:
                        writer=csv.writer(csvfile,quotechar = "'")
                        writer.writerow([facejpgUrl,ev_time,getface['face_score'],round(get_face_end_time-get_face_start_time,1),got_face,0,'0'])
                    continue
    
                hitjpgPath="%s/%s-%s-%s.jpg"%(path,fsplit[0],fsplit[1],time_string)
                cli= CliCommand("python /home/tommy/vca-facerecog/src/cvpy/api_face_recogn_v2.py /home/tommy/vca-facerecog/src/cvpy/face_cache/home.json %s %s"%(dstjpgPath,hitjpgPath))
                print(cli)
                get_id_end_time=time.time()
                getid = json.loads(cli)
    
                print(getid)
                
                if getface['face_detected'] is True:
                    got_face=1
    
                if len(getid['detected'])>0:
                    idjpg="%s-%s-%s-%s.jpg"%(fsplit[0],fsplit[1],time_string,getid['detected'][0])
                    idjpgPath="%s/%s"%(path,idjpg)
                    copyfile(hitjpgPath,idjpgPath)
                    hitfacejpgUrl="=IMAGE(\"http://35.201.225.82:8080/image/%s\")"%(idjpg)
                    print("%s got %s"%(f,getid['detected'][0]))
                    with open(csvpath, 'a', newline='') as csvfile:
                        writer=csv.writer(csvfile,quotechar = "'")
                        writer.writerow([hitfacejpgUrl,ev_time,getface['face_score'],round(get_face_end_time-get_face_start_time,1),got_face,round(get_id_end_time-get_face_end_time,1),getid['detected'][0]])
    
                    folder1="%s/positive/%s"%(path,idjpg)
                    copyfile(hitjpgPath,folder1)
                else:
                    hitfacejpg="=IMAGE(\"http://35.201.225.82:8080/image/%s-%s-%s-?.jpg\")"%(fsplit[0],fsplit[1],time_string)
                    print("got anybody")
                    with open(csvpath, 'a', newline='') as csvfile:
                        writer=csv.writer(csvfile,quotechar = "'")
                        writer.writerow([hitfacejpg,ev_time,getface['face_score'],round(get_face_end_time-get_face_start_time,1),got_face,round(get_id_end_time-get_face_end_time,1),'0'])
         
                    folder2="%s/negtive/%s-%s-%s-?.jpg"%(path,fsplit[0],fsplit[1],time_string)
                    copyfile(hitjpgPath,folder2)
        else:
            if file_extension == ".jpg":
                if len(fsplit) != 3:  #eg:SWB000mgLbUm-1568293362-2019_09_12_21:02:42.jpg
                    continue

            hitjpg="%s/%s-%s-%s-hit.jpg"%(path,fsplit[0],fsplit[1],fsplit[2])
            srcjpg="%s/%s"%(path,f)
            cli= CliCommand("python /home/tommy/vca-facerecog/src/cvpy/api_face_recogn_v2.py /home/tommy/vca-facerecog/src/cvpy/face_cache/home.json %s %s"%(srcjpg,hitjpg))
            print(cli)
            get_id_end_time=time.time()
            getid = json.loads(cli)

            print(getid)
            
            if len(getid['detected'])>0:
                hitfacejpg="%s-%s-%s-%s.jpg"%(fsplit[0],fsplit[1],fsplit[2],getid['detected'][0])
                hitfacejpgPath="%s/%s.jpg"%(path,hitfacejpg)
                os.rename(hitjpg,hitfacejpg)
                hitfacejpgURL="=IMAGE(\"http://35.201.225.82:8080/image/%s\")"%(hitfacejpg)
                print("%s got %s"%(f,getid['detected'][0]))
                with open(csvpath, 'a', newline='') as csvfile:
                    writer=csv.writer(csvfile,quotechar = "'")
                    writer.writerow([hitfacejpgURL,ev_time,getface['face_score'],round(get_face_end_time-get_face_start_time,1),1,round(get_id_end_time-get_face_end_time,1),getid['detected'][0]])

                positive_folder="%s/positive/%s"%(path,hitfacejpg)
                copyfile(facejpg,positive_folder)

