import os
import sys
import json
import subprocess
import csv
import time
import math
from optparse import OptionParser


def CliCommand(cmd,cb=None):
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
optParser.add_option("-f","--from", action = "store", type = "int", dest = "fromtime")
(options, args) = optParser.parse_args()
print( options)

sn=options.sn
fromtime=options.fromtime


with open('output.csv', 'a', newline='') as csvfile:
    writer=csv.writer(csvfile)
    writer.writerow(['time','resolution','got_face_cost','got_face','got_id_cost','got_id'])


for dirPath, dirNames, fileNames in os.walk("/home/tommy/vca-facerecog/src/cvpy/queue_folder/home"):

    print ("%s"%dirPath)
    for f in sorted(fileNames):
        filename, file_extension = os.path.splitext(f)
        if file_extension == ".flv":
            fsplit = filename.split("-")
            get_face_start_time=time.time()
            if fsplit[0] != sn:
                continue
            print (fsplit[0],fsplit[1])
            ev_time=int(fsplit[1])
            if ev_time<fromtime:
               continue

            cli = CliCommand("python ./api_get_video_face_v3.py /home/tommy/vca-facerecog/src/cvpy/queue_folder/home/%s-%s.flv /home/tommy/vca-facerecog/src/cvpy/queue_folder/home/%s-%s.jpg 6000"%(fsplit[0],fsplit[1],fsplit[0],fsplit[1]))
            get_face_end_time=time.time()

            got_face=0
            with open('output.csv', 'a', newline='') as csvfile:
                writer=csv.writer(csvfile)
                getface = json.loads(cli)
                if getface['face_detected']:
                    print("%s got face"%(f))
                else:
                    writer.writerow([ev_time,'1280x720',round(get_face_end_time-get_face_start_time,1),got_face,0,'0'])
                    continue

                cli= CliCommand("python /home/tommy/vca-facerecog/src/cvpy/api_face_recogn.py /home/tommy/vca-facerecog/src/cvpy/face_cache/home.json /home/tommy/vca-facerecog/src/cvpy/queue_folder/home/%s-%s.jpg /home/tommy/vca-facerecog/src/cvpy/queue_folder/home/%s-%s-hit.jpg"%(fsplit[0],fsplit[1],fsplit[0],fsplit[1]))

                get_id_end_time=time.time()
                getid = json.loads(cli)

                print(getid)
                
                if getface['face_detected'] is True:
                    got_face=1

                if len(getid['detected'])>0:
                    print("%s got %s"%(f,getid['detected'][0]))
                    writer.writerow([ev_time,'1280x720',round(get_face_end_time-get_face_start_time,1),got_face,round(get_id_end_time-get_face_end_time,1),getid['detected'][0]])
                else:
                    print("got anybody")
                    writer.writerow([ev_time,'1280x720',round(get_face_end_time-get_face_start_time,1),got_face,round(get_id_end_time-get_face_end_time,1),'0'])
         
         


