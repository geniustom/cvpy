# -*- coding: utf-8 -*-
import platform


TRACE_FACE_CNT        = 0    #至少一個路徑要有幾個疑似人臉的路徑才判定是人
DETECT_RATE           = 2    #在沒偵測到人臉之前，每次間隔幾個FRAME (目前最好=1)
DETECTED_COOLDOWN     = 2    #在偵測到人臉之後，下一次次間隔幾個FRAME  (目前最好=1)
DETECT_SCORE          = 0.2  #分數至少要在此門檻之上才有資格判定為人臉 (目前最好=0.2)
MIN_FACE_SIZE         = 40   #best 20
MAX_FACE_SIZE         = 60   #best 30
SYSTEM_IMG_WIDTH      = 200
SYSTEM_IMG_HEIGHT     = 300

####
TRACK_FACE_TIMEOUT    = 15     #追蹤人體的眶若多少個frame沒有臉就停止track
#TRACE_FACE_MOVE_SPEED = 5   #差異多少個pixel? 5->10
TRACK_BODY_HEIGHT     = 2.5
TRACK_ALGORITHM       = "DLIB"
#     X     X   V       X      V   V
# BOOSTING TLD MIL MEDIANFLOW KCF DLIB

IS_DEBUG              =(platform.system()=="Windows")


#以下為測試用db conn config
if platform.system()=="Windows":
	DB_INFO            = "cv-user:JJy-myD-4Sz-hk4@localhost:3306:cv"
else:
	DB_INFO            = "cv-user:JJy-myD-4Sz-hk4@cvgo-rds.astra.ap-northeast-1.local:3306:cv"
	#DB_INFO            = "root:a@cvgo-rds.astra.ap-northeast-1.local:3306:cv"

#以下部分為效能計算評估變數
####################################
TIME_MOTION_DET       = 0
TIME_HUMAN_DET        = 0
TIME_FACE_DET         = 0
TIME_FACE_VERIFY      = 0
TIME_BODY_TRACK       = 0
TIME_GET_FRAME        = 0
TIME_DRAW_INFO        = 0
TIME_FILTER_FRAME     = 0
TIME_UI_SHOWN         = 0
TIME_BEST_FACE        = 0
TIME_FACE_ALIGN       = 0