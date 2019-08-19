# -*- coding: utf-8 -*-

import json

'''
#########################
Ref:
JSON操作:
http://www.cnblogs.com/coser/archive/2011/12/14/2287739.html
https://python3-cookbook.readthedocs.io/zh_CN/latest/c06/p02_read-write_json_data.html
#########################
'''

def Json_log(obj,key,msg):
	obj[key]=msg
	
	
def Json_print(obj,sort=True):
	print(json.dumps(obj,sort_keys=sort,indent=4,ensure_ascii=False))


def Json_str(obj,sort=True):
	return json.dumps(obj,sort_keys=sort,indent=4)


def Json_load(obj,path):
	with open(path, 'r') as f:
		data = json.load(f)
	return data
