import pandas as pd
import numpy as np
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())

from xml.dom.minidom import parse
import xml.dom.minidom
import time
import requests
import http.client
import json
from urllib.parse import quote_plus
#通过minidom解析xml文件
import xml.dom.minidom as xmldom
import os

def geocode(lat_lon):
   path="/v3/geocode/regeo?output=xml&location="+lat_lon+"&key=a7510608de4d9d96dc8fa53474e1c5fa&radius=1000&extensions=base"
   connection = http.client.HTTPConnection('restapi.amap.com', 80)
   connection.request('GET', path)
   rawreply = connection.getresponse().read()
   # reply = json.loads(rawreply.decode('utf-8'))
   print(rawreply)
   return rawreply


parseString=geocode("116.310003,39.991957")
domobj = xmldom.parseString(parseString)

# 得到元素对象
elementobj = domobj.documentElement
subElementObj1 = elementobj.getElementsByTagName("formatted_address")
for i in range(len(subElementObj1)):
    print ("subElementObj1[i]:", type(subElementObj1[i]))
    print (subElementObj1[i].firstChild.data)  #显示标签对之间的数据

def formatted_address(lat_lon):
    parseString = geocode(lat_lon)
    domobj = xmldom.parseString(parseString)
    # 得到元素对象
    elementobj = domobj.documentElement
    subElementObj1 = elementobj.getElementsByTagName("formatted_address")
    address=""
    for i in range(len(subElementObj1)):
        print("subElementObj1[i]:", type(subElementObj1[i]))
        print(subElementObj1[i].firstChild.data)  # 显示标签对之间的数据
        address=subElementObj1[i].firstChild.data
    return address

print(formatted_address("118.71858693659306,32.20462784171105"))