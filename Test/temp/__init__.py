import pandas as pd
import numpy as np
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())

import http.client
import xml.dom.minidom as xmldom
import time

# 输入毫秒级的时间，转出正常格式的时间
def timeStamp(timeNum):
    try:
        timeStamp = float(timeNum/1000)
        timeArray = time.localtime(timeStamp)
        otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
        return otherStyleTime
    except:
        print(timeNum)
        return "error"

search_log = pd.read_csv("search.log0412_3.tsv", sep='\t')
print(search_log.head())
search_log["city_id"]=search_log["city_id"].fillna(0)
search_log["city_id"]=search_log["city_id"].astype(int)
# search_log["time"]=search_log["time"].astype(int)
search_log["time"]=search_log["time"].apply(timeStamp)

search_log["longitude"]=search_log["longitude"].astype(str)
search_log["latitude"]=search_log["latitude"].astype(str)
search_log["lat_lon"]=search_log["longitude"]+","+search_log["latitude"]
print(search_log.head())

city_id_name = pd.read_excel("city_id_name.xlsx")
print(city_id_name.head())
sql="select keyword,search_log.city_id,city_id_name.city_name,latitude,longitude,time,lat_lon from search_log left join city_id_name on search_log.city_id=city_id_name.city_id  "
table = pysqldf(sql)
print(table.head())

def geocode(lat_lon):
   path="/v3/geocode/regeo?output=xml&location="+lat_lon+"&key=a7510608de4d9d96dc8fa53474e1c5fa&radius=1000&extensions=base"
   connection = http.client.HTTPConnection('restapi.amap.com', 80)
   connection.request('GET', path)
   rawreply = connection.getresponse().read()
   return rawreply

def formatted_address(lat_lon):
    try:
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
    except:
        print(lat_lon)
        return "error"


table["address"]=table["lat_lon"].apply(formatted_address)

print(table.head())

data=table

sql="select keyword,city_id,city_name,address,time from data"
table_temp = pysqldf(sql)
print(table_temp.head())

table_temp.to_csv("/Users/jianjun.yue/KmmtML/KmmtML/Test/temp/0412_3_search_log_address.csv",index=False)
