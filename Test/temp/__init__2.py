import pandas as pd
import numpy as np
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())

import http.client
import xml.dom.minidom as xmldom
import time

# search_log = pd.read_csv("/Users/jianjun.yue/KmmtML/KmmtML/Test/temp/search_log_address.csv")
# sql="select keyword,city_id,city_name,address,time from search_log"
# table = pysqldf(sql)
# print(table.head())

# table.to_csv("/Users/jianjun.yue/KmmtML/KmmtML/Test/temp/search_log_address1.csv",index=False)


search_log = pd.read_csv("/Users/jianjun.yue/KmmtML/KmmtML/Test/temp/search_log_address.csv")