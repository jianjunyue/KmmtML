import pandas as pd
import numpy as np
import datetime
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold


# nrows = 1000000 # 测试建模
nrows = None # 实际建模

def get_data(file_path,nrows=10000,sep=',',):
    pre_path="/Users/jianjun.yue/KmmtML/data/比赛/2019京东用户对品类下店铺的购买预测/jdata/"
    return pd.read_csv(pre_path+file_path,sep=',',nrows=nrows)

jdata_action = get_data('jdata_action.csv',nrows=nrows)
print(jdata_action.head())
