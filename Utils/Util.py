#!/usr/bin/env python
#coding:utf8

from KmmtML.Utils.DataHelper import DataHelper
import pandas as pd

class Util:
    def __init__(self):
        return

    #DataFrame
    # train["Survived"]=train["Survived"].astype('str')
    def CSV2Libsvm(data):
        libsvm = []
        for row_array in data.values:
            tempsvm = ""
            for i in range(len(row_array)):
                if (i == 0):
                    tempsvm = str(row_array[i]) + " "
                else:
                    if (int(row_array[i]) != 0):
                        tempsvm += str(i) + ":" + str(row_array[i]) + " "
            libsvm.append(tempsvm.strip())

        dataframe = pd.DataFrame(libsvm,index=None, columns=None)
        return dataframe


