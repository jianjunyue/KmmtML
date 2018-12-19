from Utils.FileUtil import FileUtil
import pandas as pd

class TitanicData:
    def __init__(self):
        return

    def data_original():
        """
        Titanic 原始数据
        --------

        >>> def f(a,b,c=2) :
        >>>     return a*b*c
        >>> print(_make_signature(f))
        a,b,c=2
        """
        titanicPath = FileUtil.dataKagglePath("titanic/original/")
        path = titanicPath + "train.csv"
        data = pd.read_csv(path)
        path_test = titanicPath + "test.csv"
        data_test = pd.read_csv(path_test)
        return data,data_test

    def data(self,name):
        titanicPath = FileUtil.dataKagglePath("titanic/")
        path = titanicPath + "train.csv"
        data = pd.read_csv(path)
        path_test = titanicPath + "test.csv"
        data_test = pd.read_csv(path_test)
        return data,data_test

