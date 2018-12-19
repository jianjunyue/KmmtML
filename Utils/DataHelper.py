import pandas as pd
import numpy as np

class DataHelper:
    def __init__(self):
        return

    #-----------------------数据类型转换 DataFrame，dict,ndarray,Series,list----------------------------------------------------------
    # DataFrame < --> ndarray
    # Series < --> DataFrame
    # dict --> list
    # ndarray < --> list
    # tuple < --> list
    # Index --> list

    ########### DataFrame 相关转换###########
    # DataFrame < --> ndarray
    def df_to_ndarray(data,columns_name=None):
        """
        DataFrame --> ndarray
        :param columns_name:
        :return:
        """
        ndarray=np.array()
        if columns_name==None:
            ndarray=data.values
        else:
            ndarray = data[columns_name].values
        return ndarray

    def ndarray_to_df(data,columns_name=None):
        """
        DataFrame <-- ndarray
        :param columns_name:
        :return:
        """
        dataframe=pd.DataFrame()
        if columns_name==None:
            dataframe=pd.DataFrame(data)
        else:
            dataframe = data[columns_name].values
        return dataframe

    # DataFrame <--> list
    def df_to_list(data,columns_name=None):
        """
        DataFrame --> list
        :param columns_name:
        :return:
        """
        list=[]
        if columns_name==None:
            list=data.values.tolist()
        else:
            list = data[columns_name].values.tolist()
        return list

    def list_to_df(data,columns_name=None):
        """
        DataFrame <-- list
        :param columns_name:
        :return:
        """
        dataframe=pd.DataFrame()
        if columns_name==None:
            dataframe=pd.DataFrame(data)
        else:
            dataframe = data[columns_name].values
        return dataframe

        # DataFrame <--> dict
    def df_to_list(data, columns_name=None):
        """
        DataFrame --> dict
        :param columns_name:
        :return:
        """
        dict = {}
        if columns_name == None:
            dict = data.to_dict()
        else:
            dict = data[columns_name].to_dict()
        return dict

    def dict_to_df(data, columns_name=None):
        """
        DataFrame <-- dict
        :param columns_name:
        :return:
        """
        dataframe = pd.DataFrame()
        if columns_name == None:
            dataframe = pd.DataFrame(data)
        else:
            dataframe = data[columns_name].values
        return dataframe

    ########### Series 列 数据 ###########
    # Series < --> DataFrame
    def df_to_series(data, columns_name):
        """
        DataFrame --> Series
        :param columns_name:
        :return:
        """
        series = data[columns_name]
        return series

    def series_to_df(series, columns_name):
        """
        DataFrame <-- Series
        :param columns_name:
        :return:
        """
        dataframe =pd.DataFrame({columns_name:series})
        return dataframe

    # Series < --> ndarray
    def ndarray_to_Series(ndarray):
        """
        ndarray --> Series
        :param columns_name:
        :return:
        """
        series   = pd.Series( ndarray  )  # 这里的ndarray是1维的
        return series

    def series_to_ndarray(series):
        """
        ndarray <-- Series
        :param columns_name:
        :return:
        """
        # ndarray  = np.array(series )
        ndarray  = series.values
        return ndarray

        # Series < --> list
    def list_to_Series(list):
        """
        list --> Series
        :param columns_name:
        :return:
        """
        series = pd.Series(list)
        return series

    def series_to_list(series):
        """
        list <-- Series
        :param columns_name:
        :return:
        """
        # list = list(series)
        list = series.tolist()
        return list

        # Series < --> dict
    def dict_to_series(dict):
        """
        dict --> Series
        :param columns_name:
        :return:
        """
        series = pd.Series(dict)
        return series

    def series_to_dict(series):
        """
        list <-- Series
        :param columns_name:
        :return:
        """
        dict = series.to_dict()
        return dict

    ########### 其它 list ###########
       # dict --> list
    def dict_values_to_list(dict):
        """
        dict --> list
        :param columns_name:
        :return:
        """
        list = dict.values() # list of values
        return list
    
    def dict_keys_to_list(dict):
        """
        dict --> list
        :param columns_name:
        :return:
        """
        # list = list(dict)
        list = dict.keys() # list of keys
        return list

    def series_to_dict(series):
        """
        list <-- Series
        :param columns_name:
        :return:
        """
        dict = series.to_dict()
        return dict

        # ndarray --> list
    def ndarray_to_list(ndarray):
        """
        ndarray --> list
        :param columns_name:
        :return:
        """
        list  =  ndarray.tolist()
        return list

    def list_to_ndarray(list):
        """
        ndarray <-- list
        """
        ndarray   = np.array(list )
        return ndarray

        # tuple --> list
    def tuple_to_list(tuple):
        """
        tuple --> list
        :param columns_name:
        :return:
        """
        list  =  list(tuple)
        return list

    def list_to_tuple(list):
        """
        tuple <-- list
        """
        tuple   = tuple(list )
        return tuple

        # Index --> list
    def index_to_list(index):
        """
        Index --> list
        """
        # list  =  dataframe.columns.tolist()
        list = index.tolist()
        return list





