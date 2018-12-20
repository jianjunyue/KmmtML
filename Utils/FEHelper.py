#!/usr/bin/env python
#coding:utf8

#特征工程帮助类
class FEHelper:
    def __init__(self):
        return

    def rename_columns(pre_name,link_char, columns_name):
        name_dict = {}
        for name in columns_name:
            name_dict[name] =pre_name+link_char+ str(name)
        return name_dict

    def data_info(data,keyName):
        print("-----------------------------------data:"+keyName+"-------------------------------------------")
        print(data[keyName].info())# ?
        print(data[keyName].describe())
        print(data[keyName].unique())
        print(data[keyName].value_counts())
        print(data[keyName].isnull().sum())
        print()

    def data_info(data):
        print(data.info())
        print(data.describe())


