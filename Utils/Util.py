#!/usr/bin/env python
#coding:utf8

class Util:
    def __init__(self):
        return

    def rename_columns(pre_name,link_char, columns_name):
        name_dict = {}
        for name in columns_name:
            name_dict[name] =pre_name+link_char+ str(name)
        return name_dict