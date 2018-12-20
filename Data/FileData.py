from KmmtML.Utils.FileUtil import FileUtil
import pandas as pd

class FileData:

    def __init__(self):
        return

    def getData(dir_name,file_name,type='.csv'):
        prePath = FileUtil.dataKagglePath(dir_name)
        path = prePath+"/"+file_name + type
        data = pd.read_csv(path)
        return data
