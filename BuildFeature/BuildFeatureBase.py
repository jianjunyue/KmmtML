import pandas as pd
from KmmtML.BuildFeature.Util import Util
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import Normalizer

#参数多列数据则为DataFrame，单列数据则为Series
#返回多列数据则为DataFrame，单列数据则为Series ？
class BuildFeatureBase:
    def __init__(self):
        return

    def mean(data,name):
        """
        :param data:
        :param name:
        :return: 均值 离散型一般用众数，连续型用中位数或者均值。
        """
        return data[name].mean()

    def median(data,name):
        """
        :param data:
        :param name:
        :return: 中位数 离散型一般用众数，连续型用中位数或者均值。
        """
        return data[name].median()

    def mode(data,name):
        """
        :param data:
        :param name:
        :return: 众数 离散型一般用众数，连续型用中位数或者均值。
        """
        return data[name].mode()[0]

    def modes(data,name):
        """
        :param data:
        :param name:
        :return: 众数
        """
        modes=data[name].mode()
        dict_modes = {'mode_index': modes.index, 'mode': modes.values}
        df_modes = pd.DataFrame(dict_modes,index=None)
        return df_modes

    #-------------------------------------特征编码-------------------------------------------#
    def one_hot_encoder(data,name):
        """
        独热编码
        对于无序特征，用独热编码
        >>> df_one_hot_encoder=BuildFeatureBase.one_hot_encoder(df,"gender")
        >>> df2 = df.join(df_one_hot_encoder)
        :param name: 待独热编码特征
        :return: 返回独热编码DF
        """
        df_one_hot_encoder = pd.get_dummies(data[name])
        occ_cols = [name+'_' + columns_name for columns_name in df_one_hot_encoder.columns]
        df_one_hot_encoder.rename(columns=Util.rename_columns(name, df_one_hot_encoder.columns), inplace=True)
        return df_one_hot_encoder

    def labelencoder_mapping(data,name):
        """
        数值编码
        对于有序特征，用数值编码（可区分特征大小关系）；对于无序特征，用独热编码
        >>> labelencoder_mapping=BuildFeatureBase.labelencoder_mapping(df,"gender")
        {'data': 0, 'group': 1, 'mall': 2, 'room': 3}
        :param name:
        :return:
        """
        le = preprocessing.LabelEncoder()
        le.fit_transform(data[name])
        mapping = {lable: index for index, lable in enumerate(le.classes_)}
        return mapping

    def labelencoder_df(data,column_name,mapping=None):
        """
        数值编码
        对于有序特征，用数值编码（可区分特征大小关系）；对于无序特征，用独热编码
        >>> labelencoder_df=BuildFeatureBase.labelencoder_df(df,"gender",labelencoder_mapping)

        >>> labelencoder_df=BuildFeatureBase.labelencoder_df(data,"target")
        >>> data["target"]=labelencoder_df["target"]
        {'data': 0, 'group': 1, 'mall': 2, 'room': 3}
        :param name:
        :return:
        """
        df = pd.DataFrame(index=None)
        if mapping==None:
            mapping = BuildFeatureBase.labelencoder_mapping(data, column_name)
        df[column_name]=data[column_name].map(mapping)
        return df

    #-------------------------------------特征映射 --> 归一化，正太化，标准化，二值化 ------------------------------------#

    def minMaxScaler(fit_data,transform_data=None):
        """
        以列为单位处理数据，列的转化值=列的实际值/(每列的最大值-每列的最少值). 缺点:不免疫outlier
        >>> listdata=np.array([[1, 2],[2, 3],[3, 5]])
        >>> listdata=np.array([[2],[3],[5]])
        >>> temp=BuildFeatureBase.minMaxScaler(np.array([[1],[3],[8]]),np.array([[2],[3],[30]]))
        :return: 返回归一化数据
        """
        if transform_data==None:
            transform_data=fit_data
        data_scaler = MinMaxScaler(feature_range=(0, 1))
        # 数据转换模型训练
        data_scaler.fit(fit_data)
        y = data_scaler.transform(transform_data)
        return y

    def standardScaler(fit_data,transform_data=None):
        """
        正太化 数据
        以列为单位处理数据，处理后每一列均值(和)=0，方差=1
        >>> listdata=np.array([[4],[6],[10]])
        >>> listdata1=np.array([[2],[3],[5]])
        >>> temp=BuildFeatureBase.standardScaler(listdata,listdata1)
        :param transform_data:
        :return:
        """
        # if transform_data==None:
        #     transform_data=fit_data
        scaler = StandardScaler().fit(fit_data)
        data = scaler.transform(transform_data)
        return data

    def normalizer(fit_data,transform_data=None):
        """
        标准化 数据
        以行为单位处理数据，调整后，每一行数据矢量距离(每一行数据向量长度)为1
        >>> listdata=np.array([[1, 2],[2, 3],[3, 5]])
        >>> listdata1=np.array([[1, 2],[2, 3],[4, 5]])
        >>> temp=BuildFeatureBase.normalizer(listdata,listdata1)
        :param transform_data:
        :return:
        """
        # if transform_data==None:
        #     transform_data=fit_data
        scaler = Normalizer().fit(fit_data)
        # 数据转换
        data = scaler.transform(fit_data)
        return data


    #-----离散化 --> 连续数据离散-等宽离散，等频离散，聚类离散。  把数据按不同区间划分（等宽划分或等频划分），聚类编码/按层次进行编码 ----------#
    def bin_point(series,level):
        lam=lambda x: int(x/level)
        temp= series.apply(lam)
        return temp

    def fe_quantile(series,quantile=[0.25, 0.5, 0.75]):
        """
        分位数分箱计数
        quantile=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        :return:
        0.25    22.0
        0.50    28.0
        0.75    35.0
        """
        temp=series.quantile(quantile)
        return temp

    def bin_quantile(series,quantile=[0.25, 0.5, 0.75]):
        """
        把值转换为指定分位数分箱计数
        quantile=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        :param quantile:
        :return:
        """
        temp = series.quantile(quantile)
        def funx(x):
            for index,value in zip(temp.index,temp.values):
                if x<value:
                    return index
        lambda1=lambda x:funx(x)
        temp= series.apply(lambda1)
        return temp

