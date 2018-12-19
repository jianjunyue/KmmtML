from sklearn.preprocessing import MinMaxScaler

class FEMinMaxScaler:
    def __init__(self):
        return

    def fit_transform(basedata,min=0,max=1):
        """
        >>> istdata=np.array([[1, 2],[2, 3],[3, 5]])
        >>> listdata=np.array([[2],[3],[5]])
        待预测数据归一化到0-1返回
        :param min:
        :param max:
        :return:
        """
        data_scaler = MinMaxScaler(feature_range=(min, max))
        # 数据转换
        data_rescaledX = data_scaler.fit_transform(basedata)
        return data_rescaledX

    def fit_transform(basedata):
        """
        >>> istdata=np.array([[1, 2],[2, 3],[3, 5]])
        >>> listdata=np.array([[2],[3],[5]])
        待预测数据归一化到0-1返回
        :param min:
        :param max:
        :return:
        """
        data_scaler = MinMaxScaler(feature_range=(0, 1))
        # 数据转换
        data_rescaledX = data_scaler.fit_transform(basedata)
        return data_rescaledX

    def transform(model,data):
        """
        :param model 已经训练好的归一化模型
        :param data: 待预测的数据
        :return: 待预测数据归一化结果
        """
        # 数据转换
        data_rescaledX = model.transform(data)
        return data_rescaledX
