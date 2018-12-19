
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())


class SQLHelper:
    def __init__(self):
        return

    def select(data,sql):
        """
        sql查询
        :param sql:
        :return:
        """
        table = pysqldf(sql)
        return table