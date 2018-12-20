import pandas as pd
import numpy as np
from pandasql import sqldf

pysqldf = lambda q: sqldf(q, globals())
data_df = pd.DataFrame({'row_id': [1, 2, 3, 4, 5],
                   'total_bill': [16.99, 10.34, 23.68, 23.68, 24.59],
                   'tip': [1.01, 2.0444922e-04, 2.0444922e-05, 3.300011, 4.4516532e-06],
                   'sex': ['Female', 'Male', 'Male', 'Male', 'Female']})

sql="select * from data_df where total_bill>20"
table=pysqldf(sql)
print(table.head())

sql="select row_id,ROUND(tip,4) as tip from data_df"
table=pysqldf(sql)
print(table.head())

table.to_csv('y_pred_submission.csv',index=None)

sql="select gender,gender_1," \
    "CASE when gender_1=0 then 1 else 0 end class_0," \
    "CASE when gender_1=1 then 1 else 0 end class_1 ," \
    "CASE when gender_1=2 then 1 else 0 end class_2," \
    "CASE when gender_1=3 then 1 else 0 end class_3 " \
    "from df "