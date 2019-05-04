import time, datetime


def get_week_day(date):
    week_day_dict = {
        0: '星期一',
        1: '星期二',
        2: '星期三',
        3: '星期四',
        4: '星期五',
        5: '星期六',
        6: '星期天',
    }
    day = date.weekday()
    return week_day_dict[day]


print(get_week_day(datetime.datetime.now()))


timestamp=1541191585
t=time.localtime(timestamp)
h=time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(timestamp))
print(h)
print(str(t.tm_year)+"-"+str(t.tm_mon))
print(type(t))

