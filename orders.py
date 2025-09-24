import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os


#خواندن فایل با دو روش مختلف
orders=pd.read_csv('orders_v2.csv', parse_dates=["ts"])
orders.to_parquet('orders_v2.parquet', engine='pyarrow', compression='snappy')

#مشاهده سایز و زمان اجرا با روش اول
start = time.time()
orders_csv = pd.read_csv('orders_v2.csv')
end = time.time()
Totaltime_csv=end-start
size_csv=os.path.getsize('orders_v2.csv')/(1024*1024)
print(size_csv, 'MB')
print(Totaltime_csv, 'seconds')
#print("--------------------------------------\n")

#محاسبه زمان و اجرا با روش دوم
start = time.time()
orders_parquet = orders.to_parquet('orders_v2.parquet',engine='pyarrow')
end = time.time()
Totaltime_parquet=end-start
size_parquet=os.path.getsize('orders_v2.parquet')/(1024*1024)
print(size_parquet, 'MB')
print(Totaltime_parquet, 'seconds')

#اضافه کردن ستون ماه از روی ستونts
orders_csv["ts"] = pd.to_datetime(orders_csv["ts"], errors="coerce")
orders_csv["month"]=orders_csv["ts"].dt.month
print(orders_csv.head(5))

#ایجاد یک پیووت تیبل سنگین
pivottable = pd.pivot_table(
    orders_csv,
    values='amount',
    index=['province'],
    columns=['channel','month'],
    aggfunc=['sum','mean','count']
)
print(pivottable.head(5))

#محاسبه سه کانال برتر و نمایش روی نمودار

# 1️⃣ محاسبه جمع فروش هر کانال
Channel_sum = orders.groupby("channel")["amount"].sum()
Top3_channel = Channel_sum.sort_values(ascending=False).head(3)

# 2️⃣ فیلتر داده‌ها برای سه کانال برتر
top_channels_data = orders_csv[orders_csv["channel"].isin(Top3_channel.index)]

# 3️⃣ جمع فروش هر استان برای هر کانال
Province_Channel_Sum = top_channels_data.groupby(["province","channel"])["amount"].sum().unstack()
print(Province_Channel_Sum.head(5))

# 4️⃣ رسم نمودار
Province_Channel_Sum.plot(kind="bar", figsize=(12,6))
plt.title("Total Sales for Top 3 Channels in Each Province")
plt.xlabel("Province")
plt.ylabel("Total Amount")
plt.xticks(rotation=45)
plt.legend(title="Channel")
plt.tight_layout()
plt.show()



