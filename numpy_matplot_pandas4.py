import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from numpy_matplot_pandas import months

store_A=np.random.randint(1000,5000,size=12)
store_B=np.random.randint(1000,5000,size=12)
store_C=np.random.randint(1000,5000,size=12)

months1=["jan","feb","mar","apr","may","jun",
        "jul","aug","sep","oct","nov","dec"]

df=pd.DataFrame({"month":months1,
                 "Store1":store_A,
                 "Store2":store_B,
                 "Store3":store_C
                 })

df["total"]=df[["Store_A","Store_B","Store_C"]].sum(axis=1)
print(df)

avg_A=np.mean(store_A)
avg_B=np.mean(store_B)
avg_C=np.mean(store_C)
print(avg_A,avg_B,avg_C)

max=df[["Store1","Store2","Store3"]].sum().idxmax()
print(max)
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(df["month"],df["Store1"],marker="o",label="Store1")
plt.plot(df["month"],df["Store2"],marker="o",label="Store2")
plt.plot(df["month"],df["Store3"],marker="o",label="Store3")
plt.title("Monthly Store")
plt.xlabel("Month")
plt.ylabel("Store")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.bar(df["month"],df["total"],marker="o",color="r",label="Total")
plt.title("Total Store")
plt.xlabel("Month")
plt.ylabel("Store")
plt.xlabel("Total")
plt.ylabel("Store")

plt.tight_layout()
plt.show()

