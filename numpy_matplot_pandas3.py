import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#تمرین تحلیل فروش روزانه

sales =np.random.randint(50,200,size=30)
df=pd.DataFrame({"day":np.arange(1,31),"sales":sales})

df["cumulativesales"]=np.cumsum(sales)
print(df)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.plot(df["day"],df["sales"],marker="o")
plt.title("Sales vs Day")
plt.xlabel("Day")
plt.ylabel("Sales")

plt.subplot(1,2,2)
plt.plot(df["day"],df["cumulativesales"],marker="o",color="r")
plt.title("Cumulative Sales vs Day")
plt.xlabel("Day")
plt.ylabel("Cumulative Sales")

plt.tight_layout()
plt.show()