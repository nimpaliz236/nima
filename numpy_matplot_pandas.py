import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#تمرین آمار فروش روزانه

sales=np.random.randint(1000,5000,size=12)

months=["jan","feb","mar","apr","may","jun",
        "jul","aug","sep","oct","nov","dec"]

df = pd.DataFrame({
    "month": months,
    "sales": sales
})
print(df)

plt.plot(df["month"], df["sales"],marker="o")
plt.title("monthly sales")
plt.xlabel("month")
plt.ylabel("sales")
plt.grid(True)
plt.show()
