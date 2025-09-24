import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#تمرین تحلیل نمرات دانشجویان

data={
    "name": ["Nima", "Fatima", "Sadegh", "Milad", "Nargrs"],
    "math":[18,18,16,16,13],
    "english":[17,19,16,15,12],
    "datamining":[18,17,16,15,14]
}
df = pd.DataFrame(data)

df["Avarege"]=np.mean(df[["math","english","datamining"]],axis=1)
df["Max"]=np.max(df[["math","english","datamining"]],axis=1)
print(df)

plt.bar(df["name"],df["Avarege"],color="orange")
plt.title("Average Scores")
plt.xlabel("Students")
plt.ylabel("Scores")
plt.show()