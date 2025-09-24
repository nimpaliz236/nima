import pandas as pd
import numpy as np
import csv
data = [
    ["name", "age", "score"],
    ["Nima", 23, 88],
    ["Ali", 25, 95],
    ["Hossein", 22, 78],
    ["mmd",35,60],
    ["reza",23,68]
]

with open("a.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(data)


with open("a.csv", mode="r") as file:
    csv = file.read()
    print(csv)
df=pd.read_csv("a.csv")

#df = pd.DataFrame(df)
#print(df.head(2))
#print("\n ---------------------\n")
#print(df.tail(2))
#print("\n ---------------------\n")
#print(df.info())
#print("\n ---------------------\n")
#print(df.describe())
#print("\n ---------------------\n")
#print(df[["name","age"]])
#print("\n ---------------------\n")
#print(df["score"])
#print("\n ---------------------\n")
#print(df)
#print(df.iloc[2,1])
#df.to_csv("a1.csv", index=False)
#print(df[df["age"] > 22])
#print(df[(df["score"] > 80) & (df["age"] < 25)])
df["passed"] = df["score"] > 60
print(df)
df["score"] = df["score"]+5
print(df)



