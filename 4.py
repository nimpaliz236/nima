import csv
import numpy as np

data = [
    ["name", "age", "score"],
    ["Nima", 23, 88],
    ["Ali", 25, 95],
    ["Hossein", 22, 78],
    ["mmd",35,60],
    ["reza",23,68]
]

with open("c.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(data)


with open("c.csv", mode="r") as file:
    csv = file.read()
    print(csv)

data = np.genfromtxt("c.csv", delimiter=",", dtype=None, names=True, encoding="utf-8")
print(data)

scores = data['score']
average_score = np.mean(scores)
print("avg", average_score)

max_age = np.max(data["age"])
print("Max age:", max_age)


