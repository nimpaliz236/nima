
import pandas as pd

df = pd.read_csv("income_data_50_en.csv")
grouped = df.groupby("City").size()
g=df.groupby("Age").agg({"Age": "mean"})
print(g)
print("\n -------------------------------------------\n")
h=df.groupby("Age").agg(
    avg_rev=("Revenue_Oct", "mean"),
    users=("Name", "nunique")
)
print(h)
print("\n -------------------------------------------\n")
m=df.sort_values("Age", kind="quicksort")
print(m)
print("\n -------------------------------------------\n")
i=pd.pivot_table(df,values="Age", index=["Name","City","Revenue_Jan"], columns=["Revenue_Mar"])
print(i)
print("\n -------------------------------------------\n")

g1=df.groupby("City").agg({"Age": "mean"})
print(g1)
print("\n --------------------------------------------\n")
g2=df.groupby("City")["Revenue_Feb"].transform("mean")
print(g2)
print("\n ---------------------------------------------\n")
months = [
    "Revenue_Jan", "Revenue_Feb", "Revenue_Mar", "Revenue_Apr",
    "Revenue_May", "Revenue_Jun", "Revenue_Jul", "Revenue_Aug",
    "Revenue_Sep", "Revenue_Oct", "Revenue_Nov", "Revenue_Dec"
]

for m in months:
    df[m + "_z"] = df.groupby("City")[m].transform(
        lambda x: (x - x.mean()) / x.std()
    )
print(df.head())
print("\n --------------------------------------------\n")
z_cols = [col for col in df.columns if col.endswith("_z")]
outliers = df[(df[z_cols] > 2).any(axis=1)]

outliers_high = outliers.copy()
for col in z_cols:
    outliers_high.loc[outliers_high[col] <= 2, col] = None

print(outliers_high[["Name", "City"] + z_cols])
print("\n --------------------------------------------\n")


value_cols = [
    "Revenue_Jan", "Revenue_Feb", "Revenue_Mar", "Revenue_Apr",
    "Revenue_May", "Revenue_Jun", "Revenue_Jul", "Revenue_Aug",
    "Revenue_Sep", "Revenue_Oct", "Revenue_Nov", "Revenue_Dec"
]

df_long = df.melt(
    id_vars=["Age","Name","City"],
    value_vars=value_cols,
    var_name="Month",
    value_name="Revenue"
)
print(df_long)
print("\n -------------------------------------------------\n")


df_wide = df_long.pivot_table(
    index=["Name", "Age", "City"],
    columns="Month",
    values="Revenue",
    fill_value=0
).reset_index()

#print(df_wide)



