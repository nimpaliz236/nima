import pandas as pd

df=pd.read_csv('E:\\django\\Pandas2.csv')
m=df.info(memory_usage='deep')
n=df.convert_dtypes()
#print(m)
#print(n.dtypes)

df1=pd.read_csv('E:\\django\\Pandas2.csv',dtype_backend="pyarrow")
m1=df1.info(memory_usage='deep')
#print(m1)

mask_base = (df['Age'] > 30) & (df['City'].notna()) & (df['Income'].between(2000, 3000))
q1 = df.loc[mask_base, :]
q2 = df.loc[mask_base, ['Name','Income']]
q3 = df.loc[mask_base, ['Name','Age']]
q4 = df.loc[mask_base, ['Name','City']]
q5 = df.loc[mask_base, ['Name','Score']]
q6 = df.loc[mask_base, ['Age','Income','Score']]
q7 = df.loc[mask_base, :].head(5)
q8 = df.loc[mask_base, ['Name','Age','Income']]
q9 = df.loc[mask_base, ['Name','City','Score']]
q10 = df.loc[mask_base, :].tail(10)
grouped = df.groupby('City')
print(grouped)






