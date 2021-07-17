import numpy as py
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('./mtcars.csv')
d=pd.crosstab(index=data['cyl'],columns="count",dropna=True)
print(d)

print(data.info())

print("Total Null:",data.isnull().sum())

plt.hist(data['mpg'],bins=1)
plt.show()

plt.scatter(data['mpg'],data['wt'])
plt.show()

df=pd.DataFrame(data,columns=['gear'])
print("Count How many values:\n",df['gear'].value_counts())
