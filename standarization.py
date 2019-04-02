import scipy
from scipy import stats
import pandas as pd
import numpy as np
from sklearn import preprocessing

dataset = pd.read_csv('iris.csv', sep='[;\t,]', engine='python')

# print(dataset.head())
'''
v1 = dataset.iloc[:, 1]
v2 = dataset.iloc[:, 2]
v3 = dataset.iloc[:, 3]
v4 = dataset.iloc[:, 4]


print("\nPierwsza kolumna: \n", scipy.stats.normaltest(v1))  # rozkład normalny
print(v1.mean(axis=0))
print(v1.std(axis=0))
print("\nDruga kolumna: \n", scipy.stats.normaltest(v2))  # rozkład normalny
print(v2.mean(axis=0))
print(v2.std(axis=0))
print("\nTrzecia kolumna: \n", scipy.stats.normaltest(v3))  # rozkład nienormalny
print(v3.mean(axis=0))
print(v3.std(axis=0))
print("\nCzwarta kolumna: \n", scipy.stats.normaltest(v4))  # rozkład nienormalny
print(v4.mean(axis=0))
print(v4.std(axis=0))
'''


x = dataset.iloc[:, 1:].values

print("\n\n ----------------------- First method ----------------- \n\n ")

X_scaled = preprocessing.scale(x)
print("\n", X_scaled)
print("\nśrednia: ", X_scaled.mean(axis=0))
print("\nodchylenie st.: ", X_scaled.std(axis=0))

scaler = preprocessing.StandardScaler().fit(x)
print(scaler)



print("\n\n ----------------------- Second method ----------------- \n\n ")


scaler2 = preprocessing.StandardScaler().fit(x)
rescaledX = scaler2.transform(x)

# summarize transformed data
np.set_printoptions(precision=3)
print("\nRescaled: \n", rescaledX[0:5, :])
print(rescaledX.mean(axis=0))
print(rescaledX.std(axis=0))

# The values for each attribute now have a mean value of 0 and a standard deviation of 1.

print("\n\n ----------------------- Third method ----------------- \n\n ")


standscaler = preprocessing.StandardScaler()
standscaler_df = standscaler.fit_transform(x)
st_df = pd.DataFrame(standscaler_df)
print("st_df", st_df)
print(st_df.mean(axis=0))
print(st_df.std(axis=0))
