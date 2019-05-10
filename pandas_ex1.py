import numpy as np
import pandas as pd

def header(msg):
    print('-'*50)
    print('[ ' + msg + ' ]')

filename = 'Fremont_weather.txt'
df = pd.read_csv(filename)
#print(df)
#print(df.head())
#print(df.tail(3))
#print(df.dtypes)
#print(df.index)
#print(df.columns)
#print(df.values)
#header("1. print df")
#print(df)

header("7. slicing -- df[['avg_low', 'avg_high']]")
print(df[['avg_low', 'avg_high']])

header("7. slicing -- df.loc[:, ['avg_low', 'avg_high']]")
print(df.loc[:,['avg_low', 'avg_high']])

header("7. slicing -- df.loc[9, ['precipitation']]")
print(df.loc[9,['avg_low', 'avg_high']])

header("7. slicing -- df.iloc[3:5, [0,3]]")
print(df.iloc[3:5, [0,3]])

# 8. filtering
header("8. df[df.avg_precipitation > 1.0]")
print(df[df.avg_precipitation > 1.0])

header("8. df[df['month'].isin(['Jun','Jul','Aug'])]")
print(df[df['month'].isin(['Jun','Jul','Aug'])])