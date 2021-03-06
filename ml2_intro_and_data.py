import numpy as np
import pandas as pd
import quandl, math
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def header(msg):
    print('-'*50)
    print('[ ' + msg + ' ]')

df = quandl.get('WIKI/GOOGL')
header("df.head")
print(df.head)

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
header("df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]")
print(df.tail)

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

header("Add columns['HL_PCT','PCT_Change']")
print(df.head)

header("Slicing columns['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']")
df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]
print(df.head)

'''
forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])
X = preprocessing.scale(X)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

clf = LinearRegression()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print(accuracy)
'''