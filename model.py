import pandas as pd
import numpy as np
import pickle
df=pd.read_csv('hiring.csv')
df
df['experience'].fillna(0,inplace=True)
df['test_score'].fillna(df['test_score'].mean(),inplace=True)
x=df.iloc[:,:3]
y=df.iloc[:,-1]
def converttoint(word):
    word_dict = {'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'zero':0,0:0}
    return word_dict[word]
x['experience'] = x['experience'].apply(lambda x: converttoint(x))

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x,y)
pickle.dump(regressor,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[0,0,0]]))