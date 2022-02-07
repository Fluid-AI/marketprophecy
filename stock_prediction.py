import warnings
import pandas as pd
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

df_btc = pd.read_csv('BTC-USD Training Data - 1st Jan 2016 to 1st Jan 2022.csv',index_col='Date')
df2_btc = pd.read_csv('BTC-USD Out of Time Testing 1st Jan 2022 to 4th Feb 2022.csv',index_col='Date')

df_nasdaq = pd.read_csv('NASDAQ Training Data - 1st Jan 2016 to 1st Jan 2022.csv',index_col='Date')
df2_nasdaq = pd.read_csv('NASDAQ Out of Time Data - 1st Jan 2022 to 4th Feb 2022.csv',index_col='Date')

df_nse = pd.read_csv('NSE Training Data - 1st Jan 2016 to 1st Jan 2022.csv',index_col='Date')
df2_nse = pd.read_csv('NSE Out of Time Testing Data - 1st Jan 2022 to 4th Feb 2022.csv',index_col='Date')

def fillNA(df): #Filling NULL/NaN values

    cols = df.columns
    for i in cols:
        if(df[i].isnull().sum()>0):
            df[i].fillna(value=df[i].mean(), inplace=True)

    return df

def transformDF(df):    #Scaling and Transforming Data

    df.drop(['Close'],1,inplace=True)
    df['HL_PCT'] = ((df['High']-df['Adj Close'])/df['Adj Close'])*100
    df['PCT_change'] = ((df['Adj Close']-df['Open'])/df['Open'])*100

    df = df[['Adj Close', 'HL_PCT', 'PCT_change','Low', 'Volume',]]

    features = ['Adj Close', 'HL_PCT', 'PCT_change','Low', 'Volume']
    scaler = MinMaxScaler()
    transformed = scaler.fit_transform(df[features])
    df = pd.DataFrame(columns=features, data=transformed, index=df.index)

    return df

def setDivision(df):    #Dividing Training and Testing Set

    X = np.array(df.drop(['Adj Close'],1))

    df.dropna(inplace=True)
    y = np.array(df['Adj Close'])

    return X,y

def Pred(X_test, clf):  #Prediction 

    prediction_set = clf.predict(X_test)
    return prediction_set
    
def Acc(X_train,y_train,X_test,y_test,clf): #Training and Testing Accuracy

    accuracyTest = clf.score(X_test, y_test)
    accuracyTrain = clf.score(X_train, y_train)
    print("Training score: {0}\nTesting Score: {1}"
          .format(round(accuracyTrain*100,3),round(accuracyTest*100,3)))

def classifier(X,y,X2,y2):  #Building Classifier

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4)
    clf1 = LinearRegression()
    clf1.fit(X_train,y_train)
    clf2 = svm.SVR()
    clf2.fit(X_train,y_train)
    print("Using Linear Regression:\n")
    Acc(X_train,y_train,X_test,y_test,clf1)
    predReg = Pred(X_test,clf1)
    predOut1 = Pred(X2,clf1)
    print("\nUsing Support Vector Regression:\n")
    Acc(X_train,y_train,X_test,y_test,clf2)
    predSVR = Pred(X_test,clf2)
    predOut2 = Pred(X2,clf2)

    string1 = "Prediction Using Linear Regression"
    string2 = "Prediction Using SVR"
    visual(y2,predReg,y_test,predOut1,string1)
    visual(y2,predSVR, y_test,predOut2,string2)

def visual(y2,prediction_set, y_test,predOut,string):   #Visualization

    #plt.plot(y_test, label='Original')
    #plt.plot(prediction_set, label='Predicted')
    plt.plot(y2, label='Out of time Original')
    plt.plot(predOut, label='Out of time Predicted')
    plt.legend()
    plt.title(string)
    plt.xlabel('Date')
    plt.ylabel('Adj Close')
    plt.show()

def data(df_):

    df_ = fillNA(df_)
    df_ = transformDF(df_)
    X,y = setDivision(df_)

    return X,y

def calling(df_,df2):   #For using Different Datasets

    X,y = data(df_)
    X2,y2 = data(df2)
    classifier(X,y,X2,y2)

    
print("\nFor NSE dataset:\n")
calling(df_nse,df2_nse)
print("\nFor NASDAQ dataset:\n")
calling(df_nasdaq,df2_nasdaq)
print("\nFor BTC dataset:\n")
calling(df_btc,df2_btc)


