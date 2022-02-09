import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data= pd.read_csv('NSE Training Data - 1st Jan 2016 to 1st Jan 2022.csv', index_col='Date')
print(data.isna().sum())
data.dropna(inplace=True)
print(data.isna().sum().sum())

print(data.duplicated().any())
df=data['Close']


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df=scaler.fit_transform(np.array(df).reshape(-1,1))

training_size=int(len(df)*0.65)
test_size=len(df)-training_size
train_data=df[0:training_size,:]
test_data=df[training_size:len(df),:1]

print(training_size,test_size)

#Creating train & test data 
def create_dataset(dataset, step_time=1):
	data_X, data_Y = [], []
	for i in range(len(dataset)-step_time-1):
		a = dataset[i:(i+step_time), 0]   
		data_X.append(a)
		data_Y.append(dataset[i + step_time, 0])
	return np.array(data_X), np.array(data_Y)

step_time = 50
X_train, y_train = create_dataset(train_data, step_time)
X_test, y_test = create_dataset(test_data, step_time)

print(X_train.shape), print(y_train.shape)

#tesorflow-LSTM model requires three dimentional array!
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout

regressor =Sequential()
regressor.add(LSTM(50,return_sequences=True,input_shape=(50,1)))
regressor.add(LSTM(50,return_sequences=True))
regressor.add(LSTM(50))
regressor.add(Dense(1))
regressor.compile(loss='mean_squared_error',optimizer='adam')

regressor.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=64,verbose=1)

y_pred=regressor.predict(X_test)
y_pred=scaler.inverse_transform(y_pred)
y_test=y_test.reshape(y_test.shape[0],1)
y_test=scaler.inverse_transform(y_test)
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test, y_pred)

rmse=(mse)**0.5
print(rmse)

plt.plot(y_pred, label= "predicted")
plt.plot(y_test, label="actual")
plt.ylabel("NIFTY")
plt.legend()
plt.show()


print(len(test_data))
fut_inp=test_data[467:]
fut_inp=fut_inp.reshape(1,-1)
print(fut_inp.shape)

tmp_inp=list(fut_inp)
tmp_inp=tmp_inp[0].tolist()

#predicting for next 40 days
lst_output=[]
n_steps=50
i=0
while(i<40):
    
    if(len(tmp_inp)>50):
        fut_inp = np.array(tmp_inp[1:])
        fut_inp=fut_inp.reshape(1,-1)
        fut_inp = fut_inp.reshape((1, n_steps, 1))
        yhat = regressor.predict(fut_inp, verbose=0)
        tmp_inp.extend(yhat[0].tolist())
        tmp_inp = tmp_inp[1:]
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        fut_inp = fut_inp.reshape((1, n_steps,1))
        yhat = regressor.predict(fut_inp, verbose=0)
        tmp_inp.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)

print(len(df))

plot_new=np.arange(1,51)
plot_pred=np.arange(51,91)

plt.plot(plot_new, scaler.inverse_transform(df[1425:]))
plt.plot(plot_pred, scaler.inverse_transform(lst_output))
plt.show()

new_df=df.tolist()
print(len(new_df))

new_df.extend(lst_output)
new=scaler.inverse_transform(new_df)
plt.plot(new)
plt.title("Prediction")
plt.ylabel('NIFTY')
plt.show()

print(scaler.inverse_transform(lst_output))








