

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # Holt Winter's Exponential Smoothing


###########################################################
df=pd.read_csv('H:/vrenv/assignments/KAGGLE PROJECT FORCASTING/NEC stock market/NSE Training Data - 1st Jan 2016 to 1st Jan 2022.csv',parse_dates=True)

################################# Datetype ########################
df['Date']=pd.to_datetime(df['Date'])

############# Custom Impute  #############
df[df['Close'].isnull()]['Close'][151].astype('str')
custom_impute=np.where(df['Close'].astype('str')=='nan')
for j in list(custom_impute[0]):
    df['Close'][j]=(df['Close'][j-1]+df['Close'][j+1])/2

########################## Date Index ####################
df1=df.set_index('Date')

################## Close market Plot ########################
df1['Close'].plot()
df1['Close'].rolling(12).std().plot()
########## seasonal decompose plot #########################
decom=seasonal_decompose(df1['Close'],model='additive',period=12)
decom.seasonal.plot()
decom.plot()

########### decomposing seasonality using log transformation ##########
df1['log_deseasonal']=df1['Close'].apply(lambda x: np.log(x))
log_deseason=df1['log_deseasonal']-df1['log_deseasonal'].shift(12)
log_deseason=log_deseason.dropna()
log_deseason.plot()
###############  before and after removing seasonality #########

plot_acf(df1['Close']) # before removing seasonality
plot_acf(log_deseason) # after removing seasonality

######################### checking stationary or not after deseasonalizing #########
adfuller(log_deseason)[1] # result is stationary bcz p value is less than 0.05

############### 
adfuller((df['Close']**0.5).rolling(12).mean().dropna())
#####################
import pmdarima as pm
ar_model = pm.auto_arima(log_deseason, start_p=0, start_q=0,
                      max_p=12, max_q=12, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, trace=True,
                      error_action='warn', stepwise=True)
#######################################
# ARIMA IS NOT WORKING GOOD ON THIS DATA 

ar=ARIMA(log_deseason[:1457],(1,0,0))
fitted=ar.fit(disp=1)
f=fitted.predict(start=1457,end=1480)
#f,se,cof=fitted.forecast(1480)
df['Close'].plot()
pd.Series(f).plot()


######################## SARIMAX ####################
##################### SARIMAX PERFORMED WELL ON THIS DATA ##############
test=pd.read_csv('H:/vrenv/assignments/KAGGLE PROJECT FORCASTING/NEC stock market/test.csv')

model=sm.tsa.statespace.SARIMAX(df['Close'][0:1457],order=(0,1,0),seasonal_order=(1,1,1,12))
result=model.fit()
pred=result.predict(start=1457,end=1480,dynamic=True)
final_pred=result.predict(start=1481,end=1503)
from sklearn.metrics import mean_squared_log_error
mean_squared_log_error(df['Close'][1457:],pred)

mean_squared_log_error(df['Close'][1458:],final_pred)

df['Close'].plot()
df['Close'][1456:].plot()
pred.plot()
final_pred.plot()
final=pd.DataFrame(final_pred)
final=final.reset_index(drop=True)
final=pd.concat([test,final],axis=1)
final.columns
mean_squared_log_error(final['Close'],final['predicted_mean'])
############################

