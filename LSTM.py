import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import tensorflow
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping
from keras.layers import ConvLSTM2D

# load the dataset
df = read_csv('C:/Users/DELL/Downloads/NYA_Stock1.csv', usecols=[5])

#clean dataset
df = df[df['Close'].notna()]
df=df.reset_index()['Close']
print(df.head())
print(df.tail())
df = df.values
df = df.astype('float32')

#plotting dataset
plt.plot(df)
plt.show()

#checking if the dataset is stationary
from statsmodels.tsa.stattools import adfuller
adf, pvalue, usedlag_, nobs_, critical_values_, icbest_ = adfuller(df)
print("pvalue = ", pvalue, " if above 0.05, data is not stationary")

#plotting autocorrelation
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df)
plt.show()

# normalize the dataset
scaler=MinMaxScaler(feature_range=(0,1))
df=scaler.fit_transform(np.array(df).reshape(-1,1))

# split into train and test sets
training_size=int(len(df)*0.65)
test_size=len(df)-training_size
train_data,test_data=df[0:training_size,:],df[training_size:len(df),:1]

#create x_train,y_train,x_test,y_test
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)

# Number of time steps to look back
time_step = 100
x_train, y_train = create_dataset(train_data, time_step)
x_test, y_test = create_dataset(test_data, time_step)

# reshape input to be [samples, time steps, features] which is required for LSTM
x_train =x_train.reshape(x_train.shape[0],x_train.shape[1] , 1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1] , 1)

# Stacked LSTM model creation
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.summary()
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=50,batch_size=64,verbose=1)

# make predictions
train_predict=model.predict(x_train)
test_predict=model.predict(x_test)

#Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)
y_train =y_train.reshape(y_train.shape[0],1)
y_test =y_test.reshape(y_test.shape[0],1)
y_train=scaler.inverse_transform(y_train)
y_test=scaler.inverse_transform(y_test)

#performance metrics
print('Performance Metrics')
print('RMSE',math.sqrt(mean_squared_error(y_train[:,0],train_predict[:,0])))
### Test Data RMSE
print('RMSE:',math.sqrt(mean_squared_error(y_test[:,0],test_predict[:,0])))

# Shifting value and Plotting

look_back=100
train_predict1 = np.empty_like(df)
train_predict1[:, :] = np.nan
train_predict1[look_back:len(train_predict)+look_back, :] = train_predict

test_predict1 = np.empty_like(df)
test_predict1[:, :] = np.nan
test_predict1[len(train_predict)+(look_back*2)+1:len(df)-1, :] = test_predict

plt.plot(scaler.inverse_transform(df))
plt.plot(train_predict1)
plt.plot(test_predict1)
plt.show()

# predicting next 30 future stock value
k=len(test_data) - 100
x_input=test_data[k:].reshape(1,-1)
x_input.shape
temp_input=list(x_input)
temp_input=temp_input[0].tolist()
y_output = []
n_steps = 100
i = 0
while (i < 30):

	if (len(temp_input) > 100):
		x_input = np.array(temp_input[1:])
		x_input = x_input.reshape(1, -1)
		x_input = x_input.reshape((1, n_steps, 1))
		# print(x_input)
		y_out = model.predict(x_input, verbose=0)
		temp_input.extend(y_out[0].tolist())
		temp_input = temp_input[1:]
		y_output.extend(y_out.tolist())
		i = i + 1
	else:
		x_input = x_input.reshape((1, n_steps, 1))
		y_out = model.predict(x_input, verbose=0)
		temp_input.extend(y_out[0].tolist())
		y_output.extend(y_out.tolist())
		i = i + 1
print('Future 30 stock value predicted :')
print(scaler.inverse_transform(y_output))
day_new=np.arange(1,101)
day_pred=np.arange(101,131)
k=len(df)-100
plt.plot(day_new,scaler.inverse_transform(df[k:]))
plt.plot(day_pred,scaler.inverse_transform(y_output))
plt.show()
df3=df.tolist()
df3.extend(y_output)
df3=scaler.inverse_transform(df3).tolist()
plt.plot(df3)
plt.show()