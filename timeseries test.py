#%%
import pandas as pd
import numpy as np
import time 
import tensorflow as tf
import tensorflow_datasets as tfds
# from transformer_redone import *

import tensorflow.keras as keras
from tensorflow.keras.layers import Bidirectional, Flatten
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Embedding, LSTM, Embedding
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam

from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures

import seaborn as sns

import mlflow
import mlflow.tensorflow
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
#%%

# final df3 is just the bitcoins, df5 is max length 800.
# df = pd.read_csv("data/full_df_trimmeddown.csv")#.tail(25000)

#%%
# new_arr = np.concatenate((np.array(list(df.sents)), np.array(list(df['diff'])).reshape(-1, 1)), axis=1)
#%%
# df2 = pd.DataFrame(new_arr, columns=['neg', 'neu', 'pos', 'comp', 'diff'])

#%%
# df3 = pd.read_csv("data/bitcoin_price_diffs.csv", parse_dates=[1])

# cols = ['high', 'close', 'open', 'low']

# for col in cols:
#   try:
#     df3[col] = df3[col].apply(lambda x: float(x.replace(',', '')))
#   except:
#     continue

#%%
# newretdf = pd.read_csv("data/bitcoin_ret_diff_weekly.csv")
# #%%
# newretdf.date = newretdf.date.apply(pd.to_datetime)
# #%%
# newretdf = newretdf[newretdf.date > pd.to_datetime('2018-01-01')]

#%%
# newretdf.corr()

df3 = pd.read_csv("data/bitcoin_ret_diff_weekly.csv", parse_dates=[0])
df3 = df3[df3.date > pd.to_datetime('2018-01-01')]

#%%
df3.corr()
#%%
start_date = pd.to_datetime('January, 2018')
end_date = pd.to_datetime('October, 2020')
view = df3[df3.date > start_date]
plt.figure(figsize=(20, 10))
# plt.plot(view.date, view.open, c='r')
plt.plot(view.date, view['ret'], c='b')
#%%
fig, (ax1) = plt.subplots(ncols=1, figsize=(8, 5))
ax1.set_title('Before Scaling')
sns.kdeplot(df3['ret'], ax=ax1)

#%%
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()#feature_range = (0, 1))
df3['scaled_ret'] = scaler.fit_transform(pd.DataFrame(df3['ret']))#pd.DataFrame(scaler.fit_transform(pd.DataFrame(df3['ret'])),columns=['ret'])
print('Shape:' , df3.shape[0])
df3.head(5)


#%%
df = df3
fig, (ax1) = plt.subplots(ncols=1, figsize=(8, 5))
ax1.set_title('After Scaling')
sns.kdeplot(df['scaled_ret'], ax=ax1)


#%%

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)-n_steps*2):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:end_ix+n_steps]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

# choose a number of time steps
n_steps = 4
# split into samples
rawsequence = np.array(df3[['pos', 'neg', 'num_topics', 'num_responses', 'scaled_ret']])
print(rawsequence[0:12])
X, y = split_sequence(rawsequence, n_steps)
# summarize the data
print(len(X), len(y))
print(X.shape, y.shape)
print(n_steps)
#for i in range(len(X)):
#	print(X[i], y[i])
X =y[:, :,:-1].reshape(X.shape[0], X.shape[1], 4)
y =y[:, :,-1].reshape(y.shape[0], y.shape[1], 1)

#%%
# model = Sequential()
# model.add(LSTM(units=32, activation='relu', input_shape=(n_steps, 1), return_sequences=True))
# #model.add(LSTM(units=32, activation='relu', input_shape=(len_sequence, 1), return_sequences=False))
# model.add(Dense(1, activation='softmax'))

from keras.layers import Bidirectional, Flatten

input_layer = Input( shape=(n_steps, 4))
lstm = Bidirectional(LSTM(32, return_sequences=True, dropout=0.3))(input_layer, training = True)
lstm = Bidirectional(LSTM(16, return_sequences=True, dropout=0.3))(lstm, training = True)
lstm = LSTM(1, return_sequences=True, dropout=0.3)(lstm, training = True)

model = Model(input_layer, lstm)

model.compile(optimizer='adam', loss='mse', metrics=['mse', 'accuracy'])
print(model.summary())

#%%

print(X.shape)
X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))
print(X.shape)
#%%
print(y.shape)
y = y.reshape((y.shape[0],y.shape[1], y.shape[2]))
print(y.shape)

#%%
batch_size=16
epochs=100
from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('timelstm4.hdf5', monitor='mse', verbose=1, save_best_only=True, mode='max')
model.fit(x=X, y=y,
        batch_size=batch_size, epochs=epochs,
        verbose=1, 
        callbacks=[checkpoint])

#%%
from ast import literal_eval
#%%
import math
from sklearn.metrics import mean_squared_error

sequence = np.array(df[['pos', 'neg', 'num_topics', 'num_responses', 'scaled_ret']])
print(sequence)
time_steps = 4
samples = len(sequence)
trim = samples % time_steps
subsequences = int(samples/time_steps)
sequence_trimmed = sequence[:samples - trim]



print(samples, subsequences)
sequence_trimmed.shape = (subsequences, time_steps, 5)
print(sequence_trimmed.shape)

testing_dataset = sequence_trimmed
print("testing_dataset: ", testing_dataset.shape)

testing_pred = model.predict(x=testing_dataset)
print("testing_pred: ", testing_pred.shape)

testing_dataset = testing_dataset.reshape((testing_dataset.shape[0]*testing_dataset.shape[1]), testing_dataset.shape[2])
print("testing_dataset: ", testing_dataset.shape)

testing_pred = testing_pred.reshape((testing_pred.shape[0]*testing_pred.shape[1]), 1)
print("testing_pred: ", testing_pred.shape)
errorsDF = testing_dataset - testing_pred
print(errorsDF.shape)
rmse = math.sqrt(mean_squared_error(testing_dataset, testing_pred))
print('Test RMSE: %.3f' % rmse)
#%%
#based on cutoff after sorting errors
dist = np.linalg.norm(testing_dataset - testing_pred, axis=-1)

scores =dist.copy()
print(scores.shape)
scores.sort()
cutoff = int(0.999 * len(scores))
print(cutoff)
#print(scores[cutoff:])
threshold= scores[cutoff]
print(threshold)

#%%
plt.figure(figsize=(24,16))
plt.plot(testing_dataset[:,-1], color='green')
plt.plot(testing_pred, color='red')


#%%
model.save("time_series_lstm2.h5")
#%%

new_data = pd.read_csv('data/bitcoin_prices_november.csv', parse_dates=[0])

# %%
cols = ['High', 'Close', 'Open', 'Low']

for col in cols:
  try:
    new_data[col] = new_data[col].apply(lambda x: float(x.replace(',', '')))
  except:
    continue
# %%
plt.plot(new_data['Date'], new_data['Close'])
# %%
#%%
fig, (ax1) = plt.subplots(ncols=1, figsize=(8, 5))
ax1.set_title('Before Scaling')
sns.kdeplot(new_data['Close'], ax=ax1)

#%%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))
new_data['scaled_close'] = pd.DataFrame(scaler.fit_transform(pd.DataFrame(new_data['Close'])),columns=['close'])
print('Shape:' , new_data.shape[0])
new_data.head(5)


#%%
df2 = new_data
fig, (ax1) = plt.subplots(ncols=1, figsize=(8, 5))
ax1.set_title('After Scaling')
sns.kdeplot(df2['scaled_close'], ax=ax1)


#%%

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)-n_steps*2):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:end_ix+n_steps]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

# choose a number of time steps
n_steps = 7
# split into samples
rawsequence = np.array(new_data[['scaled_close']])
print(rawsequence[0:12])
X, y = split_sequence(rawsequence, n_steps)
# summarize the data
print(len(X), len(y))
print(X.shape, y.shape)
print(n_steps)
#for i in range(len(X)):
#	print(X[i], y[i])

# %%
preds = model.predict(X)
y_true = y.reshape((y.shape[0]*y.shape[1]), y.shape[2])
print("testing_dataset: ", y_true.shape)

preds = preds.reshape((preds.shape[0]*preds.shape[1]), 1)
print("testing_pred: ", preds.shape)
errorsDF = y_true - preds
print(errorsDF.shape)
rmse = math.sqrt(mean_squared_error(y_true, preds))
print('Test RMSE: %.3f' % rmse)
#%%
#based on cutoff after sorting errors
dist = np.linalg.norm(y_true - preds, axis=-1)

scores =dist.copy()
print(scores.shape)
scores.sort()
cutoff = int(0.999 * len(scores))
print(cutoff)
#print(scores[cutoff:])
threshold= scores[cutoff]
print(threshold)

#%%
plt.figure(figsize=(24,16))
plt.plot(y_true, color='green')
plt.plot(preds, color='red')
# %%
