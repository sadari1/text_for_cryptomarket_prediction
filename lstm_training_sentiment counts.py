#%%
import pandas as pd
import numpy as np
import time 
import tensorflow as tf
import tensorflow_datasets as tfds
# from transformer_redone import *
from playsound import playsound

import tensorflow.keras as keras
from tensorflow.keras.layers import Bidirectional, Flatten
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Embedding, LSTM, Embedding
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam

from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures

import mlflow
import mlflow.tensorflow
# from sentence_transformers import SentenceTransformer, util
# model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
from sklearn.metrics import roc_auc_score

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

#%%

# final df3 is just the bitcoins, df5 is max length 800.
# df = pd.read_csv("data/full_df_trimmeddown.csv")#.tail(25000)
# df = pd.read_csv("data/full_df_trimmeddown.csv")#.tail(25000)
prices = pd.read_csv("data/bitcoin_price_diffs.csv", parse_dates=[1])

#%%
offset_interval = 7
max_date = prices.date.max()
start_date = prices.date.iloc[0]

current_date = start_date
running = True

pd_array = []
cols = ['start_date', 'end_date', 'return', 'dif']
while(running):
  
  curr_price = prices[prices.date == current_date].close.iloc[0]
  next_week = current_date + pd.DateOffset(days=offset_interval)

  if next_week > max_date:
    running = False
    break

  next_price = prices[prices.date == next_week].close.iloc[0]
  ret = (next_price - curr_price) / curr_price
  dif = 1 if next_price > curr_price else 0
  row = [current_date, next_week, ret, dif]
  pd_array.append(row)

  current_date = next_week




#%%
retdf = pd.DataFrame(pd_array, columns=cols)
retdf

#%%
retdf.to_csv("data/bitcoin_weeklyreturns_dif.csv", index=False)
#%%
retdf = pd.read_csv("data/bitcoin_weeklyreturns_dif.csv")
#%%
tic = time.time()
bitcoindf = pd.read_csv("data/bitcoin_smallthreads.csv", parse_dates=[0])
# newsdf = pd.read_csv("data/abcnews-date-text.csv").tail(210000)
# getnewsdates = lambda x: pd.to_datetime(f"{str(x)[:4]}-{str(x)[4:6]}-{str(x)[-2:]}")
# newsdf.publish_date = newsdf.publish_date.apply(getnewsdates)

toc = time.time() 
print(f"Loading took {toc-tic} seconds.")
playsound('audio.wav')

# newsdf.head()
#%%
# newsdf.to_csv("data/newsdata_correcteddate.csv", index=False)
#%%

# tic = time.time()
# pd_array = []
# columns = ['date', 'pos', 'neg', 'ret']
# for f in range(len(retdf.iloc[:])):
#   start_date = retdf.iloc[f].start_date
#   end_date = retdf.iloc[f].end_date 
#   print(f"On date: {start_date}")
#   timea = time.time()
#   window = newsdf[(newsdf.publish_date >= start_date) & (newsdf.publish_date <= end_date)]
#   # lengths = [len(f"{g}") for g in list(window.text)]
#   # window['lengths'] = lengths

#   # window = window[(window.lengths > 50) & (window.lengths < 300)]

#   pos = 0
#   neg = 0

#   # for topic in window.topic.unique():
#   #   topic = list(sid.polarity_scores(topic).values())
#   #   if topic[-1] > 0.11:# or topic[-1] > 0.31:
#   #     pos+=1
#   #   if topic[-1] < -0.11: #or topic[-1] < -0.31:
#   #     neg +=1

#   for g in range(len(window)):
    
#     text = list(sid.polarity_scores(window.headline_text.iloc[g]).values())

#     if text[-1] >= 0.31:# or topic[-1] > 0.31:
#       pos+=1
#     if text[-1] < 0.31: #or topic[-1] < -0.31:
#       neg +=1
  
#   ret = retdf.iloc[f]['return']

#   row = [end_date, pos, neg, ret]
#   pd_array.append(row)
#   timeb= time.time()
#   print(f"\tTook {timeb-timea} seconds.")
#     # print(f"{window.topic.iloc[g][:50]}, {topic}")
#     # print(f"\n\n{window.text.iloc[g][:150]}, {text}\n\n")

#   # print(topic, text)
# toc = time.time()
# print(f"Job took {toc-tic} seconds")
# playsound("audio.wav")

# #%%

# newretdf = pd.DataFrame(pd_array, columns=columns)
# newretdf
# #%%
# newretdf.ret = newretdf.ret.apply(lambda x: x * 100)
# #%%
# newretdf.corr()
#%%
#%%
# Now go through the bitcoin data one week at a time, get sentiments, and get
# the number of counts of each sentiment. 

tic = time.time()
pd_array = []
columns = ['date', 'pos', 'neg', 'num_topics', 'num_responses', 'ret', 'dif']
for f in range(len(retdf.iloc[:])):
  start_date = retdf.iloc[f].start_date
  end_date = retdf.iloc[f].end_date 
  print(f"On date: {start_date}")
  timea = time.time()
  window = bitcoindf[(bitcoindf.date >= start_date) & (bitcoindf.date <= end_date)]
  lengths = [len(f"{g}") for g in list(window.text)]
  window['lengths'] = lengths

  window = window[(window.lengths > 10)]# & (window.lengths < 600)]

  pos = 0
  neg = 0

  for topic in window.topic.unique():
    topic = list(sid.polarity_scores(topic).values())
    if topic[-1] > 0.31:# or topic[-1] > 0.31:
      pos+=1
    if topic[-1] < -0.31: #or topic[-1] < -0.31:
      neg +=1

  for g in range(len(window)):
    text = window.text.iloc[g]

    chunks, chunk_size = len(text), 100
    texts = [ text[i:i+chunk_size] for i in range(0, chunks, chunk_size) ]
    for text in texts:

      sent = list(sid.polarity_scores(text).values())

      if sent[-1] > 0.31:# or topic[-1] > 0.31:
        pos+=1
      if sent[-1] < -0.31: #or topic[-1] < -0.31:
        neg +=1
    
  ret = retdf.iloc[f]['return']

  row = [end_date, pos, neg, len(window.topic.unique()), len(window), ret, retdf.iloc[f]['dif']]
  pd_array.append(row)
  timeb= time.time()
  print(f"\tTook {timeb-timea} seconds.")
    # print(f"{window.topic.iloc[g][:50]}, {topic}")
    # print(f"\n\n{window.text.iloc[g][:150]}, {text}\n\n")

  # print(topic, text)
toc = time.time()
print(f"Job took {toc-tic} seconds")
playsound("audio.wav")

#%%

newretdf = pd.DataFrame(pd_array, columns=columns)
newretdf
#%%
# newretdf.ret = newretdf.ret.apply(lambda x: x * 100)
#%%
newretdf.corr()
#%%
newretdf.to_csv("data/bitcoin_ret_diff_weekly.csv", index=False)

#%%
newretdf = pd.read_csv("data/bitcoin_ret_diff_weekly.csv")
#%%
newretdf.date = newretdf.date.apply(pd.to_datetime)
#%%
newretdf = newretdf[newretdf.date > pd.to_datetime('2017-01-01')]
# 2018-01-01
# 2019-05-01
# 2020-02-01
# 2020-08-01
#%%
newretdf.corr()
#%%

#%%
until = 50
x_train = newretdf[['pos', 'neg', 'num_topics', 'num_responses']].iloc[:-until]
y_train = newretdf['ret'].iloc[:-until]#keras.utils.to_categorical(df['ret'].iloc[:-100])

x_test =newretdf[['pos', 'neg', 'num_topics', 'num_responses']].iloc[-until:]
y_test = newretdf['ret'].iloc[-until:]#keras.utils.to_categorical(df['ret'].iloc[-100:])
#%%
y_train, y_test = np.array(y_train), np.array(y_test)
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)
#%%
#%%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# scaler.fit(np.array(newretdf['ret']).reshape(len(newretdf), 1))

y_train = scaler.fit_transform(y_train)
y_test = scaler.transform(y_test)
#%%
# x_train = np.array([np.array(f).reshape(np.array(f).shape[0],  1) for f in x_train])
# x_test = np.array([np.array(f).reshape(np.array(f).shape[0], 1) for f in x_test])

x_train, x_test = np.array(x_train), np.array(x_test)

x_train = np.array(x_train).reshape(x_train.shape[0], x_train.shape[1])
x_test = np.array(x_test).reshape(x_test.shape[0], x_test.shape[1])
#%%

from tensorflow.keras.layers import Conv1D, BatchNormalization, LeakyReLU
from tensorflow.keras.regularizers import l1_l2
# from keras.utils.vis_utils import plot_model

batch_size = 64
# vocab_size = len(vocab)
input_layer = Input( shape=(x_train.shape[1]))
# emb = Embedding(vocab_size+1, 400, input_length = x_train.shape[1])(input_layer)
# lstm = Bidirectional(LSTM(800, batch_input_shape=(batch_size, x_train.shape[1], x_train.shape[2]),  return_sequences=True,  dropout=0.2))(input_layer, training = True)
# lstm = Bidirectional(LSTM(400,  batch_input_shape=(batch_size, x_train.shape[1], x_train.shape[2]),return_sequences=True,dropout=0.2))(lstm, training = True)
# lstm = Bidirectional(LSTM(200,  batch_input_shape=(batch_size, x_train.shape[1], x_train.shape[2]),return_sequences=True,dropout=0.2))(lstm, training = True)
# lstm = Bidirectional(LSTM(50,  batch_input_shape=(batch_size, x_train.shape[1], x_train.shape[2]),return_sequences=True,dropout=0.2))(lstm, training = True)
# lstm = Bidirectional(LSTM(200,  batch_input_shape=(batch_size, x_train.shape[1], x_train.shape[2]),return_sequences=True,dropout=0.2))(lstm, training = True)
# lstm = Bidirectional(LSTM(400,  batch_input_shape=(batch_size, x_train.shape[1], x_train.shape[2]),return_sequences=True, dropout=0.2))(lstm, training = True)
# lstm = Bidirectional(LSTM(800,  batch_input_shape=(batch_size, x_train.shape[1], x_train.shape[2]),return_sequences=True, dropout=0.2))(lstm, training = True)
# # cnn = Conv1D(x_train.shape[2], 1, 1, activation='softmax')(lstm)
# dense = Dense(x_train.shape[2], activation='softmax')(lstm)
# # lstm = LSTM(x_train.shape[1], return_sequences=True, activation='relu')(lstm)

h1 = Dense(16)(input_layer)
bn = BatchNormalization()(h1)
lr = LeakyReLU(alpha=0.999)(bn)
h2 = Dense(16, activation = 'linear')(h1)
bn = BatchNormalization()(h2)
lr = LeakyReLU(alpha=0.999)(bn)
# h2 = Dense(16, activation = 'linear')(h2)
# bn = BatchNormalization()(h2)
# lr = LeakyReLU(alpha=0.9)(bn)
h2 = Dense(8, activation = 'linear')(h2)
bn = BatchNormalization()(h2)
lr = LeakyReLU(alpha=0.999)(bn)
# h2 = Dense(8, activation = 'linear')(h2)
# lr = LeakyReLU(alpha=0.9)(h2)
# h2 = Dense(8, activation = 'linear')(h2)
# lr = LeakyReLU(alpha=0.9)(h2)
h2 = Dense(8, activation = 'linear')(h2)
lr = LeakyReLU(alpha=0.999)(h2)
# h2 = Dense(8, activation = 'linear')(h2)
# lr = LeakyReLU(alpha=0.9)(h2)
h2 = Dense(4, activation = 'linear')(h2)
lr = LeakyReLU(alpha=0.999)(h2)
h2 = Dense(4, activation = 'linear')(h2)
lr = LeakyReLU(alpha=0.999)(h2)
h2 = Dense(4, activation = 'linear')(h2)
lr = LeakyReLU(alpha=0.999)(h2)
# h2 = Dense(128, activation = 'relu', kernel_regularizer=l1_l2(1e-3, 1e-3))(h2)
# h2 = Dense(128, activation = 'relu', kernel_regularizer=l1_l2(1e-3, 1e-3))(h2)
# h2 = Dense(64, activation = 'relu', kernel_regularizer=l1_l2(1e-3, 1e-3))(h2)
# h2 = Dense(64, activation = 'relu', kernel_regularizer=l1_l2(1e-3, 1e-3))(h2)
# h2 = Dense(64, activation = 'relu', kernel_regularizer=l1_l2(1e-3, 1e-3))(h2)
# h2 = Dense(64, activation = 'relu', kernel_regularizer=l1_l2(1e-3, 1e-3))(h2)
# h2 = Dense(32, activation = 'relu', kernel_regularizer=l1_l2(1e-3, 1e-3))(h2)
# h2 = Dense(32, activation = 'relu', kernel_regularizer=l1_l2(1e-3, 1e-3))(h2)
# h2 = Dense(8, activation = 'relu', kernel_regularizer=l1_l2(1e-3, 1e-3))(h2)

# h2 = Dense(8)(h2)
# h2 = Dense(8)(h2)
# h2 = Dense(4)(h2)
# h2 = Dense(4)(h2)
# h2 = Dense(4)(h2)
# conv1 = Conv1D(filters=4, kernel_size=3, strides=1, padding='same',
# dilation_rate=1, activation='tanh', kernel_regularizer=l1_l2(1e-3, 1e-3))(input_layer)
# conv2 = Conv1D(filters=4, kernel_size=3, strides=1, padding='same',
# dilation_rate=1, activation='tanh', kernel_regularizer=l1_l2(1e-3, 1e-3))(conv1)
# conv3 = Conv1D(filters=128, kernel_size=5, strides=1, padding='valid',
# dilation_rate=1, activation='tanh', kernel_regularizer=l1_l2(1e-3, 1e-3))(conv2)
# conv4 = Conv1D(filters=128, kernel_size=5, strides=1, padding='valid',
# dilation_rate=1, activation='tanh', kernel_regularizer=l1_l2(1e-3, 1e-3))(conv3)
# conv5 = Conv1D(filters=64, kernel_size=5, strides=1, padding='valid',
# dilation_rate=1, activation='tanh', kernel_regularizer=l1_l2(1e-3, 1e-3))(conv4)
# conv6 = Conv1D(filters=64, kernel_size=3, strides=1, padding='valid',
# dilation_rate=1, activation='tanh', kernel_regularizer=l1_l2(1e-3, 1e-3))(conv5)
# conv7 = Conv1D(filters=32, kernel_size=3, strides=1, padding='valid',
# dilation_rate=1, activation='tanh', kernel_regularizer=l1_l2(1e-3, 1e-3))(conv6)
# conv8 = Conv1D(filters=32, kernel_size=3, strides=1, padding='valid',
# dilation_rate=1, activation='tanh', kernel_regularizer=l1_l2(1e-3, 1e-3))(conv7)
# conv9 = Conv1D(filters=16, kernel_size=3, strides=1, padding='valid',
# dilation_rate=1, activation='tanh', kernel_regularizer=l1_l2(1e-3, 1e-3))(conv8)
# flat = Flatten()(conv2)
# h1 = Dense(8)(flat) 
# h1 = Dense(8)(h1) 
# h1 = Dense(8)(h1) 
# h1 = Dense(4)(h1) 
# dense = Dense(1)(h1)
dense=  Dense(1, activation = 'linear')(h2)
dnn = Model(input_layer, dense)

print(dnn.summary())

# vocab_size = len(vocab)
# input_layer = Input( shape=( 38, 1222))
# # emb = Embedding(vocab_size+1, 200)(input_layer)
# lstm = Bidirectional(LSTM(200,  return_sequences=True, dropout=0.2))(input_layer, training = True)
# lstm = Bidirectional(LSTM(200,  return_sequences=True,dropout=0.2))(lstm, training = True)
# lstm = Bidirectional(LSTM(200,  return_sequences=True,dropout=0.2))(lstm, training = True)
# lstm = Dense(1222, activation='softmax')(lstm)

# model = Model(input_layer, lstm)
# plot_model(dnn, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
#%%


dnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse', metrics=[ 'mae', 'mse'])

#%%
from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('dnn3.hdf5', monitor='mse', verbose=1, save_best_only=True, mode='max')
batch_size=8
# epochs=900
epochs=1
# epochs=300

dnn.fit(x=x_train, y=y_train,
        batch_size=batch_size, epochs=epochs,
        verbose=1, callbacks=[checkpoint])

#%%
# dnn = keras.models.load_model("final_lstmbkp5.h5")

#%%

preds = dnn.predict(x_test)
# preds = dnn.predict(x_train)
# lrpreds = lr.predict(x_test)

## %%
import matplotlib.pyplot as plt

plt.plot(preds)
# plt.plot(lrpreds)
plt.plot(y_test)
# plt.plot(y_train)
plt.xticks(range(50))
plt.show()
# print((scaler.transform(y_test)-preds)**2 / len(preds))
# auc = roc_auc_score(np.argmax(y_test,axis=1), np.argmax(preds,axis=1))
# auc = roc_auc_score(np.argmax(y_test,axis=1), np.argmax(preds,axis=1))
# print(f"AUC: {auc}")
#%%
import matplotlib.pyplot as plt

plt.clf()
plt.figure(figsize=(15,10))
plt.plot(scaler.inverse_transform(preds))
# plt.plot(lrpreds)
plt.plot(scaler.inverse_transform(y_test))
# plt.plot(y_train)
plt.xticks(range(0, 50, 5))
plt.legend(['Predicted Returns', 'True Returns'])
plt.xlabel("Weeks")
plt.ylabel("Percent Return")
plt.title("Graph of Predicted Return vs True Return")
plt.savefig("finalreport_graph.png")
plt.show()
#%%
# dnn.save('final_lstmbkp5.h5')
#%%



ones_y = [1 if abs(f) > 1 else 0 for f in scaler.inverse_transform(y_test)]
ones_pred = [1 if abs(f) > 1 else 0 for f in scaler.inverse_transform(preds)]
##%%
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

one_auc = roc_auc_score(ones_y, ones_pred)
one_acc = accuracy_score(ones_y, ones_pred)
one_creport = classification_report(ones_y, ones_pred)
one_cm = confusion_matrix(ones_y, ones_pred)
#%%



ones_y = [1 if abs(f) > 10 else 0 for f in scaler.inverse_transform(y_test)]
ones_pred = [1 if abs(f) > 10 else 0 for f in scaler.inverse_transform(preds)]
##%%
from sklearn.metrics import accuracy_score, roc_auc_score

ten_auc = roc_auc_score(ones_y, ones_pred)
ten_acc = accuracy_score(ones_y, ones_pred)
ten_creport = classification_report(ones_y, ones_pred)
ten_cm = confusion_matrix(ones_y, ones_pred)

#%%
pd_array = [['1%', one_auc, one_acc], ['10%', ten_auc, ten_acc]]
pd.DataFrame(pd_array, columns=['Threshold', 'AUC', 'Accuracy'])

#%%

import seaborn as sns
# plt.figure(figsize=(10,10))
print("Threshold 1%")
plt.figure()
plt.clf()
ax = sns.heatmap(one_cm, annot=True,fmt='g')
ax.set_ylim([0,2])
ax.invert_xaxis()
ax.invert_yaxis()
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("Confusion Matrix - Threshold 1%")
plt.show()
print(one_creport)

#%%

import seaborn as sns
# plt.figure(figsize=(10,10))
print("Threshold 10%")
plt.figure()
plt.clf()
ax = sns.heatmap(ten_cm, annot=True,fmt='g')
ax.set_ylim([0,2])
ax.invert_xaxis()
ax.invert_yaxis()
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("Confusion Matrix - Threshold 10%")
plt.show()
print(ten_creport)

#%%
ipreds = scaler.inverse_transform(preds)
iytest = scaler.inverse_transform(y_test)
#%%
from sklearn.metrics import accuracy_score, roc_auc_score


roc_auc_score(ones_y, ones_pred), accuracy_score(ones_y, ones_pred)

#%%

#%%
batch_size=8
# epochs=900
epochs=1
# epochs=500

dnn.fit(x=x_train, y=y_train,
        batch_size=batch_size, epochs=epochs,
        verbose=1, callbacks=[checkpoint])

preds = dnn.predict(x_test)
# preds = dnn.predict(x_train)
# lrpreds = lr.predict(x_test)
#%%
plt.plot(preds - 2)
# plt.plot(lrpreds)
plt.plot(y_test)
# plt.plot(y_train)
plt.xticks(range(50))
plt.show()

#%%
# dnn.save("return_lstm2.h5")
#%%

new_data = pd.read_csv('data/bitcoin_prices_november.csv', parse_dates=[0])

#%%
new_data = new_data.rename(columns={"Date": "date", "Close": "close"}).sort_values('date')

#%%
str_replacer = lambda x: float(x.replace(',', ''))
new_data.close = new_data.close.apply(str_replacer)
#%%
offset_interval = 7
max_date = new_data.date.max()
start_date = new_data.date.iloc[0]

current_date = start_date
running = True

pd_array = []
cols = ['start_date', 'end_date', 'return', 'dif']
while(running):
  
  curr_price = new_data[new_data.date == current_date].close.iloc[0]
  next_week = current_date + pd.DateOffset(days=offset_interval)

  if next_week > max_date:
    running = False
    break

  next_price = new_data[new_data.date == next_week].close.iloc[0]
  ret = (next_price - curr_price) / curr_price
  dif = 1 if next_price > curr_price else 0
  row = [current_date, next_week, ret, dif]
  pd_array.append(row)

  current_date = next_week




#%%
novdf = pd.DataFrame(pd_array, columns=cols)
novdf

#%%
novdf.to_csv("data/bitcoin_november_weekly_returns.csv", index=False)

#%%
tic = time.time()
# novbitcoindf = pd.read_json("coinmarkcap_scrape/bitcoin_november_threads.json")#, parse_dates=[0])
# novbitcoindf.to_csv("data/bitcoin_nov_threads.csv", index=False)
df = pd.read_csv("data/bitcoin_nov_threads.csv", parse_dates=[0])
# newsdf = pd.read_csv("data/abcnews-date-text.csv").tail(210000)
# getnewsdates = lambda x: pd.to_datetime(f"{str(x)[:4]}-{str(x)[4:6]}-{str(x)[-2:]}")
# newsdf.publish_date = newsdf.publish_date.apply(getnewsdates)

toc = time.time() 
print(f"Loading took {toc-tic} seconds.")
playsound('audio.wav')

#%%

tic = time.time()
pd_array = []
columns = ['date', 'pos', 'neg', 'num_topics', 'num_responses', 'ret']
for f in range(len(novdf.iloc[:])):
  start_date = novdf.iloc[f].start_date
  end_date = novdf.iloc[f].end_date 
  print(f"On date: {start_date}")
  timea = time.time()
  window = df[(df.date >= start_date) & (df.date <= end_date)]
  lengths = [len(f"{g}") for g in list(window.text)]
  window['lengths'] = lengths

  window = window[(window.lengths > 10)]# & (window.lengths < 600)]

  pos = 0
  neg = 0

  for topic in window.topic.unique():
    topic = list(sid.polarity_scores(topic).values())
    if topic[-1] > 0.31:# or topic[-1] > 0.31:
      pos+=1
    if topic[-1] < -0.31: #or topic[-1] < -0.31:
      neg +=1

  for g in range(len(window)):
    text = window.text.iloc[g]

    chunks, chunk_size = len(text), 100
    texts = [ text[i:i+chunk_size] for i in range(0, chunks, chunk_size) ]
    for text in texts:

      sent = list(sid.polarity_scores(text).values())

      if sent[-1] > 0.31:# or topic[-1] > 0.31:
        pos+=1
      if sent[-1] < -0.31: #or topic[-1] < -0.31:
        neg +=1
    
  ret = novdf.iloc[f]['return']

  row = [end_date, pos, neg, len(window.topic.unique()), len(window), ret]
  pd_array.append(row)
  timeb= time.time()
  print(f"\tTook {timeb-timea} seconds.")
    # print(f"{window.topic.iloc[g][:50]}, {topic}")
    # print(f"\n\n{window.text.iloc[g][:150]}, {text}\n\n")

  # print(topic, text)
toc = time.time()
print(f"Job took {toc-tic} seconds")
playsound("audio.wav")

#%%

novbitcoindf = pd.DataFrame(pd_array, columns=columns)
novbitcoindf
#%%
# newretdf.ret = newretdf.ret.apply(lambda x: x * 100)
#%%
novbitcoindf.corr()

#%%
until = 50
x_nov = novbitcoindf[['pos', 'neg', 'num_topics', 'num_responses']]#.iloc[:-until]
y_nov = novbitcoindf['ret']#.iloc[:-until]#keras.utils.to_categorical(df['ret'].iloc[:-100])

# x_test =newretdf[['pos', 'neg', 'num_topics', 'num_responses']]#.iloc[-until:]
# y_test = newretdf['ret']#.iloc[-until:]#keras.utils.to_categorical(df['ret'].iloc[-100:])
#%%
y_nov = np.array(y_nov)
y_nov = y_nov.reshape(y_nov.shape[0], 1)
#%%
#%%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# scaler.fit(np.array(newretdf['ret']).reshape(len(newretdf), 1))

y_nov = scaler.fit_transform(y_nov)
# y_test = scaler.transform(y_test)
#%%
# x_train = np.array([np.array(f).reshape(np.array(f).shape[0],  1) for f in x_train])
# x_test = np.array([np.array(f).reshape(np.array(f).shape[0], 1) for f in x_test])

x_nov = np.array(x_nov)

x_nov = np.array(x_nov).reshape(x_nov.shape[0], x_nov.shape[1])

#%%
nov_preds = dnn.predict(np.concatenate((x_test, x_nov)))
#%%

plt.plot(nov_preds - 2)
plt.plot(np.concatenate((y_test, y_nov)))
#%%


ones_y = [1 if f > 0 else 0 for f in np.concatenate((y_test, y_nov))]
ones_pred = [1 if f > 0 else 0 for f in nov_preds-2]
#%%
from sklearn.metrics import accuracy_score, roc_auc_score


roc_auc_score(ones_y, ones_pred), accuracy_score(ones_y, ones_pred)
# # Purge the weird number entries.
# df = df[~(df.date == df.text)]#.iloc[:1000]
# df = df[~(df.text == "")]
# #%%

# tic = time.time()
# df.date = df.date.apply(pd.to_datetime)


#%%
# df.to_csv("data/bitcoin_nov_threads.csv", index=False)

#%%


#%%


#%%


#%%


#%%


#%%

import keras.backend as K
K.clear_session()
#%%



#%%



#%%



#%%



#%%


