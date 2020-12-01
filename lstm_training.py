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

import mlflow
import mlflow.tensorflow
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
from sklearn.metrics import roc_auc_score
#%%

# final df3 is just the bitcoins, df5 is max length 800.
# df = pd.read_csv("data/full_df_trimmeddown.csv")#.tail(25000)
df = pd.read_csv("data/full_df_trimmeddown.csv")#.tail(25000)
#%%
new_arr = np.concatenate((np.array(list(df.sents)), np.array(list(df['diff'])).reshape(-1, 1)), axis=1)
#%%
df2 = pd.DataFrame(new_arr, columns=['neg', 'neu', 'pos', 'comp', 'diff'])

#%%
df3 = pd.read_csv("data/bitcoin_price_diffs.csv")

cols = ['high', 'close', 'open', 'low']

for col in cols:
  try:
    df3[col] = df3[col].apply(lambda x: float(x.replace(',', '')))
  except:
    continue


#%%
# tic = time.time()
# # df.text = df.text.apply(lambda x: x[:400])
# # df.text = df.text.apply(lambda x: model.encode(x, convert_to_tensor=False))
# encodings = model.encode(list(df['text']), convert_to_tensor=False)
# toc = time.time()
# print(f'Job took {toc-tic} seconds.')

# #%%

# df['encoded'] = list(encodings)
df['diff'] = df['diff'].apply(lambda x: int(x))

#%%

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

#%%

tic = time.time()
# sentiments = df.text.apply(sid.polarity_scores)
# sid.polarity_scores(df.text.iloc[10])
sentiments = []
for f in range(len(df)):
  print(f"On index {f} / {len(df)}")
  # neg, neutral, pos, compound (all added together)
  timea = time.time()
  vec = sid.polarity_scores(df.text.iloc[f]).values()
  timeb = time.time()
  print(f"\tTook {timeb-timea} seconds.")

  sentiments.append(vec)

toc = time.time()
print(f"Job took {toc-tic} seconds")

#%%
df['sents'] = sentiments
df['sents'] = df['sents'].apply(lambda x: list(x))
#%%

df.to_csv("data/fulldf5_with_sentiments.csv", index=False)

#%%
df = pd.read_csv("data/fulldf5_with_sentiments.csv")
#%%
# from nltk.corpus import stopwords
# import nltk
# words = set(nltk.corpus.words.words())
punctuation_list = ['.', ':', '!', '?', ',', '^', '(', ')', '。', '、', "'", ":/", '-', '/', '&', ';', '$', '*', '+', '\\', '_', '`', '"', '=', '[', ']']
purge_set = ['in', 'a', 'I', 'you', 'he', 'she', 'the', 'is', 'and', 'it', 'be', 'that', 'they']
stopset = set(punctuation_list)# + purge_set)
def arr_to_str(array):
    _ = ""
    for f in array:
        _ = _ + f"{f} "
    _ = _[:-1]

    return _

def purge_stopwords(text):
    
    text = set([word.lower() for word in text.split(" ")])
    text = arr_to_str(list(text.difference(stopset)))
    return text

#%%
from ast import literal_eval
#%%
df.sents = df.sents.apply(literal_eval)
#%%
# Get counts of each


#%%


#%%
# Get the average of each column.

dates = df.date.unique()

tic = time.time()
dataset_x = []
dataset_y = []
for date in dates:
  print(f"On date: {date}")
  filt = df[df.date == date]
  filt.sents = filt.sents.apply(np.array)

  averaged = np.mean(filt.sents, axis=0)
  dataset_x.append(np.array(averaged))
  dataset_y.append(np.array(filt['diff'].iloc[0], dtype=np.int32))

toc = time.time()
print(f"Job took {toc-tic} seconds.")

#%%
dataset_y = np.array(dataset_y)
dataset_x = np.array(dataset_x)
dataset_y = keras.utils.to_categorical(dataset_y)
#%%
np.save('dataset_x.npy', dataset_x)
np.save('dataset_y.npy', dataset_y)
#%%
x_train = dataset_x[:-100]
y_train = dataset_y[:-100]

x_test = dataset_x[-100:]
y_test = dataset_y[-100:]
# x_train = np.array(df['sents'].iloc[:-100])
# y_train = np.array(df['diff'].iloc[:-100])

# x_test = np.array(df['sents'].iloc[-100:])
# y_test = np.array(df['diff'].iloc[-100:])
#%%
# y_train = tf.keras.utils.to_categorical(y_train)
# y_test = tf.keras.utils.to_categorical(y_test)

#%%
# x_train = np.array([np.array(f).reshape(np.array(f).shape[0], 1) for f in x_train])
# x_test = np.array([np.array(f).reshape(np.array(f).shape[0], 1) for f in x_test])
#%%

# np.save("x_train_sents.npy", x_train)
# np.save('y_train_sents.npy', y_train)
# np.save('x_test_sents.npy', x_test)
# np.save('y_test_sents.npy', y_test)
#%%
x_train = df.sents.iloc[:-100]
y_train = keras.utils.to_categorical(df['diff'].iloc[:-100])

x_test = df.sents.iloc[-100:]
y_test = keras.utils.to_categorical(df['diff'].iloc[-100:])

#%%
x_train = np.array([np.array(f).reshape(np.array(f).shape[0], 1) for f in x_train])
x_test = np.array([np.array(f).reshape(np.array(f).shape[0], 1) for f in x_test])

#%%

from tensorflow.keras.layers import Conv1D
from tensorflow.keras.regularizers import l1_l2

batch_size = 64
# vocab_size = len(vocab)
input_layer = Input( shape=(x_train.shape[1], 1))
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

# h1 = Dense(4)(input_layer)
# h2 = Dense(4)(h1)

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

# dense = Dense(2, activation='softmax')(flat)
# dense=  Dense(2, activation='softmax')(h2)
# dnn = Model(input_layer, dense)

# print(dnn.summary())

# vocab_size = len(vocab)
# input_layer = Input( shape=( 38, 1222))
# # emb = Embedding(vocab_size+1, 200)(input_layer)
# lstm = Bidirectional(LSTM(200,  return_sequences=True, dropout=0.2))(input_layer, training = True)
# lstm = Bidirectional(LSTM(200,  return_sequences=True,dropout=0.2))(lstm, training = True)
# lstm = Bidirectional(LSTM(200,  return_sequences=True,dropout=0.2))(lstm, training = True)
# lstm = Dense(1222, activation='softmax')(lstm)

# model = Model(input_layer, lstm)

#%%


dnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=[ 'accuracy'])

#%%
from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('dnn3.hdf5', monitor='accuracy', verbose=1, save_best_only=True, mode='max')
batch_size=64
epochs=100

dnn.fit(x=x_train, y=y_train,
        batch_size=batch_size, epochs=epochs,
        verbose=1, callbacks=[checkpoint])


#%%
# model = keras.models.load_model("lstm_model2.hdf5")

#%%

preds = dnn.predict(x_test)

auc = roc_auc_score(np.argmax(y_test,axis=1), np.argmax(preds,axis=1))
print(f"AUC: {auc}")


#%%

## Code sourced from https://medium.com/@kaijuneer/fine-tuning-tensorflow-bert-model-for-sentiment-analysis-1119fd6bef49


def convert_data_to_examples(train, test, DATA_COLUMN, LABEL_COLUMN): 
  train_InputExamples = train.apply(lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this case
                                                          text_a = x[DATA_COLUMN], 
                                                          text_b = None,
                                                          label = x[LABEL_COLUMN]), axis = 1)

  validation_InputExamples = test.apply(lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this case
                                                          text_a = x[DATA_COLUMN], 
                                                          text_b = None,
                                                          label = x[LABEL_COLUMN]), axis = 1)
  
  return train_InputExamples, validation_InputExamples

def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    features = [] # -> will hold InputFeatures to be converted later

    for e in examples:
        # Documentation is really strong for this method, so please take a look at it
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,
            max_length=max_length, # truncates if len(s) > max_length
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True, # pads to the right by default
            truncation=True
        )

        input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],
            input_dict["token_type_ids"], input_dict['attention_mask'])

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.label
            )
        )

    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )

#%%
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

#%%
DATA_COLUMN = 'text'
LABEL_COLUMN = 'diff'
BATCH_SIZE = 16

train = df.iloc[:-100]#df.iloc[:-100]
test = df.iloc[-100:]#df.iloc[-100:]

tic = time.time()
print(tic)
# train and test is your dataset
train_InputExamples, validation_InputExamples = convert_data_to_examples(train, test, DATA_COLUMN, LABEL_COLUMN)

train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
train_data = train_data.shuffle(100).batch(BATCH_SIZE).repeat(2)

validation_data = convert_examples_to_tf_dataset(list(validation_InputExamples), tokenizer)
validation_data = validation_data.batch(BATCH_SIZE)

toc=time.time()
print(f"Job took {toc-tic} seconds.")

#%%
mlflow.set_experiment("bert-tune")

with mlflow.start_run():
  
  mlflow.tensorflow.autolog()
  
  optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-08, clipnorm=1.0)
  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

  model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
  
  tic=  time.time()
  model.fit(train_data, epochs=40, verbose=1, validation_data=validation_data)

  toc=time.time()
  print(f"Job took {toc-tic} seconds.")

#%%

test_ = test.iloc[:100].text#list(np.array(test.text))
tf_batch = tokenizer(list(test_), max_length=128, padding=True, truncation=True, return_tensors='tf')
tf_outputs = model(tf_batch)
tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
labels = [0,1]
label = tf.argmax(tf_predictions, axis=1)
label = label.numpy()
preds = []
y_true = []
for i in range(len(test_)):
  y_true.append(int(test['diff'].iloc[i]))
  preds.append(label[i])
  print(test['diff'].iloc[i], ": ", labels[label[i]])

auc = roc_auc_score(y_true, preds)
print(f"AUC: {auc}")
 #%%

roc_auc_score(y_true, preds)
# %%

#%%

#%%
import keras.backend as K
K.clear_session()

#%%
x = df['text']
y = df['diff']

#%%
y = y.apply(lambda x: f"{int(x)}")

#%%

x_orig = np.array(x)
y_orig = np.array(y)
#%%
x = np.array(x.iloc[:-100])
y = np.array(y.iloc[:-100])
#%%

tic = time.time()

tokenizer_x = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (query for query in x), target_vocab_size=2**20)

tokenizer_y = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (query for query in y), target_vocab_size=2**10)

toc = time.time()

print(f"Job took {toc-tic} seconds.")

#%%


sample_string = 'Hello world!'

tokenized_string = tokenizer_x.encode(sample_string)
print ('Tokenized string is {}'.format(tokenized_string))

original_string = tokenizer_x.decode(tokenized_string)
print ('The original string: {}'.format(original_string))



#%%
for ts in tokenized_string:
  print ('{} ----> {}'.format(ts, tokenizer_x.decode([ts])))

#%%
BUFFER_SIZE = 20000
BATCH_SIZE = 32

def encode(lang1, lang2):
  lang1 = [tokenizer_x.vocab_size] + tokenizer_x.encode(
      lang1.numpy()) + [tokenizer_x.vocab_size+1]

  lang2 = [tokenizer_y.vocab_size] + tokenizer_y.encode(
      lang2.numpy()) + [tokenizer_y.vocab_size+1]

  return lang1, lang2

#%%
def tf_encode(q, r):
  result_q, result_r = tf.py_function(encode, [q, r], [tf.int64, tf.int64])
  result_q.set_shape([None])
  result_r.set_shape([None])

  return result_q, result_r

#%%
MAX_LENGTH = 1098

tic = time.time()
def filter_max_length(x, y, max_length=MAX_LENGTH):
  return tf.logical_and(tf.size(x) <= max_length,
                        tf.size(y) <= max_length)

train_examples = tf.data.Dataset.from_tensor_slices((x, y))
train_dataset = train_examples.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)
# cache the dataset to memory to get a speedup while reading from it.
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
toc=time.time()

print(f"Job took {toc-tic} seconds.")
#%%
# pt_batch, en_batch = next(iter(train_dataset))
# pt_batch, en_batch

#%%
num_layers = 4
d_model = 128
dff = 128
num_heads = 4

input_vocab_size = tokenizer_x.vocab_size + 2
target_vocab_size = tokenizer_y.vocab_size + 2
dropout_rate = 0.1

#%%
learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)



#%%
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

#%%
transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size, 
                          pe_input=input_vocab_size, 
                          pe_target=target_vocab_size,
                          rate=dropout_rate)


#%%

name_ = 'test4'
checkpoint_path = f"./checkpoints/train/{name_}"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')



#%%
EPOCHS = 10

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]

  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

  with tf.GradientTape() as tape:
    predictions, _ = transformer(inp, tar_inp, 
                                 True, 
                                 enc_padding_mask, 
                                 combined_mask, 
                                 dec_padding_mask)
    loss = loss_function(tar_real, predictions)

  gradients = tape.gradient(loss, transformer.trainable_variables)    
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  train_loss(loss)
  train_accuracy(tar_real, predictions)

#%%

#%%
def evaluate(inp_sentence):
  start_token = [tokenizer_x.vocab_size]
  end_token = [tokenizer_x.vocab_size + 1]

  # inp sentence is portuguese, hence adding the start and end token
  inp_sentence = start_token + tokenizer_x.encode(inp_sentence) + end_token
  encoder_input = tf.expand_dims(inp_sentence, 0)

  # as the target is english, the first word to the transformer should be the
  # english start token.
  decoder_input = [tokenizer_y.vocab_size]
  output = tf.expand_dims(decoder_input, 0)

  for i in range(MAX_LENGTH):
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        encoder_input, output)

    # predictions.shape == (batch_size, seq_len, vocab_size)
    predictions, attention_weights = transformer(encoder_input, 
                                                 output,
                                                 False,
                                                 enc_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask)

    # select the last word from the seq_len dimension
    predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # return the result if the predicted_id is equal to the end token
    if predicted_id == tokenizer_y.vocab_size+1:
      return tf.squeeze(output, axis=0), attention_weights

    # concatentate the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0), attention_weights


#%%

def process_sent(sentence):
  sentence = pd.Series([sentence])
  sentence = sentence.apply(purge_stopwords)
  # sentence = sentence.apply(arr_to_str)
  return sentence.iloc[0]

def translate(sentence, plot=''):
  sentence = process_sent(sentence)

  result, attention_weights = evaluate(sentence)

  predicted_sentence = tokenizer_y.decode([i for i in result 
                                            if i < tokenizer_y.vocab_size])  

  # print('Input: {}'.format(sentence))
  print('\nPredicted: {}'.format(predicted_sentence))
  
  return predicted_sentence

#%%

mlflow.set_experiment("transformer")

with mlflow.start_run():
  
  mlflow.tensorflow.autolog()

  mlflow.tensorflow.mlflow.log_param("batch_size", BATCH_SIZE)
  mlflow.tensorflow.mlflow.log_param("num_layers", num_layers)
  mlflow.tensorflow.mlflow.log_param("d_model", d_model)
  mlflow.tensorflow.mlflow.log_param("dff", dff)
  mlflow.tensorflow.mlflow.log_param("num_heads", num_heads)
    
  for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    # inp -> portuguese, tar -> english
    for (batch, (inp, tar)) in enumerate(train_dataset):
      # print(tf.shape(inp), tf.shape(tar))
      # print(tokenizer_y.decode(tar[0]), tokenizer_y.decode(tar[0]))
      train_step(inp, tar)

      if batch % 1 == 0:
        print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
            epoch + 1, batch, train_loss.result(), train_accuracy.result()))

    if (epoch + 1) % 5 == 0:
      ckpt_save_path = ckpt_manager.save()
      print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                           ckpt_save_path))

    print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                  train_loss.result(), 
                                                  train_accuracy.result()))

    print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

  transformer.save_weights(f'./checkpoints/{name_}.ckpt')

  preds = []
  for i in range(-100, 0):
    preds.append(translate(x_orig[i]))
    print(f"True: {y_orig[i]}\n")


  y_true = np.array(y_orig[-100:])

  auc  = roc_auc_score(y_true, np.array([float(f) for f in preds]))

  print(f"AUC: {auc}")
  mlflow.tensorflow.mlflow.log_metric("auc", auc)

  mlflow.tensorflow.log_model(artifact_path="transformer")

mlflow.end_run()

#   if plot:
#     plot_attention_weights(attention_weights, sentence, result, plot)
#%%

# transformer.save_weights('./checkpoints/full4.ckpt')
preds = []
for i in range(-100, 0):
  preds.append(translate(x_orig[i]))
  print(f"True: {y_orig[i]}\n")


y_true = np.array(y_orig[-100:])

auc  = roc_auc_score(y_true, np.array([float(f) for f in preds]))

print(f"AUC: {auc}")
#%%
# latest_check = tf.train.latest_checkpoint(checkpoint_path)

# transformer.load_weights('./checkpoints/full3.ckpt')

#%%
translate("Don't invest in this it won't go up")
# print ("Real translation: this is a problem we have to solve .")

#%%
translate("Invest in bitcoin I predict it will go up tomorrow")
# print ("Real translation: this is a problem we have to solve .")
# %%

  # print(f"\nReal: {y_orig[i]}\n")
# %%


import keras.backend as K
K.clear_session()
#%%



#%%



#%%



#%%



#%%


