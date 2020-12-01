
#%%
# 
import pandas as pd
import numpy as np
import time 
from nltk.tokenize import word_tokenize, TweetTokenizer


tweet_tokenizer = TweetTokenizer()

#%%

d1 = pd.read_csv("data/bitcoin_threads_withdiff.csv")
# d2 = pd.read_csv("data/dogecoin_threads_withdiff.csv")
# d3 = pd.read_csv("data/steemcoin_threads_withdiff.csv")
# d4 = pd.read_csv("data/wavecoin_threads_withdiff.csv")

#%%
# df = pd.concat((d1, d2, d3, d4)).sort_values("date").reset_index(drop=True)
df =d1
df.to_csv("data/fulldf_with_diff_bitcoin.csv", index=False)
#%%

# Parsing:
'''
Each day will be its own entry in the final df.
Each day will have one giant string with the topics and text, and
a diff value associated with the day.

1. Filter df by date
    a. filter by topic. To a string, append topic and all the text.
        i. Remove the stopwords and punctuation from text.
    
    b. Then, to an array, append "date", "string", "diff"

'''
df = pd.read_csv('data/fulldf_with_diff.csv')

#%%
# Filtering out by lengths
#lengths = [len(str(df.iloc[f].text)) for f in range(len(df))]
lengths_col = df.text.apply(lambda x: len(str(x)))

df['lengths'] = lengths_col

#%%

df = df[(df.lengths > 50) & (df.lengths < 350)]

#%%
df.to_csv("data/full_df_trimmeddown.csv", index=False)
#%%

from nltk.corpus import stopwords
import nltk
words = set(nltk.corpus.words.words())
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

    
    text = set([word.lower() for word in text])
    text = arr_to_str(list(text.difference(stopset)))
    # text = arr_to_str(list(text))
    return text

def concat_to_str(ser, max_len = 800):
    big_str = ""
    ser = ser.apply(tweet_tokenizer.tokenize)
    ser = ser.apply(purge_stopwords)
    
    list_of_str = []

    for f in range(len(ser)):
        if len(big_str + f" {ser.iloc[f]} ") >= max_len:
            list_of_str.append(big_str)
            big_str = ""
        
        big_str = big_str + f" {ser.iloc[f]} "

    return list_of_str 
#%%
#Tokenize and purge topics

tic = time.time()
df.topic = df.topic.apply(tweet_tokenizer.tokenize)
df.topic = df.topic.apply(purge_stopwords)
toc = time.time()
print(f"Job took {toc-tic} seconds")

#%%
dates = df.date.unique()[:]

column_names = ['date', 'text', 'diff']
pd_array = []

df.text = df.text.fillna("None")

max_len = 200

tic = time.time()
for date in dates:
    a = time.time()
    print(f"On date: {date}")

    fd = df[df.date == date]
    diffs = fd['diff'].unique()
    
    for diff in diffs:
        fd1 = fd[fd['diff'] == diff]
        topics = fd1.topic.unique()
        
        big_str = ""
        for topic in topics:
            filt = fd1[(fd.topic == topic)]
            # topic = purge_stopwords(tweet_tokenizer.tokenize(topic))

            big_str = big_str + f" {topic} "
            content = concat_to_str(filt.text, max_len)
            
            # big_str = big_str + f" {content}"
            for str_ in content:
                to_add = big_str + f" {str_}"

                pd_array.append([date, to_add, diff])
                break
            # to_add = big_str + f" {content[0]}"
            # pd_array.append([date, to_add, diff])

    b = time.time()
    print(f"\tRow took {b-a} seconds")
toc = time.time()
print(f"Job took {toc-tic} seconds")

df_big = pd.DataFrame(pd_array, columns=column_names)

#%%
df_big.to_csv("data/final_df_200len.csv", index=False)

#%%
df_big.iloc[300000:].to_csv("data/final_df_100len_small.csv", index=False)

#%%


#%%


#%%


#%%


#%%

