#%%
import numpy as np
import pandas as pd
import time

# %%
df = pd.read_json("coinmarkcap_scrape/bitcoin_megathread.json")

topics = ['bitcoin_megathread' for f in range(df.shape[0])]
df['topic'] = topics

df1 = pd.read_json("coinmarkcap_scrape/bitcoin_threads.json")
df2 = pd.read_json("coinmarkcap_scrape/bitcoin_threads2.json")
df3 = pd.read_json("coinmarkcap_scrape/bitcoin_threads3.json")
df4 = pd.read_json("coinmarkcap_scrape/bitcoin_threads4.json")

# 
# %%

final_df = pd.concat((df,df1,df2,df3,df4))
# %%

import os

try:
    os.mkdir("data/")
except:
    print('dir already exists')
final_df.to_csv("data/full_bitcoin_thread.csv", index=False)
# %%
