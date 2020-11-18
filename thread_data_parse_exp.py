#%%
import pandas as pd
import numpy as np
import time

#%%
# df = pd.read_csv("data/dogecoin_threads.csv")
df = pd.read_json("coinmarkcap_scrape/steemcoin_threads.json")
dfp = pd.read_csv("data/steemcoin_price_diffs.csv")
#%%

def arr_to_str(arr):
    temp = ""
    for f in arr:
        temp = temp + f"{f} "
    return temp[:-2]

def bitcoin_date_parser(date_str):
    date = date_str.split(" ")[:-2]
    date = arr_to_str(date)
    return pd.to_datetime(date)

def altcoin_date_parser(date_str):
    # Hardcoding the date on this day
    print(date_str)
    date_str = date_str.replace("at", "October 21, 2020,")
    date = date_str.split(" ")[:-2]
    date = arr_to_str(date)
    return pd.to_datetime(date)

def altcoin_datetime_parser(date):
    date = str(date).split(" ")[0]
    return date
#%%

tic = time.time()
if type(df.iloc[0].date) == 'str':
    df.date = df.date.apply(altcoin_date_parser)
else:
    df.date = df.date.apply(altcoin_datetime_parser)
toc = time.time()

print(f"Job took {toc-tic} seconds.")

#%%
df.to_csv("data/steemcoin_threads_datefixed.csv", index=False)
#%%

dfp = dfp[['date', 'diff']]
dfp.date =dfp.date.apply(pd.to_datetime)
#%%

tic = time.time()

diff_finder = lambda x: np.array(dfp[dfp.date == x]['diff'])

diff_col = list(map(diff_finder, df.iloc[:].date))

toc = time.time()

print(f"Job took {toc-tic} seconds.")

# for f in range(len(dfp)):
#     date = dfp.iloc[f].date 
#     diff =dfp.iloc[f].diff

#     filtered = df[df.date == date]
#     filtered_shape = filtered.shape

#%%
def mapper(arr):
    try:
        return arr[0]
    except:
        return np.nan
    
#%%
tic = time.time()

diff_col = np.array(list(map(mapper, diff_col)))
toc = time.time()

print(f"Job took {toc-tic} seconds.")
#%%
df['diff'] = diff_col

df = df[~df['diff'].isna()]

#%%
np.save("steemcoin_diffs.npy", diff_col)
#%%
df.to_csv("data/steemcoin_threads_withdiff.csv", index=False)
#%%

#%%

#%%

#%%

#%%

#%%

#%%


# df2 = pd.read_csv("data/full_bitcoin_thread.csv")#pd.read_json("coinmarkcap_scrape/bitcoin_threads3.json")
# #%%
# df2 = df2[df2.date != df2.text]
# df2 = df2.dropna()
#%%


#%%

#%%

#%%
# d1 = pd.read_json("coinmarkcap_scrape/bitcoin_threads.json")
# d2 = pd.read_json("coinmarkcap_scrape/bitcoin_threads2.json")
# d3 = pd.read_json("coinmarkcap_scrape/bitcoin_threads3.json")
# d4 = pd.read_json("coinmarkcap_scrape/bitcoin_threads4.json")
# # %%


# d1 = d1[d1.date != d1.text]
# d1 = d1.dropna()

# d2 = d2[d2.date != d2.text]
# d2 = d2.dropna()


# d3 = d3[d3.date != d3.text]
# d3 = d3.dropna()


# d4 = d4[d4.date != d4.text]
# d4 = d4.dropna()


# # %%
# bitcoin_threads_df = pd.concat((d1, d2, d3, d4))

# #%%
# bitcoin_threads_df.to_csv("data/bitcoin_smallthreads.csv", index=False)
# # %%
