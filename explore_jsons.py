#%%
import numpy as np
import pandas as pd
import time

# %%
df = pd.read_json("coinmarkcap_scrape/bitcoin_megathread.json")
# 
# %%
# Purge the weird number entries.
df = df[~(df.date == df.text)].iloc[:1000]
df = df[~(df.text == "")]
#%%

tic = time.time()
df.date = df.date.apply(pd.to_datetime)

toc = time.time()
print(f"Job took {toc-tic} seconds")
#%%

lengther = lambda x: len(x)
df["lengths"] = list(map(lengther, df.text))

#%%
df2 = df[(df.lengths > 4) & (df.lengths < 300)]
#%%

#%%