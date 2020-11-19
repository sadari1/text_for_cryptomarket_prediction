#%%
import numpy as np
import pandas as pd
import time

# %%
df = pd.read_json("coinmarkcap_scrape/steemcoin_prices.json")

# %%
df_sorted = df.sort_values('date')

# %%

replace_comma = lambda x: x.replace(',', '') if type(x) == str else x

df_sorted.close = df_sorted.close.apply(replace_comma)
# %%
df_sorted_shifted = df_sorted.shift(1, fill_value=0)

#%%

diffs = np.array(df_sorted.close, dtype=np.float32) - np.array(df_sorted_shifted.close, dtype=np.float32)
# %%

diffs = [1 if diff > 0 else 0 for diff in diffs]
# %%

df_sorted['diff'] = diffs


# %%
df_sorted.to_csv("data/steemcoin_price_diffs.csv", index=False)
# %%
