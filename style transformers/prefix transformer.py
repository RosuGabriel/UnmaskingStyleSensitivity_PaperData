#%%
# Imports
import pandas as pd



#%%
# Prefixes to be used for text transformation
prefixes = ["It's important to acknowledge that:",
            "It's worth noting that:",
            "With that in mind:"
            ]

#%%
# Load original dataset
df = pd.read_csv("crows-pairs data/original.csv")
df = df[["sent_more","sent_less"]].dropna().reset_index(drop=True)

#%%
# Function to add prefix to a sentence
def apply_prefix(row, prefixIndex):
    return pd.Series({
        "sent_more_prefixed": f"{prefixes[prefixIndex]} {row['sent_more']}",
        "sent_less_prefixed": f"{prefixes[prefixIndex]} {row['sent_less']}"
    })

#%%
# Apply the prefix transformation
for i in range(len(prefixes)):
    df_pref = df.apply(apply_prefix, axis=1, prefixIndex=i)
    print(df_pref.head())
    print("...........")
    df_pref.to_csv(f"crows-pairs data/prefix{i}.csv")
