#%%
# Imports
import pandas as pd



#%%
# Load original dataset
df = pd.read_csv("crows-pairs data/original.csv")
df = df[["sent_more","sent_less"]].dropna().reset_index(drop=True)

#%%
# Function to add evidential marker to a sentence
def apply_evidential_marker(row):
    return pd.Series({
        "sent_more_evidential": f"Evidently {row['sent_more'][0].lower()}{row['sent_more'][1:]}",
        "sent_less_evidential": f"Evidently {row['sent_less'][0].lower()}{row['sent_less'][1:]}"
    })

#%%
# Apply the prefix transformation
df_pref = df.apply(apply_evidential_marker, axis=1)
print(df_pref.head())
df_pref.to_csv(f"crows-pairs data/evidential.csv")
