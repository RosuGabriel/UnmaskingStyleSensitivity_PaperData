#%%
# Imports
import pandas as pd



#%%
# Load original dataset
df = pd.read_csv("crows-pairs data/original.csv")
df = df[["sent_more","sent_less"]].dropna().reset_index(drop=True)

#%%
# Function to add punctuation to a sentence
def replace_punctuation_mark(row):
    def fix(s):
        s = s.strip()
        if s.endswith(("!")):
            return s
        elif s.endswith("."):
            return s[:-1] + "!"
        else:
            return s + "!"
    
    return pd.Series({
        "sent_more_punct": fix(row["sent_more"]),
        "sent_less_punct": fix(row["sent_less"])
    })


#%%
# Apply the punctuation transformation
df_pref = df.apply(replace_punctuation_mark, axis=1)
print(df_pref.head())
df_pref.to_csv(f"crows-pairs data/punctuation.csv")
