#%%
# Imports
import pandas as pd



#%%
# Load original dataset
df = pd.read_csv("crows-pairs data/original.csv")
df = df[["sent_more","sent_less"]].dropna().reset_index(drop=True)

#%%
# Function to apply QA format to a sentence
def apply_qa_format(row):
    return pd.Series({
        "sent_more_qa": f"Know what? {row['sent_more']}",
        "sent_less_qa": f"Know what? {row['sent_less']}"
    })

#%%
# Apply the prefix transformation
df_pref = df.apply(apply_qa_format, axis=1)
print(df_pref.head())
df_pref.to_csv(f"crows-pairs data/qa.csv")