#%%
# Imports
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")
from datasets import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import re
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"



#%%
# Load dataset
df = pd.read_csv("formality dataset/emails.csv")
formal = df["formal_email"].dropna().reset_index(drop=True).to_list()
informal = df["informal_email"].dropna().reset_index(drop=True).to_list()


#%%
# Train classification model
stemmer = PorterStemmer()

def stem_text(x):
    return " ".join(stemmer.stem(w) for w in x.split())

formal_stem = [stem_text(x) for x in formal]
informal_stem = [stem_text(x) for x in informal]

X = formal_stem + informal_stem
y = [1]*len(formal_stem) + [0]*len(informal_stem)

vectorizer = CountVectorizer(min_df=2)
X_vec = vectorizer.fit_transform(X)

clf = LogisticRegression(max_iter=2000)
clf.fit(X_vec, y)


#%%
# Extract style terms
feature_names = np.array(vectorizer.get_feature_names_out())
coefs = clf.coef_[0]

formal_terms   = set(feature_names[coefs > 0.001])
informal_terms = set(feature_names[coefs < -0.2])


#%%
# POS masking function
def pos_mask(sentence, style_terms):
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)

    out = []
    for token, pos in tagged:
        stemmed = stemmer.stem(token.lower())
        if stemmed in style_terms:
            out.append(f"<pos_{pos}>")
        else:
            out.append(token)
    return " ".join(out)


#%%
# Build training pairs
train_pairs = []

for x_inf, x_for in zip(informal, formal):
    x_masked = pos_mask(x_inf, informal_terms)
    train_pairs.append({
        "input":  f"formalize: {x_masked}",
        "target": x_for.strip()
    })

for x_for, x_inf in zip(formal, informal):
    x_masked = pos_mask(x_for, formal_terms)
    train_pairs.append({
        "input":  f"informalize: {x_masked}",
        "target": x_inf.strip()
    })


#%%
# Convert into HF Dataset
train_dataset = Dataset.from_list(train_pairs)
train_dataset = train_dataset.shuffle()


#%%
# Add POS tokens to tokenizer and model
pos_tags = [
    "NN","NNS","NNP","NNPS",
    "VB","VBD","VBG","VBN","VBP","VBZ",
    "JJ","JJR","JJS",
    "RB","RBR","RBS",
    "PRP","PRP$","WP","WP$",
    "DT","PDT","WDT",
    "IN","CC","TO",
    "MD","CD","POS","UH",
    "RP","EX","FW","LS","SYM",
    "WRB"
]

special_tokens = [f"<pos_{t}>" for t in pos_tags]
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
tokenizer.add_tokens(special_tokens)

model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
model.resize_token_embeddings(len(tokenizer))


#%%
# Preprocess function
max_len = 256

def preprocess(example):
    encoded = tokenizer(
        example["input"],
        padding="max_length",
        truncation=True,
        max_length=max_len
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["target"],
            padding="max_length",
            truncation=True,
            max_length=max_len
        )["input_ids"]

    # replace PAD tokens with -100 for loss
    labels = [l if l != tokenizer.pad_token_id else -100 for l in labels]

    encoded["labels"] = labels
    return encoded

train_dataset = train_dataset.map(preprocess, batched=False)


#%%
# Trainer
model.to("cuda")
print(next(model.parameters()).device)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

training_args = Seq2SeqTrainingArguments(
    output_dir="./t5_small_formality_model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=1e-4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=50,
    save_strategy="epoch",
    fp16=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)


#%%
# Train T5 model
trainer.train()


#%%
# Generate function
def generate_candidates(text, task="formalize"):
    prompt = f"{task}: {text}"

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to("cuda")

    outputs = model.generate(
        **encoded,
        num_return_sequences=5,          # generate 5 variants
        num_beams=1,                     # disable beam search
        do_sample=True,                  # enable sampling
        top_k=50,                        # k-sampling
        top_p=0.95,                      # nucleus sampling
        max_length=256
    )

    return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]


#%%
# Demo
sample_informal = "The man was highly skilled in CAD"
result = generate_candidates(sample_informal, "formalize")
print(f"Original: {sample_informal}")
print(f"Formalized:")
for r in result:
    print(f"- {r}")


#%%
# Save model
save_dir = "./t5_small_formality_model_final"

model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print("Model saved to:", save_dir)
