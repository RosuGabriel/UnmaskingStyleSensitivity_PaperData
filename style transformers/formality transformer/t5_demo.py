#%%
# Load model
from transformers import T5Tokenizer, T5ForConditionalGeneration

save_dir = "./t5_small_formality_model_final"

tokenizer = T5Tokenizer.from_pretrained(save_dir)
model = T5ForConditionalGeneration.from_pretrained(save_dir).to("cuda")


#%%
# Formalize
input_text = "formalize: The man was highly skilled in CAD."
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
print(input_text)
for i in range(5):
    outputs = model.generate(
        **inputs,
        max_length=128,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    print(f"output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

#%%
# Informalize
input_text = "informalize: It is a pleasure to meet you."
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
print(input_text)
for i in range(5):
    outputs = model.generate(
        **inputs,
        max_length=128,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    print(f"output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
