from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
import torch

configuration = RobertaConfig()
model = RobertaModel(configuration)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base")

inputs = "Hello, my dog is cute"
encoded = tokenizer(inputs, return_tensors="pt")
# print(f"Encoded: {encoded['input_ids']}")
# decoded = tokenizer.decode(encoded["input_ids"])
# print(f"Decoded: {decoded}")
outputs = model(**encoded)

last_hidden_states = outputs.last_hidden_state

print("Model")
print(model)
print("=====")
print(last_hidden_states.shape)
