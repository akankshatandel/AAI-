# PRACTICAL 3: Simple chatbot using a pre-trained GPT-2 model
# Install: python -m pip install transformers torch
# First run downloads the model, so an internet connection is required.

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def chatbot():
    model_name = "gpt2"
    print("Loading GPT-2 model. The first run may take a while...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    print("Chatbot ready. Type 'exit' to quit.")
    while True:
        user_text = input("You: ").strip()
        if user_text.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        if not user_text:
            continue

        inputs = tokenizer(user_text, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(
                inputs["input_ids"],
                max_new_tokens=60,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        full_text = tokenizer.decode(output[0], skip_special_tokens=True)
        answer = full_text[len(user_text):].strip()
        print("Chatbot:", answer if answer else full_text)

if __name__ == "__main__":
    chatbot()
