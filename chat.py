from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load trained model
tokenizer = GPT2Tokenizer.from_pretrained("model")
model = GPT2LMHeadModel.from_pretrained("model")
model.eval()


def chat(question):
    prompt = f"Q: {question} A:"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=120,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("A:")[-1].strip()


# ---- THIS PART MUST EXIST ----
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    print("Bot:", chat(user_input))
