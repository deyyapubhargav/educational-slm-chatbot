import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

st.set_page_config(page_title="Educational SLM Chatbot")


@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    return tokenizer, model


tokenizer, model = load_model()


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


st.title("📚 Educational SLM Chatbot")
st.write("Ask questions related to AI / ML basics")

question = st.text_input("Enter your question")

if question:
    answer = chat(question)
    st.text_area("Answer", value=answer, height=150)
