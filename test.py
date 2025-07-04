
import torch
from transformers import pipeline

# from llama_cpp import Llama

# llm = Llama(model_path="./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
#             n_ctx=2048,
#             n_threads=8,
#             n_gpu_layers=35)
# # llm = Llama(model_path="./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", chat_format="llama-2")
# llm.create_chat_completion(
#       messages = [
#         {
#           "role": "system",
#           "content": "You are story writing assistant"

#         },
#         {
#           "role": "user",
#           "content": "Write an extensive story about Life as a young Adult"
#         }
#       ]
# )
# print()

pipe = pipeline("text-generation", model="tinyllama-merged", torch_dtype=torch.bfloat16, device_map="auto")

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot",
    },
    {"role": "user", "content": "What is the main purpose of the OECD Transfer Pricing Guidelines?"},
]
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"])