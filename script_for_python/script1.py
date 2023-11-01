from transformers import AutoTokenizer
import transformers
import torch

tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
pipeline = transformers.pipeline(
    "text-generation",
    model="codellama/CodeLlama-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
)

system = "Provide answers in Python"
user = "Write a function that computes the set of sums of all contiguous sublists of a given list."

prompt = f"<s><<SYS>>\\n{system}\\n<</SYS>>\\n\\n{user}"

sequences = pipeline(
    prompt ,
    do_sample=True,
    temperature=0.1,
    top_p=0.9,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=400,
)

for seq in sequences:
    print(f"Result: {seq['generated_text']}")