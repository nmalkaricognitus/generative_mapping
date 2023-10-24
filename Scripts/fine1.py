import transformers
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the LLAMA 2 model and tokenizer
model = "meta-llama/Llama-2-7b-chat-hf" 

tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=True)

# Load the SQL data in JSON format
with open("/home/ubuntu/generative_mapping/generative_mapping/Data/sql_create_context_v4.json", "r") as f:
    sql_data = json.load(f)

# Prepare the training data
training_data = []
for sql_query in sql_data:
    # Split the SQL query into input and output sequences
    input_sequence = tokenizer(sql_query["question"])["input_ids"]
    output_sequence = tokenizer(sql_query["answer"], return_tensors="pt")["input_ids"]


    # Add the input and output sequences to the training data
    training_data.append((input_sequence, output_sequence))

# Finetune the LLAMA 2 model on the SQL data
model.train()
for epoch in range(10):
    for input_sequence, output_sequence in training_data:
        # Calculate the loss
        loss = model(input_ids=input_sequence, labels=output_sequence)

        # Update the model parameters
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

# Save the fine-tuned LLAMA 2 model
model.save_pretrained("finetuned_llama_2_sql_json")
