import accelerate
import transformers
import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AdamW

# Load the LLAMA 2 model and tokenizer
model = transformers.AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

# Create an accelerator
accelerator = accelerate.Accelerator()

# Initialize the optimizer
optimizer = AdamW(model.parameters())

# Load the SQL data in JSON format
with open("/home/ubuntu/generative_mapping/generative_mapping/Data/sql_create_context_v4.json", "r") as f:
    sql_data = json.load(f)

# Prepare the training data
training_data = []
for sql_query in sql_data:
    # Split the SQL query into input and output sequences
    input_sequence = tokenizer(sql_query["question"])["input_ids"]
    output_sequence = tokenizer(sql_query["answer"], return_tensors="pt")["input_ids"]

    input_sequence = torch.tensor(input_sequence)
    output_sequence = torch.tensor(output_sequence)

    input_sequence = input_sequence.unsqueeze(0)

    # Add the input and output sequences to the training data
    training_data.append((input_sequence, output_sequence))

# Finetune the LLAMA 2 model on the SQL data
model.train()
for epoch in range(10):
    for input_sequence, output_sequence in training_data:
        # Move the input and output sequences to the accelerator
        input_sequence = accelerator.prepare_input(input_sequence)
        output_sequence = accelerator.prepare_input(output_sequence)

        # Calculate the loss
        model_output = model(input_ids=input_sequence)
        loss = model_output.loss

        # Update the model parameters
        accelerator.backward(loss)
        optimizer.step()

# Save the fine-tuned LLAMA 2 model
accelerator.wait_for_everyone()
model.save_pretrained("finetuned_llama_2_sql_json", optimizer=optimizer)
