from flask import Flask, render_template, request
import os
from langchain.agents import *
from langchain.sql_database import SQLDatabase
import creds
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain


db_user = "cognitus"
db_password = "student"
db_host = "localhost"
db_name = "generative_mapping"
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")


model_path = "openlm-research/open_llama_7b"

# Create the tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map='auto',offload_folder="/Users/nikhilmalkari/Documents/Cognitus/Generative Mapping/off_folder",
)

# Create the pipeline
pipeline = pipeline(
    "text-generation", #task
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=1000,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

# Create the LLM
llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

# Create the toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Create the SQL database chain
db_chain = SQLDatabaseChain.from_llm(llm,db,verbose=True,resturn_sql=False,use_query_checker=True)

# Run the query
result = db_chain.run("How many tables are there in the database?")

# Print the result
print(result)

# def create_sql_agent(llm, toolkit, verbose=True):
#     # Implement your SQL agent creation logic here
#     # This function should return an agent that can execute SQL queries
#     pass

# def main():
#     # Load your language model (llm) and toolkit here
#     # llm = load_language_model()
#     # toolkit = load_toolkit()

#     print("SQL Agent - Interactive Mode")
#     print("Type your SQL query or 'exit' to quit.")

#     agent_executor = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)

#     while True:
#         user_message = input("User: ")
#         if user_message.lower() == 'exit':
#             print("Exiting...")
#             break

#         bot_response = agent_executor.run(user_message)
#         print("Bot:", bot_response)

# if __name__ == "__main__":
#     main()
