import langchain
from langchain.sql_database import SQLDatabase
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from huggingface_hub import login
login()
import transformers
from transformers import pipeline
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
import torch
import sys
sys.setrecursionlimit(100000)


class LLM:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def generate(self, prompt):
        return self.pipeline.generate(prompt)


class LLMPipeline:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text


# Create a SQL database object
db_user = "cognitus"
db_password = "student"
db_host = "localhost"
db_name = "generative_mapping"
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")

# Create an LLM model
model = AutoModelForCausalLM.from_pretrained('learnanything/llama-7b-huggingface')

# Create an LLM tokenizer
tokenizer = AutoTokenizer.from_pretrained('learnanything/llama-7b-huggingface')

# Create an LLM pipeline
llm_pipeline = LLMPipeline(model, tokenizer)

# Create an LLM object
llm = LLM(llm_pipeline)

# Create a SQL agent object
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True)

# Run the bot request
bot_response = agent.run("How many tables are there")

# Print the bot response
print(bot_response)


# # Create the agent executor
# agent_executor = langchain.AgentExecutor(llm=llm, db=db, verbose=True)

# # Run the query
# bot_response = agent_executor.run("How many tables are there in the database?")

# # Print the bot response
# print(bot_response)

# db_chain = SQLDatabase.from_uri(llm, db, verbose=True)

# # Run a SQL query using natural language prompt
# db_chain.run("How many tables are there ?")
