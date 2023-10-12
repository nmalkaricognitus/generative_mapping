import langchain
from langchain.sql_database import SQLDatabase
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from huggingface_hub import login
login()
import transformers
from transformers import pipeline
import torch
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit

model = "meta-llama/Llama-2-7b-chat-hf" 

tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=True)

pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device=-1,
    max_length=10000,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)

llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': 0.7})

# Create a SQL database object
db_user = "cognitus"
db_password = "student"
db_host = "localhost"
db_name = "generative_mapping"
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")


toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True)
bot_response = agent_executor.run("How many tables are there in the database")

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
