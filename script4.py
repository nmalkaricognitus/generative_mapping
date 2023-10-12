from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from transformers import AutoTokenizer
import transformers
import torch
from transformers import pipeline
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
import langchain
from langchain.sql_database import SQLDatabase
from langchain import HuggingFacePipeline
from huggingface_hub import hf_hub_download
from huggingface_hub import login
login()
from langchain_experimental.sql import SQLDatabaseChain

db_user = "cognitus"
db_password = "student"
db_host = "localhost"
db_name = "generative_mapping"
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")

model = "meta-llama/Llama-2-7b-chat-hf" 

tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=True)

llama_pipeline = pipeline(
    "text-generation",  # LLM task
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

#Create the SQL database chain
db_chain = SQLDatabaseChain.from_orm(llama_pipeline,db,verbose=True,resturn_sql=False,use_query_checker=True)

# Run the query
result = db_chain.run("How many tables are there in the database?")

# Print the result
print(result)

# toolkit = SQLDatabaseChain.from_orm(db=db, llm=llama_pipeline)

# agent_executor = create_sql_agent(
#         llm=llama_pipeline,
#         toolkit=toolkit,
#         verbose=True)
# bot_response = agent_executor.run("How many tables are there")

# print(bot_response)