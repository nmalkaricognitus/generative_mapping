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

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_path = 'gaussalgo/T5-LM-Large-text2sql-spider'
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device=0,
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
        handle_parsing_errors=True,
        verbose=True)
bot_response = agent_executor.run("How many tables are there in the database")

print(bot_response)