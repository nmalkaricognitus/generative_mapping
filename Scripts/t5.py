import torch
from transformers import BitsAndBytesConfig
from llama_index import HuggingFaceLLM
from llama_index import PromptTemplate

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

def messages_to_prompt(messages):
  prompt = ""
  for message in messages:
    if message.role == 'system':
      prompt += f"<|system|>\n{message.content}</s>\n"
    elif message.role == 'user':
      prompt += f"<|user|>\n{message.content}</s>\n"
    elif message.role == 'assistant':
      prompt += f"<|assistant|>\n{message.content}</s>\n"

  # ensure we start with a system prompt, insert blank if needed
  if not prompt.startswith("<|system|>\n"):
    prompt = "<|system|>\n</s>\n" + prompt

  # add final assistant prompt
  prompt = prompt + "<|assistant|>\n"

  return prompt


llm = HuggingFaceLLM(
    model_name="HuggingFaceH4/zephyr-7b-alpha",
    tokenizer_name="HuggingFaceH4/zephyr-7b-alpha",
    query_wrapper_prompt=PromptTemplate("<|system|>\n</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"),
    context_window=3900,
    max_new_tokens=256,
    model_kwargs={"quantization_config": quantization_config},
    # tokenizer_kwargs={},
    generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    messages_to_prompt=messages_to_prompt,
    device_map="auto",
)

from llama_index import ServiceContext

service_context = ServiceContext.from_defaults(llm=llm, embed_model="local:BAAI/bge-small-en-v1.5")

# from llama_index import VectorStoreIndex

# vector_index = VectorStoreIndex.from_documents(documents, service_context=service_context)

# from llama_index import SummaryIndex

# summary_index = SummaryIndex.from_documents(documents, service_context=service_context)

from llama_index import display_response

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, select, column

db_user = "cognitus"
db_password = "student"
db_host = "localhost"
db_name = "generative_mapping"


engine = create_engine("mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")

from llama_index import SQLDatabase

sql_database = SQLDatabase(engine)


from llama_index import NLSQLTableQueryEngine

query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["MATERIAL.xlsx", "SALES ORDER.xlsx", "Customer.xlsx"],
    service_context=service_context
)

response = query_engine.query("In Material.xslx, what is the value of the short Description where Table Name is AENAM")

display_response(response)