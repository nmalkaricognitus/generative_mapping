import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
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
from langchain_experimental.sql import SQLDatabaseChain

## v2 models
#model_path = 'openlm-research/open_llama_3b_v2'
model_path = 'openlm-research/open_llama_7b_v2'

## v1 models
# model_path = 'openlm-research/open_llama_3b'
# model_path = 'openlm-research/open_llama_7b'
# model_path = 'openlm-research/open_llama_13b'

tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map='auto',
)

# prompt = 'Q: What is the largest animal?\nA:'
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# generation_output = model.generate(
#     input_ids=input_ids, max_new_tokens=32
# )
# print(tokenizer.decode(generation_output[0]))

