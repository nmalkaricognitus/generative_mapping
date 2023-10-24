import os
from llama_index.llms import GradientBaseModelLLM
from llama_index.finetuning.gradient.base import GradientFinetuneEngine
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

os.environ["GRADIENT_ACCESS_TOKEN"] = "GELlkbMRDD1t07FvT1xeijuYCpxXCAiT"
os.environ["GRADIENT_WORKSPACE_ID"] = "de8167e3-8275-4a77-a8b2-44a83f7a88dd_workspace"

dialect = "sql"

from datasets import load_dataset
from pathlib import Path
import json


def load_jsonl(data_dir):
    data_path = Path(data_dir).as_posix()
    data = load_dataset("json", data_files=data_path)
    return data


def save_jsonl(data_dicts, out_path):
    with open(out_path, "w") as fp:
        for data_dict in data_dicts:
            fp.write(json.dumps(data_dict) + "\n")


def load_data_sql(data_dir: str = "/home/ubuntu/generative_mapping/generative_mapping/Data/sql_create_context_v4.json"):
    dataset = load_dataset("b-mc2/sql-create-context")

    dataset_splits = {"train": dataset["train"]}
    out_path = Path(data_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    for key, ds in dataset_splits.items():
        with open(out_path, "w") as f:
            for item in ds:
                newitem = {
                    "input": item["question"],
                    "context": item["context"],
                    "output": item["answer"],
                }
                f.write(json.dumps(newitem) + "\n")

load_data_sql(data_dir="/home/ubuntu/generative_mapping/generative_mapping/Data/sql_create_context_v4.json")

from math import ceil


def get_train_val_splits(
    data_dir: str = "/home/ubuntu/generative_mapping/generative_mapping/Data/sql_create_context_v4.json",
    val_ratio: float = 0.1,
    seed: int = 42,
    shuffle: bool = True,
):
    data = load_jsonl(data_dir)
    num_samples = len(data["train"])
    val_set_size = ceil(val_ratio * num_samples)

    train_val = data["train"].train_test_split(
        test_size=val_set_size, shuffle=shuffle, seed=seed
    )
    return train_val["train"].shuffle(), train_val["test"].shuffle()


raw_train_data, raw_val_data = get_train_val_splits(data_dir="/home/ubuntu/generative_mapping/generative_mapping/Data/sql_create_context_v4.json")
save_jsonl(raw_train_data, "train_data_raw.jsonl")
save_jsonl(raw_val_data, "val_data_raw.jsonl")

text_to_sql_tmpl_str = """\
<s>### Instruction:\n{system_message}{user_message}\n\n### Response:\n{response}</s>"""

text_to_sql_inference_tmpl_str = """\
<s>### Instruction:\n{system_message}{user_message}\n\n### Response:\n"""

### Alternative Format
### Recommended by gradient.ai docs, but empirically we found worse results here

# text_to_sql_tmpl_str = """\
# <s>[INST] SYS\n{system_message}\n<</SYS>>\n\n{user_message} [/INST] {response} </s>"""

# text_to_sql_inference_tmpl_str = """\
# <s>[INST] SYS\n{system_message}\n<</SYS>>\n\n{user_message} [/INST] """


def _generate_prompt_sql(input, context, dialect="sqlite", output=""):
    system_message = f"""You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables. 

You must output the SQL query that answers the question.
    
    """
    user_message = f"""### Dialect:
{dialect}

### Input:
{input}

### Context:
{context}

### Response:
"""
    if output:
        return text_to_sql_tmpl_str.format(
            system_message=system_message,
            user_message=user_message,
            response=output,
        )
    else:
        return text_to_sql_inference_tmpl_str.format(
            system_message=system_message, user_message=user_message
        )


def generate_prompt(data_point):
    full_prompt = _generate_prompt_sql(
        data_point["input"],
        data_point["context"],
        dialect="sqlite",
        output=data_point["output"],
    )
    return {"inputs": full_prompt}

train_data = [
    {"inputs": d["inputs"] for d in raw_train_data.map(generate_prompt)}
]
save_jsonl(train_data, "train_data.jsonl")
val_data = [{"inputs": d["inputs"] for d in raw_val_data.map(generate_prompt)}]
save_jsonl(val_data, "val_data.jsonl")

base_model_slug = "llama2-7b-chat"
base_llm = GradientBaseModelLLM(
    base_model_slug=base_model_slug, max_tokens=300
)

finetune_engine = GradientFinetuneEngine(
    base_model_slug=base_model_slug,
    # model_adapter_id='805c6fd6-daa8-4fc8-a509-bebb2f2c1024_model_adapter',
    name="text_to_sql",
    data_path="train_data.jsonl",
    verbose=True,
    max_steps=200,
    batch_size=4,
)

epochs = 1
for i in range(epochs):
    print(f"** EPOCH {i} **")
    finetune_engine.finetune()

ft_llm = finetune_engine.get_finetuned_model(max_tokens=300)

tokenizer = AutoTokenizer.from_pretrained(ft_llm, use_auth_token=True)

pipeline = pipeline(
    "text-generation",
    model=ft_llm,
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
        verbose=True)
bot_response = agent_executor.run("What is the total number of tables present in the database")

print(bot_response)