from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
# Assume llama7b is the module for Llama 7B
from llama7b import Llama7B

db_user = "cognitus"
db_password = "student"
db_host = "localhost"
db_name = "generative_mapping"
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")


llm = Llama7B()  # Assume Llama7B() initializes a Llama 7B model
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

# Run a SQL query using natural language prompt
db_chain.run("How many tables are there ?")
