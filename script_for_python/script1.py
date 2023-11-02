from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name_or_path = "TheBloke/CodeLlama-7B-Instruct-GPTQ"

model = AutoModelForCausalLM.from_pretrained(model_name_or_path,device_map="auto",trust_remote_code=True,revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

prompt = "Write a python code where it takes a dataframe and multiples all the numerical columns by two and update the dataframe"
prompt_template=f'''[INST] Write a python code where it takes a dataframe and multiples all the numerical columns by two and update the dataframe. Please wrap your code answer using ```:
{prompt}
[/INST]

'''

print("\n\n*** Generate:")

input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
print(tokenizer.decode(output[0]))


print("*** Pipeline:")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1
)

print(pipe(prompt_template)[0]['generated_text'])
