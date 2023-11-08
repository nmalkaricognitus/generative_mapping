from transformers import AutoModelForCausalLM, AutoTokenizer

model_name_or_path = "TheBloke/CodeLlama-7B-Instruct-GPTQ"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", trust_remote_code=True, revision="main")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

def generate_code(prompt):
    prompt_template = f'''[INST] {prompt} [/INST]\n'''
    
    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
    generated_code = tokenizer.decode(output[0])
    
    return generated_code

if __name__ == '__main__':
    prompt = "write a python code to multply all the columns by 2 in the dataframe"
    generated_code = generate_code(prompt)
    print("Generated Code:")
    print(generated_code)
