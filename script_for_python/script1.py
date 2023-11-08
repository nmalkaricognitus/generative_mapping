from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from flask import Flask, request, jsonify

app = Flask(__name__)

model_name_or_path = "TheBloke/CodeLlama-7B-Instruct-GPTQ"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", trust_remote_code=True, revision="main")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

@app.route('/generate', methods=['POST'])
def generate_code():
    data = request.get_json()
    prompt = data.get('prompt', '')
    prompt_template = f'''[INST] {prompt} [/INST]\n'''
    
    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
    generated_code = tokenizer.decode(output[0])
    
    response = {'generated_code': generated_code}
    return jsonify(response)

if __name__ == '__main':
    app.run(debug=True)
