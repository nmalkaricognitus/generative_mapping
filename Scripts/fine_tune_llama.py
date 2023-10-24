import torch
import transformers
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from my_dataset import SpiderDataset 
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from huggingface_hub import login



dataset = SpiderDataset('/Users/nikhilmalkari/Documents/Cognitus/GenerativeMapping/generative_mapping/spider')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = "openlm-research/open_llama_7b_v2" 

tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=True)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()


for epoch in range(3):
    for batch in dataloader:
        inputs = tokenizer(batch['text'], padding=True, truncation=True, return_tensors='pt')
        outputs = model(**inputs, labels=batch['labels'])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')


model.save_pretrained('path_to_save_model')

