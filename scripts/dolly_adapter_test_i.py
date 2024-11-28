import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSeq2SeqLM
import os, sys
from pytorchtools import EarlyStopping
from pathlib import Path
from transformers import pipeline

#torch.cuda.set_device(1)

device = torch.device("cuda:0") 

df_train = pd.read_csv("train.csv")
df_val = pd.read_csv("val.csv")
df_test = pd.read_csv("test.csv")

#model_name = "TinyPixel/Llama-2-7B-bf16-sharded"

model_name = "databricks/dolly-v2-3b"


tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")








exp_name = "dolly_i"
exp_path = "vaibhav"
#exp_path = "path_to_saved_files/Sarcasm_ModelCkpt/"+exp_name
lr=0.0001


quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

#model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map="auto")
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config)


model.gradient_checkpointing_enable()


from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    #target_modules=["query_key_value"],
    #target_modules = ["q_proj", "v_proj"],
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)



model.to(device)

print(model)
model.config.use_cache = False


chk_file = os.path.join(exp_path, 'checkpoint_'+exp_name+'.pt')


model.load_state_dict(torch.load(chk_file))





gen = pipeline("text-generation",model=model,tokenizer=tokenizer,torch_dtype=torch.bfloat16, trust_remote_code=True,device=device,temperature=0.8,max_length=700,return_full_text=False,eos_token_id=[int(tokenizer.convert_tokens_to_ids('###'))])




img_id = []
meme_text = []
gen_response = []

responselist=[]

for i in range(len(df_test)):

    img_path = str(df_test["img_id"][i])
    OCR_text = str(df_test["meme_text"][i])

    print('**************$$$$$$$$$$####################')

    response=gen(OCR_text)
    print(response[0]["generated_text"])
    response = response[0]["generated_text"]
    print('**************$$$$$$$$$$####################')
    

    img_id.append(img_path)
    meme_text.append(OCR_text)
    gen_response.append(response)
    
    
    
   
data = {}
 
data["img_id"] = img_id
data["meme_text"] = meme_text
data["response"] = gen_response
 
df_generated = pd.DataFrame(data)
    
df_generated.to_csv(f"generated_data_{exp_name}.csv")    
  



    
