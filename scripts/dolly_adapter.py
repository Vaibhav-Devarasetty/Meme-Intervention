import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSeq2SeqLM
import os, sys
from pytorchtools import EarlyStopping
from pathlib import Path

#torch.cuda.set_device(1)

device = torch.device("cuda:0") 

df_train = pd.read_csv("train.csv")
df_val = pd.read_csv("val.csv")
df_test = pd.read_csv("test.csv")

model_name = "databricks/dolly-v2-3b"


tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")







class MemesDataset(torch.utils.data.Dataset):
    """Uses jsonl data to preprocess and serve 
    dictionary of multimodal tensors for model input.
    """

    def __init__(
        self,
      	data,
      	tokenizer,
    ):

        self.data = data
        self.tokenizer = tokenizer
       
        
    def __len__(self):
        """This method is called when you do len(instance) 
        for an instance of this class.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """This method is called when you do instance[key] 
        for an instance of this class.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = self.data["img_id"][idx]
        input_text = str(self.data["meme_text"][idx]) + "target_group: " + str(self.data["target_group"][idx]) + "attack_type: " + str(self.data["attack_type"][idx]) + "intervention: " + str(self.data["interventive_content"][idx]) + " " + str(self.data["interventive_filler"][idx])
             
        input_tokenized = self.tokenizer(input_text, return_tensors="pt")

       	if input_tokenized["input_ids"].shape[1]<256:
       		input_ids = torch.cat((input_tokenized["input_ids"],torch.zeros(1,256-input_tokenized["input_ids"].shape[1])),dim=1)
       	else:
       		input_ids = input_tokenized["input_ids"][:,:256]


       	if input_tokenized["attention_mask"].shape[1]<256:
       		attention_mask = torch.cat((input_tokenized["attention_mask"],torch.zeros(1,256-input_tokenized["attention_mask"].shape[1])),dim=1)
       	else:
       		attention_mask = input_tokenized["attention_mask"][:,:256]




        	
       
       
        sample = {
        "img_id":img_id,
        "input_ids": input_ids.flatten().long(),
        "attention_mask": attention_mask.flatten().long(),
        "labels": input_ids.flatten().long(),
        }


        return sample





bm_dataset_train = MemesDataset(df_train,tokenizer)
dataloader_train = DataLoader(bm_dataset_train, batch_size=32,shuffle=False, num_workers=0)

print("train_data loaded")

bm_dataset_val = MemesDataset(df_val,tokenizer)
dataloader_val = DataLoader(bm_dataset_val, batch_size=32,shuffle=False, num_workers=0)
print("validation_data loaded")


bm_dataset_test = MemesDataset(df_test,tokenizer)
dataloader_test = DataLoader(bm_dataset_test, batch_size=32,shuffle=False, num_workers=0)
print("test data loaded")





exp_name = "dolly"
exp_path = "AAAI2024"
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

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)

import os, sys
#sys.path.append('path_to_the_module/early-stopping-pytorch')
from pytorchtools import EarlyStopping
#from torchsample.callbacks import EarlyStopping


# For cross entropy loss
def train_model(model, patience, n_epochs):
    epochs = n_epochs
#     clip = 5

  
    train_loss_list=[]
    val_loss_list=[]
    
        # initialize the experiment path
    Path(exp_path).mkdir(parents=True, exist_ok=True)
    # initialize early_stopping object
    chk_file = os.path.join(exp_path, 'checkpoint_'+exp_name+'.pt')
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=chk_file)


    model.train()
    for i in range(epochs):
#         total_acc_train = 0

      
        #scheduler.step()

        total_loss_train = 0
        total_train = 0
        correct_train = 0

        for data in dataloader_train:

	       	img_id = data["img_id"]
	        input_ids = data["input_ids"].to(device)
	        attention_mask = data["attention_mask"].to(device)
	        labels = data["labels"].to(device)
	        model.zero_grad()

	        output = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
	        loss = output["loss"]
	        loss.backward()
	        optimizer.step()
	        total_loss_train += loss.item()
	        total_train += labels.size(0)

        
        train_loss = total_loss_train/total_train
        model.eval()

        total_loss_val = 0
        total_val = 0
        correct_val = 0

        with torch.no_grad():
            for data in dataloader_val:   

	           	img_id = data["img_id"]
	           	input_ids = data["input_ids"].to(device)
	           	attention_mask = data["attention_mask"].to(device)
	           	labels = data["labels"].to(device)
	           	model.zero_grad()
	           	output = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
	           	val_loss = output["loss"]
	           	total_loss_val += val_loss.item()
	           	total_val += labels.size(0)



     
        val_loss = total_loss_val/total_val

   
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        
        early_stopping(val_loss, model)
        

        if early_stopping.early_stop:
            print("Early stopping")
            break
            

        
        print(f'Epoch {i+1}: train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}')
        model.train()
        torch.cuda.empty_cache()
        
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(chk_file))
    #model.load_state_dict(torch.load(os.path.join(exp_path,"final.pt")))
    
    return  model, train_loss_list, val_loss_list, i
        

n_epochs = 30
 
patience = 10
model, train_loss_list, val_loss_list, epoc_num = train_model(model, patience, n_epochs)

