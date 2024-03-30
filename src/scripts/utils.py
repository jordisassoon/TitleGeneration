import pandas as pd
from torch.utils.data import Dataset, DataLoader

class DFSet(Dataset):
    def __init__(self, df, text_col='text', title_col='titles'):
        self.inputs = df[text_col].tolist()
        self.labels = df[title_col].tolist()
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        label = self.labels[idx]
        return input_text, label

def loader_from_df(df, batch_size, text_col='text', title_col='titles'):
    dataset = DFSet(df)
    dataloader = DataLoader(dataset, batch_size, text_col, title_col, shuffle=True)
    return dataloader

def df_from_csv(filename):
    return pd.read_csv(filename)

def loader_from_csv(filename, text_col='text', title_col='titles'):
    return loader_from_df(df_from_csv(filename), text_col, title_col)

def generate_summary(text, tokenizer, model, device):
    inputs = tokenizer([text], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    output = model.generate(input_ids, attention_mask=attention_mask)
    return tokenizer.batch_decode(output[0], skip_special_tokens=True)

def predict_headlines(articles, tokenizer, model, device):   
    inputs = tokenizer(articles, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    output = model.generate(input_ids, attention_mask=attention_mask)
    return tokenizer.batch_decode(output, skip_special_tokens=True)