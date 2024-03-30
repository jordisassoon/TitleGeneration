import torch
import transformers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import 

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(f"Using {device}.")


val_df = pd.read_csv("data/validation.csv")

def df_to_loader(df, batch_size):
    
