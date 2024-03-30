import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import torch
import sys

def summarize_df(df):
    summaries = []
    for text in tqdm(df['text']):
        input_ids = tokenizer.encode("summarize: "+text, return_tensors='pt', max_length=512, truncation=True).to(device)
        summary_ids = model.generate(input_ids, max_length = 50)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    pred_df = pd.DataFrame({'text':df['text'], 'titles':df['titles'], 'pred_title':summaries})
    return pred_df

def rouge_from_df(df, gt_col='titles', pred_col='pred_title'):
    scorer = rouge_scorer.RougeScorer(['rougeL'])
    rouges = []
    titles = list(df[gt_col])
    preds = list(df[pred_col])
    for title, pred in zip(titles, preds):
        rouges.append(scorer.score(pred, title)['rougeL'][2])
    return np.mean(rouges)

if __name__=="__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_path = sys.argv[1]

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    val_df = pd.read_csv("data/validation.csv")
    print(rouge_from_df(summarize_df(val_df)))

