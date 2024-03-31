
# %%
import torch
from tqdm import tqdm
import pandas as pd

device = torch.device("cuda:1")

tqdm.pandas()

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, create_reference_model
from trl.core import LengthSampler


# %%
from datasets import load_dataset

data_files = {"train": "data/train.csv", "test": "data/validation.csv"}
billsum = load_dataset("csv", data_files=data_files)

# %%
config = PPOConfig(
    model_name="t5_small_train_out",
    # model_name="almanach/camembert-base",
    learning_rate = 1.41e-4,
    remove_unused_columns=False,
    # log_with='wandb',
)

sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply" : "none",
    "batch_size": 16
}

# %%
# import wandb
# wandb.init("t5_ppo")

# %%
from datasets import load_dataset, Dataset

data_files = {"train": "data/train.csv", "test": "data/validation.csv"}
dataset = load_dataset("csv", data_files=data_files)

# %%
dataset['train']

# %%
model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

tokenizer.pad_token = tokenizer.eos_token

# %%
def build_dataset(config, input_min_text_length=2):

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    ds = load_dataset("csv", data_files=data_files)
    

    input_size = LengthSampler(input_min_text_length, 512)

    def tokenize(sample):
        if 't5' in config.model_name : sample['text'] = "summarize: " + sample['text'] 
        sample["input_ids"] = tokenizer.encode(sample["text"], max_length=512, truncation=True)
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds

# %%
dataset = build_dataset(config)

# %%
dataset['train']

# %%
def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use the first GPU


ref_model = create_reference_model(model)
ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset['train'], data_collator=collator)

# %%
from rouge_score import rouge_scorer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

NLI_CONFIG = "cmarkea/distilcamembert-base-nli"
# NLI_CONFIG = "mtheo/camembert-base-xnli"

nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_CONFIG).to(device)
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_CONFIG) 

def get_rewards(title, pred, use_entailment=False, text=None, nli_model=None, nli_tokenizer=None):
    
    entailment = 0
    scorer = rouge_scorer.RougeScorer(['rougeL'])
    rouge_l_f1 = scorer.score(pred, title)['rougeL'][2]
    wt = 1

    if use_entailment:
        nli_model.eval()
        with torch.no_grad():
            text = " ".join(text.split(" ")[:512])
            x = nli_tokenizer.encode(text, pred, return_tensors='pt', max_length=512).to(device)
            # print(x.shape)
            logits = nli_model(x)[0]
            entail_contradiction_logits = logits[:,::2]
            probs = entail_contradiction_logits.softmax(dim=1)
            entailment = probs[:,0].detach().cpu().numpy()[0]
        wt = 2
        
    return (rouge_l_f1 + entailment)/wt

# """
# Expected Behaviour :

text = "This is a relatively longer text about reference vs predicted titles. Only to see if the computation of entailment changes things significantly."
title = "This is the reference title"
pred = "This is the predicted title"

rouge_l_f1_score = get_rewards(title, pred)
rouge_entailment_score = get_rewards(title, pred, True, text, nli_model, nli_tokenizer)

print("Rouge-L F1 Score:", rouge_l_f1_score)

print("Rouge-L F1 + Entailment Score:", rouge_entailment_score)

# OUTPUT:
# Rouge-L F1 Score: 0.8000000000000002
# Rouge-L F1 + Entailment Score: 0.4215694475919009
# """

print()
    

# %%
for ep, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    gen = ppo_trainer.generate(batch['input_ids'])
    print(batch.keys())
    print(tokenizer.batch_decode(gen))
    break

# %%

generation_kwargs = {
    "min_length": -1,
    "max_length": 50,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}

flag = 0
for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    #### Get response from gpt2
    # response_tensors = []
    # for query in query_tensors:
    #     gen_len = output_length_sampler()
    #     generation_kwargs["max_new_tokens"] = gen_len
    #     response = ppo_trainer.generate(query, **generation_kwargs)
    #     response_tensors.append(response.squeeze()[-gen_len:])
    
    response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
    # batch["response"] = [tokenizer.decode(r.squeeze())[0] for r in response_tensors]
    
    batch['response'] = tokenizer.batch_decode(response_tensors)
    

    #### Compute sentiment score
    # texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    # pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    # rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
    if not flag:
        print(batch['titles'])
        flag = 1
    if batch['response'] and batch['titles']:
        rewards = [torch.tensor(get_rewards(pred, title, True, text, nli_model, nli_tokenizer)) for text, title, pred in zip(batch['text'], batch['response'], batch['titles'])]
    else: rewards = []
    #### Run PPO step
    try:
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
    except Exception as e:
        print(e)
    # print(stats, batch, rewards)

# %%
model.save_pretrained("t5_ppo")
tokenizer.save_pretrained("t5_ppo")

# %%
# #### get a batch from the dataset
# bs = 16
# game_data = dict()
# dataset.set_format("pandas")
# df_batch = dataset[:].sample(bs)
# game_data["query"] = df_batch["query"].tolist()
# query_tensors = df_batch["input_ids"].tolist()

# response_tensors_ref, response_tensors = [], []

# #### get response from gpt2 and gpt2_ref
# for i in range(bs):
#     gen_len = output_length_sampler()
#     output = ref_model.generate(
#         torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(device), max_new_tokens=gen_len, **gen_kwargs
#     ).squeeze()[-gen_len:]
#     response_tensors_ref.append(output)
#     output = model.generate(
#         torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(device), max_new_tokens=gen_len, **gen_kwargs
#     ).squeeze()[-gen_len:]
#     response_tensors.append(output)

# #### decode responses
# game_data["response (before)"] = [tokenizer.decode(response_tensors_ref[i]) for i in range(bs)]
# game_data["response (after)"] = [tokenizer.decode(response_tensors[i]) for i in range(bs)]

# #### sentiment analysis of query/response pairs before/after
# texts = [q + r for q, r in zip(game_data["query"], game_data["response (before)"])]
# game_data["rewards (before)"] = [output[1]["score"] for output in sentiment_pipe(texts, **sent_kwargs)]

# texts = [q + r for q, r in zip(game_data["query"], game_data["response (after)"])]
# game_data["rewards (after)"] = [output[1]["score"] for output in sentiment_pipe(texts, **sent_kwargs)]

# # store results in a dataframe
# df_results = pd.DataFrame(game_data)
# df_results


