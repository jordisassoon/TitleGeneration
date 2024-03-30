# %%
import torch
from tqdm import tqdm
import pandas as pd

tqdm.pandas()

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, create_reference_model
from trl.core import LengthSampler

from utils.utils import loader_from_csv

# %%
from datasets import load_dataset

data_files = {"train": "data/train.csv", "test": "data/validation.csv"}
billsum = load_dataset("csv", data_files=data_files)

# %%
config = PPOConfig(
    model_name="t5_small_train_out",
    # model_name="almanach/camembert-base",
    learning_rate = 1.41e-2,
    remove_unused_columns=False,
    log_with='wandb',
    batch_size=6,
    mini_batch_size=6,

)

sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply" : "none",
    "batch_size": 4
}

# %%
import wandb
wandb.init("t5_ppo")

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
def build_dataset(config, dataset_name="imdb", input_min_text_length=2, input_max_text_length=8):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # load imdb with datasets
    # ds = load_dataset(dataset_name, split="train")
    ds = load_dataset("csv", data_files=data_files)
    # ds = ds.rename_columns({"text": "review"})

    input_size = LengthSampler(input_min_text_length, 512)

    def tokenize(sample):
        # sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        if 't5' in config.model_name : sample['text'] = "summarize: " + sample['text'] 
        sample["input_ids"] = tokenizer.encode(sample["text"], max_length=512, padding=True)
        sample["query"] = tokenizer.decode(sample["input_ids"])
        # sample["text"] = sample["text"]
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use the first GPU


ref_model = create_reference_model(model)
ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset['train'], data_collator=collator)

# %%
from rouge_score import rouge_scorer
def get_rouge_reward(title, pred):
    # print("Title : ", title)
    # print("Pred : ", pred)
    scorer = rouge_scorer.RougeScorer(['rougeL'])
    rouge_l_f1 = scorer.score(pred, title)['rougeL'][2]
    return rouge_l_f1

title = "This is the reference title"
pred = "This is the predicted title"

rouge_l_f1_score = get_rouge_reward(title, pred)
print("Rouge-L F1 Score:", rouge_l_f1_score)
    

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
        rewards = [torch.tensor(get_rouge_reward(pred, title)) for title, pred in zip(batch['response'], batch['titles'])]
    else: rewards = []
    #### Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)
    # print(stats, batch, rewards)

# %%
model.save_pretrained("t5_ppo")
tokenizer.save_pretrained("t5_ppo")

# %%
#### get a batch from the dataset
bs = 16
game_data = dict()
dataset.set_format("pandas")
df_batch = dataset[:].sample(bs)
game_data["query"] = df_batch["query"].tolist()
query_tensors = df_batch["input_ids"].tolist()

response_tensors_ref, response_tensors = [], []

#### get response from gpt2 and gpt2_ref
for i in range(bs):
    gen_len = output_length_sampler()
    output = ref_model.generate(
        torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(device), max_new_tokens=gen_len, **gen_kwargs
    ).squeeze()[-gen_len:]
    response_tensors_ref.append(output)
    output = model.generate(
        torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(device), max_new_tokens=gen_len, **gen_kwargs
    ).squeeze()[-gen_len:]
    response_tensors.append(output)

#### decode responses
game_data["response (before)"] = [tokenizer.decode(response_tensors_ref[i]) for i in range(bs)]
game_data["response (after)"] = [tokenizer.decode(response_tensors[i]) for i in range(bs)]

#### sentiment analysis of query/response pairs before/after
texts = [q + r for q, r in zip(game_data["query"], game_data["response (before)"])]
game_data["rewards (before)"] = [output[1]["score"] for output in sentiment_pipe(texts, **sent_kwargs)]

texts = [q + r for q, r in zip(game_data["query"], game_data["response (after)"])]
game_data["rewards (after)"] = [output[1]["score"] for output in sentiment_pipe(texts, **sent_kwargs)]

# store results in a dataframe
df_results = pd.DataFrame(game_data)
df_results


