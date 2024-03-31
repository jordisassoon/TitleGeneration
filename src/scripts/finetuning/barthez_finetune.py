# %%

## > cheesy.out 2>&1 &

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# from utils.utils import loader_from_csv
# from utils.metrics import rouge_l_fscore

# Load the model
CONFIG = "moussaKam/barthez"
SAVE_PATH = f"{CONFIG.split('/')[-1]}_finetuned_early"


print(f"Finetuning model at {CONFIG}")
print(f"Final model will be saved at {SAVE_PATH}")


# %%
from datasets import load_dataset

data_files = {"train": "data/train.csv", "test": "data/validation.csv"}
billsum = load_dataset("csv", data_files=data_files)

# %%
billsum

# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(CONFIG)

# %%
def preprocess_function(examples):
    # Prepends the string "summarize: " to each document in the 'text' field of the input examples.
    # This is done to instruct the T5 model on the task it needs to perform, which in this case is summarization.
    # inputs = ["summarize: " + doc for doc in examples["text"]]

    # Tokenizes the prepended input texts to convert them into a format that can be fed into the T5 model.
    # Sets a maximum token length of 1024, and truncates any text longer than this limit.
    model_inputs = tokenizer(examples['text'], max_length=512, truncation=True)

    # Tokenizes the 'summary' field of the input examples to prepare the target labels for the summarization task.
    # Sets a maximum token length of 128, and truncates any text longer than this limit.
    labels = tokenizer(text_target=examples["titles"], max_length=128, truncation=True)

    # Assigns the tokenized labels to the 'labels' field of model_inputs.
    # The 'labels' field is used during training to calculate the loss and guide model learning.
    model_inputs["labels"] = labels["input_ids"]

    # Returns the prepared inputs and labels as a single dictionary, ready for training.
    return model_inputs

# %%
tokenized_billsum = billsum.map(preprocess_function, batched=True)

# %%
tokenized_billsum['test'][0]['text']

# %%
tokenized_billsum['test'][0]['titles']

# %%
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=CONFIG)

# %%
import evaluate

rouge = evaluate.load("rouge")

# %%
import numpy as np

def compute_metrics(eval_pred):
    # Unpacks the evaluation predictions tuple into predictions and labels.
    predictions, labels = eval_pred

    # Decodes the tokenized predictions back to text, skipping any special tokens (e.g., padding tokens).
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replaces any -100 values in labels with the tokenizer's pad_token_id.
    # This is done because -100 is often used to ignore certain tokens when calculating the loss during training.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decodes the tokenized labels back to text, skipping any special tokens (e.g., padding tokens).
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Computes the ROUGE metric between the decoded predictions and decoded labels.
    # The use_stemmer parameter enables stemming, which reduces words to their root form before comparison.
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    # Calculates the length of each prediction by counting the non-padding tokens.
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]

    # Computes the mean length of the predictions and adds it to the result dictionary under the key "gen_len".
    result["gen_len"] = np.mean(prediction_lens)

    # Rounds each value in the result dictionary to 4 decimal places for cleaner output, and returns the result.
    return {k: round(v, 4) for k, v in result.items()}


# %%
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

# %%
model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG)
model.config.pad_token_id = tokenizer.pad_token_id



# %%
training_args = Seq2SeqTrainingArguments(
    output_dir=SAVE_PATH,
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=5,
    predict_with_generate=True,
    fp16=True,
)

# %%
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_billsum["train"],
    eval_dataset=tokenized_billsum["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# %%
trainer.train()

# %%
trainer.save_model(SAVE_PATH)

training_args.num_train_epochs = 1
trainer2 = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_billsum["test"],
    eval_dataset=tokenized_billsum["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer2.train()
trainer2.save_model(SAVE_PATH)


texts = billsum['test'][:]['text']
texts = ["summarize: " + text for text in texts]

from transformers import pipeline

summarizer = pipeline("summarization", model=CONFIG, max_length=100)
preds = summarizer(texts)

preds_ = [pred["summary_text"] for pred in preds]

labels = [[x] for x in billsum['test'][:]['titles']]

results = rouge.compute(predictions=preds_, references=labels)

print(results)