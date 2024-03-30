import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the T5 model
model_path = 't5_small_train_out'
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

# Load the test texts
test_texts_path = '/path/to/data/test_texts.csv'
test_texts = pd.read_csv(test_texts_path)

# Generate summaries for each text
summaries = []
for text in test_texts['text']:
    # Tokenize the input text
    input_ids = tokenizer.encode(text, return_tensors='pt')

    # Generate the summary
    summary_ids = model.generate(input_ids)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    summaries.append(summary)

# Save the summaries to a file
output_path = '/path/to/output/summaries.txt'
with open(output_path, 'w') as f:
    for summary in summaries:
        f.write(summary + '\n')

print('Summaries generated and saved to', output_path)