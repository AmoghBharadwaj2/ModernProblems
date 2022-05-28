from unittest.util import _MAX_LENGTH
from transformers import AutoTokenizer, AutoModelForTextGeneration, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
import numpy as np

dataset = load_dataset('scientific_papers', 'pubmed')

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["abstract"], padding='max_length', truncation=True)

tokenized_datasets=dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

model = AutoModelForTextGeneration.from_pretrained('huggingtweets/nature')

model.config.pad_token_id = tokenizer.pad_token_id

training_args = TrainingArguments(output_dir='Training')

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(prediction=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.push_to_hub("fine-tuned science")
