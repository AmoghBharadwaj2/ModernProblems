from datasets import load_dataset

dataset = load_dataset('scientific_papers', 'pubmed')

print(dataset["train"]["features"])
