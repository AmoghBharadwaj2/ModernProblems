from transformers import GPT2Tokenizer, GPT2Model, pipeline
token = GPT2Tokenizer.from_pretrained('gpt2')
mod = GPT2Model.from_pretrained('gpt2')
text = "Replace me by any text you'd like."
pipe = pipeline(model=mod, tokenizer=token)
pipe(text, num_return_sequences=5, max_length=30)
