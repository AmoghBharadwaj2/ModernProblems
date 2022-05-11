
#Using model from https://huggingface.co/huggingtweets/elonmusk?text=My+dream+is
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import re
import random

#generator for religion (NASA)
science = pipeline('text-generation',
                     model='huggingtweets/nasa')
#generator for philosophy (philosophy_mark, Professor of philosophy)
philosophy = pipeline('text-generation',
                     model='huggingtweets/philosophy_mark')

#generator for religion (buddha_quotes)
religion = pipeline('text-generation',
                      model='huggingtweets/_buddha_quotes')

#Type your input questions
starter = input("Enter your Question: ")
#total number of conversations
num = input("Enter the number of conversations you want: ")
i = 0

pattern = re.compile(r'[a-zA-Z]+')

defaultsentence = "I don't know"

while i < int(num):
    #generate science response 
    response = science(starter, num_return_sequences=1)
    starter = response[0]['generated_text']
    starter =  " ".join(starter.split())
    print("Science: " + starter + "\n")
    starter = response[0]['generated_text']
    #handle input for next model?
    temp = pattern.findall(starter)
    if len(temp) == 0:
        starter = defaultsentence
    else:
        random_index = random.randint(0,len(temp)-1)
        starter = temp[random_index]
    
    #generate religion response
    response = religion(starter, num_return_sequences=1)
    starter = response[0]['generated_text']
    starter =  " ".join(starter.split())
    print("Religion: " + starter + "\n")
    starter = response[0]['generated_text']
    #handle input for next model?
    temp = pattern.findall(starter)
    if len(temp) == 0:
        starter = defaultsentence
    else:
        random_index = random.randint(0,len(temp)-1)
        starter = temp[random_index]

    #generate philosophy response
    response = philosophy(starter, num_return_sequences=1)
    starter = response[0]['generated_text']
    starter =  " ".join(starter.split())
    print("Philosophy: " + starter + "\n")
    starter = response[0]['generated_text']
    #handle input for next model?
    temp = pattern.findall(starter)
    if len(temp) == 0:
        starter = defaultsentence
    else:
        random_index = random.randint(0,len(temp)-1)
        starter = temp[random_index]
  
    i += 1


