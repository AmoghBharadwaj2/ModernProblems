
#Using model from https://huggingface.co/huggingtweets/elonmusk?text=My+dream+is
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import re
import random
from textblob import TextBlob
def generate(question, nums):
    #generator for science (Nature)
    science = pipeline('text-generation',
                        model='huggingtweets/nature')
    #generator for philosophy (philosophy_mark, Professor of philosophy)
    philosophy = pipeline('text-generation',
                        model='huggingtweets/politicalmiller')

    #generator for religion (buddha_quotes)
    religion = pipeline('text-generation',
                        model='huggingtweets/_buddha_quotes')

    #Type your input questions
    #starter = input("Enter your Question: ")
    #total number of conversations
    starter = question
    #num = input("Enter the number of conversations you want: ")
    num = nums
    i = 0
    #save result of models
    result = []
    pattern = re.compile(r'[a-zA-Z]+')

    defaultsentence = "I don't know"

    while i < int(num):
        #generate science response 
        response = science(starter, num_return_sequences=1)
        starter = response[0]['generated_text']
        starter =  " ".join(starter.split())
        print("Science: " + starter + "\n")
        result.append("Science: " + starter)
        starter = response[0]['generated_text']
        #handle input for next model?
        blob = TextBlob(starter)
        temp = blob.noun_phrases
        #temp = pattern.findall(starter)
        if len(temp) == 0:
            starter = defaultsentence
        else:
            random_index = random.randint(0,len(temp)-1)
            starter = temp[random_index] + " is"
        
        #generate religion response
        response = religion(starter, num_return_sequences=1)
        starter = response[0]['generated_text']
        starter =  " ".join(starter.split())
        print("Religion: " + starter + "\n")
        result.append("Religion: " + starter)
        starter = response[0]['generated_text']
        #handle input for next model?
        blob = TextBlob(starter)
        temp = blob.noun_phrases
        #temp = pattern.findall(starter)
        if len(temp) == 0:
            starter = defaultsentence
        else:
            random_index = random.randint(0,len(temp)-1)
            starter = temp[random_index] + " is"

        #generate philosophy response
        response = philosophy(starter, num_return_sequences=1)
        starter = response[0]['generated_text']
        starter =  " ".join(starter.split())
        print("Philosophy: " + starter + "\n")
        result.append("Philosophy: " + starter)
        starter = response[0]['generated_text']
        blob = TextBlob(starter)
        temp = blob.noun_phrases
        #temp = pattern.findall(starter)
        if len(temp) == 0:
            starter = defaultsentence
        else:
            random_index = random.randint(0,len(temp)-1)
            starter = temp[random_index] + " is"
    
        i += 1
    return result

