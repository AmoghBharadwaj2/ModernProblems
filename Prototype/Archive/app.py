from flask import Flask, request, render_template
from prototype import generate
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import re
import random
from textblob import TextBlob
app = Flask(__name__)
 

if __name__ == '__main__':
    app.run(debug = True)

@app.route('/', methods=['GET', 'POST'])
def main():
    
    if request.method == "POST":
        
        input = request.form.get("input")
        input = input.split(",")
        question = input[0]
        number = int(input[1])
        prediction = generate(question, number)
        #prediction = "helloworld"
    else:
        prediction = ""
    return render_template("website.html", output = prediction)

